# VDBImage fix for using Image encoder is based on changes in 
# https://github.com/akanimax/Variational_Discriminator_Bottleneck/blob/26a39ddbf9ee2213dbc1b60894a9092b1a5d3710/source/vdb/Gan_networks.py#L126


from typing import Any, Dict, Optional, Type, Union, List, Callable, Tuple

from collections import OrderedDict
from numpy.lib.type_check import real
import gym
from gym import spaces
import numpy as np
import wandb
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from einops import rearrange
from stable_baselines3.common.utils import (
    safe_mean,
    update_learning_rate,
    configure_logger,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from util import kl_divergence, normalize_observations, get_obs_actions, eval_policy, save_model, save_model_checkpoint


# https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    SB3 function
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return abs(progress_remaining) * initial_value

    return func


def calc_gail_loss(gail_loss, discrim_criterion, real_data, fake_data, num_samples, device):
    # https://github.com/ASzot/rl-toolkit/blob/master/rlf/algos/il/gaifo.py
    if gail_loss == "gailfo":
        gail_loss = discrim_criterion(
                fake_data, th.ones((num_samples, 1), device=device)
                ) + discrim_criterion(
                real_data, th.zeros((num_samples, 1), device=device)
                )
    elif gail_loss == "gail":
        gail_loss = discrim_criterion(
                fake_data, th.zeros((num_samples, 1), device=device)
                ) + discrim_criterion(
                real_data, th.ones((num_samples, 1), device=device)
                )
    elif gail_loss == "agent":
        gail_loss = discrim_criterion(
                fake_data, th.ones((num_samples, 1), device=device)
                )

    return gail_loss


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size, device):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)
        self.to(device)

    def forward(self, x):
        x = th.tanh(self.linear1(x))
        x = th.tanh(self.linear2(x))
        prob = th.sigmoid(self.linear3(x))
        return prob


class DiscriminatorImage(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        hidden_size: int = 256,
        device: th.device = th.device("cpu"),
    ):
        super(DiscriminatorImage, self).__init__()

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor = th.as_tensor(observation_space.sample()[None])
            # tensor = tensor.permute(0, 3, 1, 2)
            n_flatten = self.cnn(tensor.float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, hidden_size), nn.ReLU())

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

        self.to(device)

    def forward(self, x):
        # x should be B x C x H x W
        x = th.tanh(self.linear(self.cnn(x)))
        x = th.tanh(self.linear2(x))
        prob = th.sigmoid(self.linear3(x))
        return prob


class GAIL(BaseAlgorithm):
    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        discriminator: Union[Discriminator, DiscriminatorImage],
        expert_traj: th.tensor,
        policy_base: Type[BasePolicy] = None,
        normalize: bool = False,
        learning_rate: Union[float, Schedule] = 3e-4,
        lr_schedule_main: bool = False,
        lr_schedule_agent: bool = False,
        ent_decay: float = 1.0,
        norm_env: Optional[Union[GymEnv, str]] = None,
        obs_only: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        tune: bool = False,
        verbose: int = 0,
        support_multi_env: bool = True,
        device: Union[th.device, str] = "auto",
        create_eval_env: bool = False,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        env_name: str = None,
        gpu_optimize: bool = False,
        gail_loss: str = "gailfo",
    ):
        super(GAIL, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_base=policy_base,
            support_multi_env=support_multi_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.discriminator = discriminator
        self.expert_traj = expert_traj
        self.device = device
        self.normalize = normalize
        self.norm_env = norm_env
        self.name = "GAIL"
        self.lr_schedule_main = lr_schedule_main
        self.lr_schedule_agent = lr_schedule_agent
        self.ent_decay = ent_decay
        if self.lr_schedule_main:
            self.learning_rate = linear_schedule(self.learning_rate)
        if self.lr_schedule_agent:
            policy_kwargs.update(
                {"learning_rate": linear_schedule(policy_kwargs["learning_rate"])}
            )
        self.policy = policy(device=self.device, seed=seed, **policy_kwargs)
        self.ts = self.policy.n_steps * self.policy.n_envs
        self.obs_only = obs_only
        self.tune = tune
        self.verbose = verbose
        self.seed = seed
        self.env_name = env_name
        self.gpu_optimize = gpu_optimize
        self.gail_loss = gail_loss

        if _init_setup_model:
            self._setup_model()

    def get_generator_batch(self, num_samples):
        """shuffle, and get a batch of num_samples of state, action tensor from
        rollout buffer
        """
        return self.policy.rollout_buffer.sample(num_samples)

    def _update_lr(
        self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]
    ) -> None:
        # Log the current learning rate
        self.policy.logger.record(
            "gail/train/learning_rate",
            self.lr_schedule(self._current_progress_remaining),
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining)
            )

    def train_discriminator(self, num_samples):
        self._update_lr(self.optimizer)
        expert_state_action = self.expert_traj[
            np.random.randint(0, self.expert_traj.shape[0], num_samples), :
        ]
        expert_state_action = th.FloatTensor(expert_state_action)

        if self.normalize:
            expert_state_action = normalize_observations(
                expert_state_action, self.norm_env, self.obs_only
            )
        expert_state_action = expert_state_action.to(self.device)

        num_samples = expert_state_action.shape[0]

        state_action = self.get_generator_batch(num_samples)
        state_action = get_obs_actions(
            state_action, self.env, self.device, self.obs_only
        )

        fake_data = self.discriminator(state_action)
        real_data = self.discriminator(expert_state_action)

        self.optimizer.zero_grad()
        discrim_loss = calc_gail_loss(self.gail_loss, self.discrim_criterion, real_data, fake_data, num_samples, self.device)
        discrim_loss.backward()
        self.optimizer.step()

        expert_acc = ((self.discriminator(expert_state_action)[0] < 0.5).float()).mean()
        learner_acc = ((self.discriminator(state_action)[0] > 0.5).float()).mean()

        self.policy.logger.record(
            "gail/train/logits_gen",
            fake_data.detach(),
            exclude=("stdout", "json", "csv"),
        )
        self.policy.logger.record(
            "gail/train/logits_expert",
            real_data.detach(),
            exclude=("stdout", "json", "csv"),
        )

        # reduce GPU memory usage
        if self.gpu_optimize:
            del expert_state_action
            del state_action
            th.cuda.empty_cache()

        return discrim_loss, expert_acc, learner_acc, fake_data.mean(), real_data.mean()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()

        self.optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.lr_schedule(1)
        )
        self.discrim_criterion = nn.BCELoss()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        callback_ppo: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "GAILAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_model_interval: int = 100,
        run_id="",
        cuda_id="",
    ) -> "GAIL":

        iteration = 0

        # default logger does not work anymore
        new_logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name)
        self.set_logger(new_logger)
        self.policy.set_logger(new_logger)

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        # expert_acc_lst, learner_acc_lst = [], []
        ep_mean_rewards = []

        while self.num_timesteps < total_timesteps:

            if iteration % 3 == 0:
                self.policy.learn(
                    total_timesteps=self.policy.n_steps,
                    reset_num_timesteps=reset_num_timesteps,
                    callback=callback_ppo,
                )
                reset_num_timesteps = False

            iteration += 1
            self.num_timesteps += 1

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if self.policy.ep_info_buffer is not None:
                mean_rew = safe_mean(
                    [ep_info["r"] for ep_info in self.policy.ep_info_buffer]
                )
                ep_mean_rewards.append([self.policy.num_timesteps, mean_rew])

                if self.tune:
                    wandb.log({"ep_rew_mean": mean_rew})

            (
                dloss,
                expert_acc,
                learner_acc,
                fake_mean,
                real_mean,
            ) = self.train_discriminator(self.ts)

            # expert_acc_lst.append(expert_acc.item())
            # learner_acc_lst.append(learner_acc.item())

            # Display training infos
            self.policy.logger.record("time/il-iterations", iteration)
            self.policy.logger.record_mean("gail/train/gail_loss", dloss.item())
            self.policy.logger.record("gail/train/expert_acc", expert_acc.item())
            self.policy.logger.record("gail/train/learner_acc", learner_acc.item())
            self.policy.logger.record("gail/train/fake_mean", fake_mean.item())
            self.policy.logger.record("gail/train/real mean", real_mean.item())
            self.policy.logger.record(
                "gail/train/rollout buffer", self.policy.rollout_buffer.size()
            )
            self.policy.logger.record("train/ent_coef", self.policy.ent_coef)

            callback.on_training_end()

            self.policy.ent_coef = self.policy.ent_coef * self.ent_decay

            # if not self.tune and iteration % save_model_interval == 0:

            if iteration % save_model_interval == 0:
                save_model(self, np.array(ep_mean_rewards), new_logger.get_dir(), self.env_name, self.seed,run_id,cuda_id)
                # save_model_checkpoint(self.name, self.vqvae.state_dict(), iteration, self.policy.num_timesteps, new_logger.get_dir(), self.env_name, self.seed,run_id,cuda_id)

        return self, np.array(ep_mean_rewards), new_logger.get_dir()


class VDB(nn.Module):
    def __init__(self, num_inputs, hidden_size, latent_size, device):
        super(VDB, self).__init__()

        self.vae_encoder_input = nn.Linear(num_inputs, hidden_size)
        self.vae_encoder_mu = nn.Linear(hidden_size, latent_size)
        self.vae_encoder_logvar = nn.Linear(hidden_size, latent_size)

        self.discrim_input = nn.Linear(latent_size, hidden_size)
        self.discrim_hidden = nn.Linear(hidden_size, hidden_size)
        self.discrim_output = nn.Linear(hidden_size, 1)

        self.discrim_output.weight.data.mul_(0.1)
        self.discrim_output.bias.data.mul_(0.0)

        self.to(device)

    def vae_encoder(self, x):
        h = th.tanh(self.vae_encoder_input(x))
        return self.vae_encoder_mu(h), self.vae_encoder_logvar(h)

    def reparameterize(self, mu, logvar):
        std = th.exp(logvar / 2)
        eps = th.randn_like(std)
        return mu + std * eps

    def discriminator(self, z):
        z = th.tanh(self.discrim_input(z))
        z = th.tanh(self.discrim_hidden(z))
        prob = th.sigmoid(self.discrim_output(z))
        return prob

    def forward(self, x):
        mu, logvar = self.vae_encoder(x)
        z = self.reparameterize(mu, logvar)
        prob = self.discriminator(z)
        return prob, mu, logvar


class VDBImage(nn.Module):
    def __init__(
        self, observation_space: gym.spaces.Box, hidden_size, latent_size, device
    ):
        super(VDBImage, self).__init__()

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor = th.as_tensor(observation_space.sample()[None])
            # tensor = tensor.permute(0, 3, 1, 2)
            cnn_dim = self.cnn(tensor.float()).shape[1]

        # self.vae_encoder_input = nn.Sequential(
        #     nn.Linear(n_flatten, hidden_size), nn.ReLU()
        # )
        # self.vae_encoder_mu = nn.Linear(hidden_size, latent_size)
        # self.vae_encoder_logvar = nn.Linear(hidden_size, latent_size)

        self.vae_encoder_mu = nn.Conv2d(cnn_dim, latent_size, 1)
        self.vae_encoder_logvar = nn.Conv2d(cnn_dim, latent_size, 1)

        with th.no_grad():
            out_dim = self.vae_encoder_mu(self.cnn(tensor.float())).size()
            ib_dim = 1
            for i in out_dim[1:]:
                ib_dim *= i

        self.discrim_input = nn.Linear(ib_dim, hidden_size)
        self.discrim_hidden = nn.Linear(hidden_size, hidden_size)
        self.discrim_output = nn.Linear(hidden_size, 1)

        self.discrim_output.weight.data.mul_(0.1)
        self.discrim_output.bias.data.mul_(0.0)

        self.to(device)

    def vae_encoder(self, x):
        # h = th.tanh(self.vae_encoder_input(self.cnn(x)))
        h = th.tanh(self.cnn(x))
        return self.vae_encoder_mu(h), self.vae_encoder_logvar(h)

    def reparameterize(self, mu, logvar):
        std = th.exp(logvar / 2)
        eps = th.randn_like(std)
        return mu + std * eps

    def discriminator(self, z):
        z = th.tanh(self.discrim_input(z))
        z = th.tanh(self.discrim_hidden(z))
        prob = th.sigmoid(self.discrim_output(z))
        return prob

    def forward(self, x):
        mu, logvar = self.vae_encoder(x)
        z = self.reparameterize(mu, logvar)
        z = z.reshape(x.size(0), -1)
        z = th.relu(z)
        prob = self.discriminator(z)
        return prob, mu, logvar


class VAIL(BaseAlgorithm):
    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        vdb: VDB,
        expert_traj: th.tensor,
        policy_base: Type[BasePolicy] = None,
        normalize: bool = False,
        obs_only: bool = False,
        learning_rate: Union[float, Schedule] = 3e-4,
        lr_schedule_main: bool = False,
        lr_schedule_agent: bool = False,
        ent_decay: float = 1.0,
        norm_env: Optional[Union[GymEnv, str]] = None,
        beta: float = 0,
        alpha_beta: float = 1e-4,
        i_c: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        tune: bool = False,
        verbose: int = 0,
        support_multi_env: bool = True,
        device: Union[th.device, str] = "auto",
        create_eval_env: bool = False,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        env_name: str = None,
        gpu_optimize: bool = False,
        gail_loss: str = "gailfo",
    ):
        super(VAIL, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_base=policy_base,
            support_multi_env=support_multi_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.vdb = vdb
        self.expert_traj = expert_traj
        self.beta = beta
        self.alpha_beta = alpha_beta
        self.i_c = i_c
        self.n_actions = (
            self.action_space.n
            if isinstance(self.action_space, gym.spaces.Discrete)
            else self.action_space.shape[0]
        )
        self.device = device
        self.normalize = normalize
        self.norm_env = norm_env
        self.name = "VAIL"
        self.lr_schedule_main = lr_schedule_main
        self.lr_schedule_agent = lr_schedule_agent
        self.ent_decay = ent_decay
        if self.lr_schedule_main:
            self.learning_rate = linear_schedule(self.learning_rate)
        if self.lr_schedule_agent:
            policy_kwargs.update(
                {"learning_rate": linear_schedule(policy_kwargs["learning_rate"])}
            )
        self.policy = policy(device=self.device, seed=seed, **policy_kwargs)
        self.ts = self.policy.n_steps * self.policy.n_envs
        self.obs_only = obs_only
        self.tune = tune
        self.verbose = verbose
        self.seed = seed
        self.env_name = env_name
        self.gpu_optimize = gpu_optimize
        self.gail_loss = gail_loss

        if _init_setup_model:
            self._setup_model()

    def get_generator_batch(self, num_samples):
        """shuffle, and get a batch of num_samples of state, action tensor from
        rollout buffer
        """
        return self.policy.rollout_buffer.sample(num_samples)

    def _update_lr(
        self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]
    ) -> None:
        # Log the current learning rate
        self.policy.logger.record(
            "vail/train/learning_rate",
            self.lr_schedule(self._current_progress_remaining),
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining)
            )

    def train_vdb(self, num_samples):
        self._update_lr(self.optimizer)
        expert_state_action = self.expert_traj[
            np.random.randint(0, self.expert_traj.shape[0], num_samples), :
        ]
        expert_state_action = th.FloatTensor(expert_state_action)

        if self.normalize or self.obs_only:
            expert_state_action = normalize_observations(
                expert_state_action, self.norm_env, self.obs_only
            )
        expert_state_action = expert_state_action.to(self.device)

        num_samples = expert_state_action.shape[0]

        state_action = self.get_generator_batch(num_samples)
        state_action = get_obs_actions(
            state_action, self.env, self.device, self.obs_only
        )

        # assert state_action.size() == expert_state_action.size(), f"Sizes of agent and expert do not match. expert: {expert_state_action.size()} , agent: {state_action.size()}"

        fake_data, l_mu, l_logvar = self.vdb(state_action)
        real_data, e_mu, e_logvar = self.vdb(expert_state_action)

        l_kld = kl_divergence(l_mu, l_logvar)
        l_kld = l_kld.mean()

        e_kld = kl_divergence(e_mu, e_logvar)
        e_kld = e_kld.mean()

        kld = 0.5 * (l_kld + e_kld)
        bottleneck_loss = kld - self.i_c

        beta = max(0, self.beta + self.alpha_beta * bottleneck_loss)

        self.optimizer.zero_grad()
        discrim_loss = calc_gail_loss(self.gail_loss, self.discrim_criterion, real_data, fake_data, num_samples, self.device)
        vdb_losses = (
            discrim_loss
            + beta * bottleneck_loss
        )
        vdb_losses.backward()
        self.optimizer.step()

        expert_acc = ((self.vdb(expert_state_action)[0] < 0.5).float()).mean()
        learner_acc = ((self.vdb(state_action)[0] > 0.5).float()).mean()

        self.policy.logger.record(
            "vail/train/logits_gen",
            fake_data.detach(),
            exclude=("stdout", "json", "csv"),
        )
        self.policy.logger.record(
            "vail/train/logits_expert",
            real_data.detach(),
            exclude=("stdout", "json", "csv"),
        )

        # reduce GPU memory usage
        if self.gpu_optimize:
            del expert_state_action
            del state_action
            th.cuda.empty_cache()

        return (
            vdb_losses,
            kld,
            bottleneck_loss,
            expert_acc,
            learner_acc,
            fake_data.mean(),
            real_data.mean(),
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.optimizer = optim.Adam(self.vdb.parameters(), lr=self.lr_schedule(1))
        self.discrim_criterion = nn.BCELoss()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        callback_ppo: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "VAILAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_model_interval: int = 100,
        run_id="",
        cuda_id="",
    ) -> "VAIL":

        iteration = 0

        # default logger does not work anymore
        new_logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name)
        self.set_logger(new_logger)
        self.policy.set_logger(new_logger)

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        # expert_acc_lst, learner_acc_lst = [], []
        ep_mean_rewards = []

        while self.num_timesteps < total_timesteps:

            if iteration % 3 == 0:
                self.policy.learn(
                    total_timesteps=self.policy.n_steps,
                    reset_num_timesteps=reset_num_timesteps,
                    callback=callback_ppo,
                )
                reset_num_timesteps = False

            iteration += 1
            self.num_timesteps += 1

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if self.policy.ep_info_buffer is not None:
                mean_rew = safe_mean(
                    [ep_info["r"] for ep_info in self.policy.ep_info_buffer]
                )
                ep_mean_rewards.append([self.policy.num_timesteps, mean_rew])

                if self.tune:
                    wandb.log({"ep_rew_mean": mean_rew})

            (
                dloss,
                kld,
                bottleneck_loss,
                expert_acc,
                learner_acc,
                fake_mean,
                real_mean,
            ) = self.train_vdb(self.ts)

            # expert_acc_lst.append(expert_acc.item())
            # learner_acc_lst.append(learner_acc.item())

            #     # Display training infos
            self.policy.logger.record("time/il-iterations", iteration)
            self.policy.logger.record_mean("vail/train/vdb_loss", dloss.item())
            self.policy.logger.record_mean("vail/train/kl_div", kld.item())
            self.policy.logger.record_mean(
                "vail/train/bottleneck_loss", bottleneck_loss.item()
            )
            self.policy.logger.record("vail/train/expert_acc", expert_acc.item())
            self.policy.logger.record("vail/train/learner_acc", learner_acc.item())
            self.policy.logger.record("vail/train/fake_mean", fake_mean.item())
            self.policy.logger.record("vail/train/real mean", real_mean.item())
            self.policy.logger.record(
                "vail/train/rollout buffer", self.policy.rollout_buffer.size()
            )
            self.policy.logger.record("train/ent_coef", self.policy.ent_coef)

            callback.on_training_end()

            self.policy.ent_coef = self.policy.ent_coef * self.ent_decay

            # if not self.tune and iteration % save_model_interval == 0:
            if iteration % save_model_interval == 0:
                save_model(self, np.array(ep_mean_rewards), new_logger.get_dir(), self.env_name, self.seed,run_id,cuda_id)
                save_model_checkpoint(self.name, self.vdb.state_dict(), iteration, self.policy.num_timesteps, new_logger.get_dir(), self.env_name, self.seed,run_id,cuda_id)

        return self, np.array(ep_mean_rewards), new_logger.get_dir()


class VQEmbeddingEMA(nn.Module):
    def __init__(
        self,
        n_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.999,
        epsilon=1e-5,
        device="cpu",
        threshold_ema_dead_code=5,
        mask_prob=0.5,
        regularization="ortho_loss",
        num_codes_ortho=128,
        ortho_weight=10.0,
    ):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.embedding_dim = embedding_dim
        self.mask_prob = mask_prob
        self.regularization = regularization
        self.num_codes_ortho = num_codes_ortho
        self.ortho_weight = ortho_weight
        self.device = device

        init_bound = 1 / n_embeddings
        embedding = th.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", th.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.to(device=device)

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = th.addmm(
            th.sum(self.embedding ** 2, dim=1)
            + th.sum(x_flat ** 2, dim=1, keepdim=True),
            x_flat,
            self.embedding.t(),
            alpha=-2.0,
            beta=1.0,
        )

        indices = th.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)

        return quantized

    def forward(self, x):
        M, D = self.embedding.size()
        # Flatten input
        x_flat = x.detach().reshape(-1, D)

        distances = th.addmm(
            th.sum(self.embedding ** 2, dim=1)
            + th.sum(x_flat ** 2, dim=1, keepdim=True),
            x_flat,
            self.embedding.t(),
            alpha=-2.0,
            beta=1.0,
        )

        indices = th.argmin(distances.float(), dim=-1)
        # print(indices.shape)
        encodings = F.one_hot(indices, M).float()

        quantized = F.embedding(indices, self.embedding)
        # print("quantized shape")
        # print(quantized.shape)
        # torch.Size([384, 128])

        # Masking of quantized output
        # https://stackoverflow.com/questions/61956893/how-to-mask-a-3d-tensor-with-2d-mask-and-keep-the-dimensions-of-original-vector
        # https://github.com/alinlab/oreo/blob/master/atari_vqvae_oreo.py
        if self.regularization == "mask_codes":
            prob = th.ones(quantized.shape[0]) * (1 - self.mask_prob)
            code_mask = th.bernoulli(prob).to(self.device)
            code_mask = code_mask.unsqueeze(-1).expand(quantized.size())
            quantized = quantized * code_mask

        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * th.sum(
                encodings, dim=0
            )
            n = th.sum(self.ema_count)
            self.ema_count = (
                (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            )

            dw = th.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        codebook_loss = F.mse_loss(x.detach(), quantized)
        e_latent_loss = F.mse_loss(x, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        # expire dead codes https://github.com/lucidrains/vector-quantize-pytorch/
        if self.regularization == "expire_codes":
            expired_codes = self.ema_count < self.threshold_ema_dead_code
            if th.any(expired_codes):
                num_samples = x_flat.shape[0]
                sample_idx = th.randperm(num_samples, device=self.device)[:M]
                x_vals = x_flat[sample_idx]
                modified_codebook = th.where(
                    expired_codes[..., None],
                    x_vals,
                    self.embedding
                )
                self.embedding.data.copy_(modified_codebook)

        # calc orthogonal loss similar to https://github.com/lucidrains/vector-quantize-pytorch/
        if self.regularization == "ortho_loss":
            unique_code_ids = th.unique(indices)
            codebk = self.embedding[unique_code_ids]

            num_codes = codebk.shape[0]
            rand_ids = th.randperm(num_codes, device=self.device)[:self.num_codes_ortho]
            codebk = codebk[rand_ids]

            n = codebk.shape[0]
            normed_codes = F.normalize(codebk, p = 2, dim = -1)
            identity = th.eye(n, device=self.device)
            cosine_sim = th.einsum('i d, j d -> i j', normed_codes, normed_codes)
            ortho_loss = ((cosine_sim - identity) ** 2).sum() / (n ** 2)
            ortho_loss = ortho_loss * self.ortho_weight
        else:
            ortho_loss = 0.0

        # avg_probs = th.mean(encodings, dim=0)
        # perplexity = th.exp(-th.sum(avg_probs * th.log(avg_probs + 1e-10)))

        return quantized, commitment_loss, codebook_loss, ortho_loss, indices


class VQVAE(nn.Module):
    def __init__(self, num_inputs, hidden_size, latent_size, Codebook, device):
        super(VQVAE, self).__init__()

        self.num_inputs = num_inputs

        self.encoder_input = nn.Linear(num_inputs, hidden_size)
        self.encoder_output = nn.Linear(hidden_size, latent_size)

        self.codebook = Codebook

        self.discrim_input = nn.Linear(latent_size, hidden_size)
        self.discrim_hidden = nn.Linear(hidden_size, hidden_size)
        self.discrim_output = nn.Linear(hidden_size, 1)
        self.discrim_output.weight.data.mul_(0.1)
        self.discrim_output.bias.data.mul_(0.0)

        self.to(device)

    def encoder(self, x):
        z = th.tanh(self.encoder_input(x))
        z = th.tanh(self.encoder_output(z))
        return z

    def discriminator(self, z):
        z = th.tanh(self.discrim_input(z))
        z = th.tanh(self.discrim_hidden(z))
        prob = th.sigmoid(self.discrim_output(z))
        return prob

    def forward(self, x):
        z = self.encoder(x)
        # convert inputs from BCHW -> BHWC
        (
            z_quantized,
            commitment_loss,
            codebook_loss,
            ortho_loss,
            indices,
        ) = self.codebook(z)
        prob = self.discriminator(z_quantized)
        return prob, commitment_loss, codebook_loss, ortho_loss, indices, z_quantized

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class VQVAEImage(nn.Module):
    def __init__(
        self,
        cnn_version,
        observation_space: gym.spaces.Box,
        hidden_size,
        embedding_dim,
        Codebook,
        device,
    ):
        super(VQVAEImage, self).__init__()

        # assume env observation space is channel last
        self.observation_space = observation_space
        self.n_input_channels = observation_space.shape[0]
        self.embedding_dim = embedding_dim
        print("Num of input channels", self.n_input_channels)
        # self.cnn = nn.Sequential(
        #     # nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     # nn.Flatten(),
        # )
        # self.encoder_output = nn.Conv2d(in_channels=64, 
        #                                 out_channels=embedding_dim,
        #                                 kernel_size=1, 
        #                                 stride=1)
        # self.encoder_input = nn.Sequential(nn.Linear(n_flatten, hidden_size), nn.ReLU())
        # self.encoder_output = nn.Linear(hidden_size, latent_size)
        self.codebook = Codebook

        h,w=self.build_cnn(cnn_version)
        
        print("built cnn ")
        h, w = conv_output_shape((h, w), 1, 1, 0, 1) # encoder = codebook
        self.discrim_input = nn.Linear(embedding_dim*h*w, hidden_size)
        self.discrim_hidden = nn.Linear(hidden_size, hidden_size)
        self.discrim_output = nn.Linear(hidden_size, 1)
        self.discrim_output.weight.data.mul_(0.1)
        self.discrim_output.bias.data.mul_(0.0)

        self.to(device)

    
    def build_cnn(self,cnn_version='miniworld'):
        print("building cnn")
        num_hiddens=128
        num_residual_layers=2
        num_residual_hiddens=32
        print("cnn_version")
        print(cnn_version)

        if cnn_version=='base':
            d_cnn = OrderedDict([
            ('conv1', nn.Conv2d(self.n_input_channels,32,5,2,0)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32,64,5,2,0)),
            ('relu2', nn.ReLU()),
            ])

        # minigrid-doorkey traininig was 3 input channels insetad of n_input_channels
        elif cnn_version =='minigrid_1':
            d_cnn = OrderedDict([
            ('conv1', nn.Conv2d(self.n_input_channels,32,3,2,1)),
            ('elu1', nn.ELU()),
            ('conv2', nn.Conv2d(32,32,3,2,1)),
            ('elu2', nn.ELU()),
            ('conv3', nn.Conv2d(32,32,3,2,1)), # 50 -> 7x7
            ('elu3', nn.ELU()),
            ])

    
        elif cnn_version =='minigrid_2':
            d_cnn = OrderedDict([
                ('conv1', nn.Conv2d(self.n_input_channels,16,2)),
                ('relu1', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ('conv2', nn.Conv2d(16,32, kernel_size=(2, 2))),
                ('relu2', nn.ReLU()),
                ('maxpool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ('conv3', nn.Conv2d(32,32, kernel_size=(2, 2))),
                ('relu3', nn.ReLU()), # 50 -> 10
                ])

        elif cnn_version=="minigrid_3":
            d_cnn =  OrderedDict([
                ('conv1', nn.Conv2d(self.n_input_channels,16,2)),
                ('relu1', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ('conv2', nn.Conv2d(16,32, kernel_size=(2, 2))),
                ('relu2', nn.ReLU()),
                ('maxpool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ('conv3', nn.Conv2d(32,32, kernel_size=(2, 2))),
                ('relu3', nn.ReLU()), # 50 -> 22
                ])


        elif cnn_version =='miniworld_1': # (8, 10)
            d_cnn = OrderedDict([
            # ('conv1', nn.Conv2d(self.n_input_channels,32,3,2,0)),
            # ('relu1', nn.ReLU()),
            # ('conv2', nn.Conv2d(32,32,3,2,0)),
            # ('relu2', nn.ReLU()),
            # ('conv3', nn.Conv2d(32,32,3,2,0)),
            # ('relu3', nn.ReLU()),
            # ])
            ('conv1', nn.Conv2d(self.n_input_channels,32,3,2,1)),
            ('elu1', nn.ELU()),
            ('conv2', nn.Conv2d(32,32,3,2,1)),
            ('elu2', nn.ELU()),
            ('conv3', nn.Conv2d(32,32,3,2,1)), # 50 -> 7x7
            ('elu3', nn.ELU()),
            ])

        elif cnn_version =='miniworld_2': # (5, 8)
            d_cnn = OrderedDict([
            # ('conv1', nn.Conv2d(self.n_input_channels,32,5,2,0)),
            # ('relu1', nn.ReLU()),
            # ('batchnorm1',  nn.BatchNorm2d(32)),
            # ('conv2', nn.Conv2d(32,32,5,2,0)),
            # ('relu2', nn.ReLU()),
            # ('batchnorm2',  nn.BatchNorm2d(32)),
            # ('conv3', nn.Conv2d(32,32,4,2,0)),
            # ('relu3', nn.ReLU()),
            # ('batchnorm3',  nn.BatchNorm2d(32)),
            # ])
            ('conv1', nn.Conv2d(self.n_input_channels,16,2)), 
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('conv2', nn.Conv2d(16,32, kernel_size=(2, 2))),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('conv3', nn.Conv2d(32,32, kernel_size=(2, 2))),
            ('relu3', nn.ReLU()), # 50 -> 10
            # ('maxpool3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            # ('conv4', nn.Conv2d(32,32, kernel_size=(2, 2))),
            # ('relu4', nn.ReLU()), # 50 -> 10

            ])
            

        elif cnn_version =='miniworld_3': # 10x15
            d_cnn= OrderedDict([
            ('conv1', nn.Conv2d(self.n_input_channels,32,3,2,0)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=(3, 3), stride=2)),
            ('conv2', nn.Conv2d(32,32, kernel_size=(3, 3))),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=(3, 3), stride=2)),
            ('conv3', nn.Conv2d(32,32, kernel_size=(3, 3))),
            ('relu3', nn.ReLU()),
            ])

        elif cnn_version == 'atari':
            d_cnn = OrderedDict([
            ('conv1', nn.Conv2d(self.n_input_channels,num_hiddens // 4,4,2,1)),  ## 42 x 42
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(num_hiddens // 4, num_hiddens // 2, 4,2,1)), ## 21 x 21
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(num_hiddens // 2, num_hiddens, 4,2,1)), ##  10x10
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(num_hiddens, num_hiddens, 3,1,0)), ##  8x8
            ('residual', ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)) ## 8x8

            ])

        self.cnn = nn.Sequential(d_cnn)
        convs = [v for k,v in d_cnn.items() if 'conv' in k or 'pool' in k]
        h, w = self.observation_space.shape[1], self.observation_space.shape[2]
        self.encoder_output = nn.Conv2d(in_channels=convs[-1].out_channels, 
                                        out_channels=self.embedding_dim,
                                        kernel_size=1, 
                                        stride=1)
        for i,conv in enumerate(convs):
            if not isinstance(conv.stride,int):
                stride=conv.stride[0]
                padding = conv.padding[0]
            else: 
                stride = conv.stride
                padding=conv.padding

            h,w= conv_output_shape((h, w), conv.kernel_size[0],  stride,  padding, 1) 
    

        for i in range(100):
            print(h,w)
        return h,w

    def encoder(self, x):
        # z = th.tanh(self.encoder_input(self.cnn(x)))
        z = th.relu(self.cnn(x))
        z = th.relu(self.encoder_output(z))
        return z

    def discriminator(self, z):
        z = th.tanh(self.discrim_input(z))
        z = th.tanh(self.discrim_hidden(z))
        prob = th.sigmoid(self.discrim_output(z))
        return prob

    def forward(self, x):
        z = self.encoder(x)
        (
            z_quantized,
            commitment_loss,
            codebook_loss,
            ortho_loss,
            indices,
        ) = self.codebook(z)
        z_quantized = z.reshape(x.size(0), -1)
        z_quantized = th.relu(z_quantized)
        prob = self.discriminator(z_quantized)
        return prob, commitment_loss, codebook_loss, ortho_loss, indices, z_quantized


class VQVAIL(BaseAlgorithm):
    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        vqvae: VQVAE,
        expert_traj: th.tensor,
        policy_base: Type[BasePolicy] = None,
        normalize: bool = False,
        learning_rate: Union[float, Schedule] = 3e-4,
        obs_only: bool = False,
        lr_schedule_main: bool = False,
        lr_schedule_agent: bool = False,
        ent_decay: float = 1.0,
        norm_env: Optional[Union[GymEnv, str]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        tune: bool = False,
        verbose: int = 0,
        support_multi_env: bool = True,
        device: Union[th.device, str] = "auto",
        create_eval_env: bool = False,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        env_name: str = None,
        gpu_optimize: bool = False,
        gail_loss: str = "gailfo",
    ):
        super(VQVAIL, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_base=policy_base,
            support_multi_env=support_multi_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.vqvae = vqvae
        self.expert_traj = expert_traj
        self.n_actions = (
            self.action_space.n
            if isinstance(self.action_space, gym.spaces.Discrete)
            else self.action_space.shape[0]
        )
        self.device = device
        self.normalize = normalize
        self.norm_env = norm_env
        self.name = "VQVAIL"
        self.lr_schedule_main = lr_schedule_main
        self.lr_schedule_agent = lr_schedule_agent
        self.ent_decay = ent_decay
        if self.lr_schedule_main:
            self.learning_rate = linear_schedule(self.learning_rate)
        if self.lr_schedule_agent:
            policy_kwargs.update(
                {"learning_rate": linear_schedule(policy_kwargs["learning_rate"])}
            )
        self.policy = policy(device=self.device, seed=seed, **policy_kwargs)
        self.ts = self.policy.n_steps * self.policy.n_envs
        self.obs_only = obs_only
        self.tune = tune
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.seed = seed
        self.env_name = env_name
        self.gpu_optimize = gpu_optimize
        self.gail_loss = gail_loss

        if _init_setup_model:
            self._setup_model()

    def get_generator_batch(self, num_samples):
        """shuffle, and get a batch of num_samples of state, action tensor from
        rollout buffer
        """
        return self.policy.rollout_buffer.sample(num_samples)

    def _update_lr(
        self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]
    ) -> None:
        # Log the current learning rate
        self.policy.logger.record(
            "vqvail/train/learning_rate",
            self.lr_schedule(self._current_progress_remaining),
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining)
            )


    def get_latents(self, num_samples, expert_only= False, factor=3):
        self.vqvae.eval()

        factor=1
        num_samples = num_samples * factor

        expert_state_action = self.expert_traj[
            np.random.randint(0, self.expert_traj.shape[0], num_samples), :
        ]
        expert_state_action = th.FloatTensor(expert_state_action).to(self.device)

        if self.normalize or self.obs_only:
            expert_state_action = normalize_observations(
                expert_state_action, self.norm_env, self.obs_only
            )

        num_samples = expert_state_action.shape[0]

        real_z = self.vqvae.encoder(expert_state_action)
        real_z_quantized, _, _, _, _ = self.vqvae.codebook(real_z)

        if expert_only:

            return real_z,real_z_quantized

        else:
            state_action = self.get_generator_batch(num_samples)
            state_action = get_obs_actions(
                state_action, self.env, self.device, self.obs_only
            )

            fake_z = self.vqvae.encoder(state_action)
            fake_z_quantized, _, _, _, _ = self.vqvae.codebook(fake_z)

            return fake_z, real_z, fake_z_quantized, real_z_quantized


    def train_vqvae(self, num_samples):
        # self._update_lr(self.optimizer)
        expert_state_action = self.expert_traj[
            np.random.randint(0, self.expert_traj.shape[0], num_samples), :
        ]
        expert_state_action = th.FloatTensor(expert_state_action)

        if self.normalize:
            expert_state_action = normalize_observations(
                expert_state_action, self.norm_env, self.obs_only
            )
        expert_state_action = expert_state_action.to(self.device)

        num_samples = expert_state_action.shape[0]


        # assert state_action.size() == expert_state_action.size(), f"Sizes of agent and expert do not match. expert: {expert_state_action.size()} , agent: {state_action.size()}"
        (
            real_data,
            commitment_loss_real,
            codebook_loss_real,
            ortho_loss_real,
            real_idx,
            rzq,
        ) = self.vqvae(expert_state_action)
    
        state_action = self.get_generator_batch(num_samples)
        state_action = get_obs_actions(
            state_action, self.env, self.device, self.obs_only
        )
        (
            fake_data,
            commitment_loss_fake,
            codebook_loss_fake,
            ortho_loss_fake,
            fake_idx,
            fzq,
        ) = self.vqvae(state_action)


        commitment_loss = 0.5 * (commitment_loss_fake + commitment_loss_real)
        codebook_loss = 0.5 * (codebook_loss_fake + codebook_loss_real)
        ortho_loss = 0.5 * (ortho_loss_fake + ortho_loss_real)

        self.optimizer.zero_grad()

        discrim_loss = calc_gail_loss(self.gail_loss, self.discrim_criterion, real_data, fake_data, num_samples, self.device)
        vqvae_loss = (
            discrim_loss
            + codebook_loss
            + commitment_loss
            + ortho_loss
        )
        vqvae_loss.backward()
        self.optimizer.step()

        expert_acc = ((self.vqvae(expert_state_action)[0] < 0.5).float()).mean()
        learner_acc = ((self.vqvae(state_action)[0] > 0.5).float()).mean()

        self.policy.logger.record(
            "vqvail/train/logits_gen",
            fake_data.detach(),
            exclude=("stdout", "json", "csv"),
        )
        self.policy.logger.record(
            "vqvail/train/logits_expert",
            real_data.detach(),
            exclude=("stdout", "json", "csv"),
        )
        self.policy.logger.record(
            "vqvail/train/codes_gen",
            fake_idx.detach(),
            exclude=("stdout", "json", "csv"),
        )
        self.policy.logger.record(
            "vqvail/train/codes_expert",
            real_idx.detach(),
            exclude=("stdout", "json", "csv"),
        )

        # reduce GPU memory usage
        if self.gpu_optimize:
            del expert_state_action
            del state_action
            th.cuda.empty_cache()

        return (
            vqvae_loss,
            commitment_loss,
            codebook_loss,
            expert_acc,
            learner_acc,
            fake_data.mean(),
            real_data.mean(),
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()

        self.optimizer = optim.Adam(self.vqvae.parameters(), lr=self.lr_schedule(1))
        self.discrim_criterion = nn.BCELoss()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        callback_ppo: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "VQVAILAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_model_interval: int = 100,
        run_id="",
        cuda_id="",
    ) -> "VQVAIL":

        iteration = 0

        # default logger does not work anymore
        new_logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name)
        self.set_logger(new_logger)
        self.policy.set_logger(new_logger)

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        # expert_acc_lst, learner_acc_lst = [], []
        ep_mean_rewards = []

        while self.num_timesteps < total_timesteps:

            if iteration % 3 == 0:
                self.policy.learn(
                    total_timesteps=self.policy.n_steps,
                    reset_num_timesteps=reset_num_timesteps,
                    callback=callback_ppo,
                )
                reset_num_timesteps = False

            iteration += 1
            self.num_timesteps += 1

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if self.policy.ep_info_buffer is not None:
                mean_rew = safe_mean(
                    [ep_info["r"] for ep_info in self.policy.ep_info_buffer]
                )
                ep_mean_rewards.append([self.policy.num_timesteps, mean_rew])

                if self.tune:
                    wandb.log({"ep_rew_mean": mean_rew})

            (
                dloss,
                commitment_loss,
                codebook_loss,
                expert_acc,
                learner_acc,
                fake_mean,
                real_mean,
            ) = self.train_vqvae(self.ts)
 
            # expert_acc_lst.append(expert_acc.item())
            # learner_acc_lst.append(learner_acc.item())

            # Display training infos
            self.policy.logger.record_mean("train/ent_coef", self.policy.ent_coef)
            self.policy.logger.record_mean("vqvail/train/vqvae_loss", dloss.item())
            self.policy.logger.record_mean(
                "vqvail/train/commitment_loss", commitment_loss.item()
            )
            self.policy.logger.record_mean(
                "vqvail/train/codebook_loss", codebook_loss.item()
            )
            self.policy.logger.record("vqvail/train/expert_acc", expert_acc.item())
            self.policy.logger.record("vqvail/train/learner_acc", learner_acc.item())
            self.policy.logger.record("vqvail/train/fake_mean", fake_mean.item())
            self.policy.logger.record("vqvail/train/real mean", real_mean.item())
            self.policy.logger.record("time/il-iterations", iteration)
            self.policy.logger.record(
                "vqvail/train/rollout buffer", self.policy.rollout_buffer.size()
            )

            callback.on_training_end()

            self.policy.ent_coef = self.policy.ent_coef * self.ent_decay

            # if not self.tune and iteration % save_model_interval == 0:
            if iteration % save_model_interval == 0:
                save_model(self, np.array(ep_mean_rewards), new_logger.get_dir(), self.env_name, self.seed,run_id,cuda_id)
                save_model_checkpoint(self.name, self.vqvae.state_dict(), iteration, self.policy.num_timesteps, new_logger.get_dir(), self.env_name, self.seed,run_id,cuda_id)

        return self, np.array(ep_mean_rewards), new_logger.get_dir()
