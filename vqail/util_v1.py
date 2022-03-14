from platform import version
import random
import yaml
import os
import time
from pathlib import Path
from collections import OrderedDict
from pprint import pprint
from typing import Any, Dict, Optional, Type, Union, Callable, Tuple
import gym
from gym import spaces
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.preprocessing import (
    preprocess_obs,
    is_image_space,
    get_obs_shape,
    get_action_dim,
    get_flattened_obs_dim,
)
from stable_baselines3.common.utils import set_random_seed


def modify_expert_data_for_train(expert_traj, env, obs_only=False):
    """Preprocess observations and one-hot encode actions"""
    print(f"Preprocess observations and one-hot encode actions on expert data")

    # Images use LfO only
    if is_image_space(env.observation_space):
        expert_data = preprocess_obs(expert_traj, env.observation_space)
        return expert_data

    # Non-image observations use LfD
    num_obs, num_actions = get_obs_action_dim(env)

    # preprocessing states
    states = expert_traj[:, :num_obs]
    states = preprocess_obs(states, env.observation_space)

    # one hot encoding of actions
    if isinstance(env.action_space, gym.spaces.Discrete):
        actions = expert_traj[:, -1]
    else:
        actions = expert_traj[:, -num_actions:]

    # concatenate states and actions
    actions = preprocess_obs(actions, env.action_space)
    expert_data = th.FloatTensor(th.cat([states, actions], 1))

    return expert_data


def eval_policy(model_test, eval_env, verbose=1, n_eval_episodes=100):
    mean_reward, std_reward = evaluate_policy(
        model_test, eval_env, n_eval_episodes=n_eval_episodes
    )

    if verbose > 0:
        print(f"Mean reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    return mean_reward, std_reward


def kl_divergence(mu, logvar):
    kl_div = 0.5 * th.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)
    return kl_div


def get_obs_action_dim(env):
    """Get dimensions of observations and actions"""
    if is_image_space(env.observation_space):
        num_obs = get_obs_shape(env.observation_space)
    else:
        num_obs = get_flattened_obs_dim(env.observation_space)
    if isinstance(env.action_space, spaces.Discrete):
        num_actions = int(env.action_space.n)
    else:
        num_actions = get_action_dim(env.action_space)

    return num_obs, num_actions


def normalize_observations(expert_data, env, obs_only=False):
    """Preprocess expert data"""
    if obs_only:
        # modify_expert_data_for_train already uses preprocess_obs
        # preprocess_obs normalizes images by dividing them by 255 (to have values in [0, 1])
        return expert_data

    num_obs, _ = get_obs_action_dim(env)

    state_action_arr = expert_data.cpu().numpy()
    obs = state_action_arr[:, :num_obs]
    obs = env.normalize_obs(obs)

    actions = state_action_arr[:, num_obs:]
    expert_data = th.FloatTensor(np.concatenate([obs, actions], 1))
    return expert_data


def get_obs_actions(state_action, env, device, obs_only=False):
    """Preprocess agent collected data"""
    states = preprocess_obs(state_action.observations, env.observation_space)
    if obs_only:
        state_action = states.to(device=device)
    else:
        actions = preprocess_obs(state_action.actions, env.action_space)
        state_action = th.cat(
            [
                states.reshape(states.shape[0], -1),
                actions.reshape(actions.shape[0], -1),
            ],
            1,
        ).to(device=device)
    return state_action


def read_hyperparameters(
    algo, env_id, _is_atari=False, custom_hyperparams=None, verbose=0
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Code taken from Stable Baselines 3 Zoo
    # Load hyperparameters from yaml file
    with open(f"hyperparams/hyperparams_{algo}.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif _is_atari:
            hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(f"Hyperparameters not found for {algo}-{env_id}")

    if custom_hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(custom_hyperparams)
    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict(
        [(key, hyperparams[key]) for key in sorted(hyperparams.keys())]
    )

    if verbose > 0:
        pprint(saved_hyperparams)

    return hyperparams, saved_hyperparams


def save_eval_rewards(model, eval_env, env_name, seed, n_eval_episodes=100):
    rew_lst, rew_len = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )

    Path("./outputs").mkdir(exist_ok=True)
    log_path = os.path.join("./outputs", env_name)
    log_path = Path(log_path)
    log_path.mkdir(exist_ok=True)
    print(f"Saving eval rewards at {log_path}")

    path = os.path.join(
        log_path, "eval_reward_{}-{}-{}.npy".format(model.name, seed, env_name)
    )
    with open(path, "wb") as f:
        np.save(f, np.array(rew_lst))



def save_model(model, ep_mean_rewards,  iteration, num_timesteps, env_name, seed,run_id,cuda_id):
    Path("./outputs").mkdir(exist_ok=True)
    log_path = os.path.join("./outputs", env_name,run_id)
    log_path = Path(log_path)
    os.makedirs(log_path,exist_ok=True)
    # log_path.mkdir(exist_ok=True)
    print(f"Saving model at {log_path}")
    path = os.path.join(log_path, "{}-{}-{}-{}-{}-{}-{}.npy".format(env_name,model.name, seed,  cuda_id, run_id,iteration, num_timesteps))
    model.policy.save(path)

    # save mean rewards
    path = os.path.join(
        log_path, "mean_reward_{}-{}-{}.npy".format(model.name, seed, env_name)
    )
    with open(path, "wb") as f:
        np.save(f, np.array(ep_mean_rewards))


def evaluate_model(model, eval_env, env_name, norm_obs, hyperparams, log_dir, seed, render=True):
    log_path = os.path.join(log_dir, "outputs")
    Path(log_path).mkdir(exist_ok=True)
    print(f"Evaluating model at {log_path}")

    if norm_obs:
        env = model.get_env()
        stats_path = os.path.join(log_path, "vec_normalize.pkl")
        env.save(stats_path)
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    mean_reward, std_reward = eval_policy(model, eval_env)
    print(f"Mean reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    path = os.path.join(log_path, "{}-{}-{}.txt".format(model.name, seed, env_name))
    with open(path, "a") as f:
        f.write(
            f"Time: {time.time()}, Mean reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f} \n"
        )
        f.close()

    path = os.path.join(log_path, "Hyperparams-{}-{}-{}.txt".format(model.name, seed, env_name))
    with open(path, "w") as f:
        f.write(
            str(hyperparams)
        )
        f.close()


def get_device(device):
    if device == "auto":
        use_cuda = th.cuda.is_available()
        device = th.device("cuda" if use_cuda else "cpu")
    elif device == "cuda":
        device = th.device("cuda")
    else:
        device = th.device("cpu")
    return device

def save_model_checkpoint(model_name, model_state_dict, il_iteration, ppo_timesteps, env_name, seed,run_id,cuda_id):
    Path("./checkpoints").mkdir(exist_ok=True)
    log_path = os.path.join("./checkpoints", env_name,run_id)
    log_path = Path(log_path)
    # log_path.mkdir(exist_ok=True)
    os.makedirs(log_path,exist_ok=True)
    print(f"Saving model checkpoints at {log_path}")

    path = os.path.join(log_path, "{}-{}-{}-{}-{}-{}-{}".format(model_name, seed, cuda_id, run_id, env_name, il_iteration, ppo_timesteps))

    # save model checkpoint
    th.save({
                'il_iteration': il_iteration,
                'ppo_timesteps': ppo_timesteps,
                'model_state_dict': model_state_dict,
                }, path)