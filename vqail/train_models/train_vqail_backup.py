import os
from textwrap import wrap
from typing import Dict

from gym import spaces
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
    DummyVecEnv,
    VecTransposeImage,
)
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack

from models import VQVAIL, VQEmbeddingEMA, VQVAE, VQVAEImage
from wrappers import VecCustomReward
from util import (
    modify_expert_data_for_train,
    save_model,
    get_obs_action_dim,
    save_eval_rewards,
    evaluate_model,
)
from env_utils import get_mujoco_vec_env, get_mujoco_img_vec_env


def train(args, hyperparams, saved_hyperparams, expert_traj, device, seed, tune=False,tb_log_name="VQAILAlgorithm"):
    # 1. Set up variables for parameters
    obs_only = True if "obs_only" in hyperparams and hyperparams["obs_only"] else False
    norm_obs = True if "norm_obs" in hyperparams and hyperparams["norm_obs"] else False
    reward_factor = (
        hyperparams["reward_factor"] if "reward_factor" in hyperparams else -1.0
    )
    lr = hyperparams["learning_rate"] if "learning_rate" in hyperparams else 3e-4
    lr_schedule_main = (
        hyperparams["lr_schedule_main"] if "lr_schedule_main" in hyperparams else False
    )
    lr_schedule_agent = (
        hyperparams["lr_schedule_agent"]
        if "lr_schedule_agent" in hyperparams
        else False
    )
    ent_decay = hyperparams["ent_decay"] if "ent_decay" in hyperparams else 1.0
    timesteps = (
        args.timesteps if args.timesteps is not None else hyperparams["total_timesteps"]
    )
    env_kwargs = {"chg_box_color": True} if args.chg_box_color else None

    # 2. Define environment
    # Cannot use multi-processing for WandB
    if tune:
        vec_class = DummyVecEnv
        n_envs = 1
    else:
        vec_class = SubprocVecEnv
        n_envs = hyperparams["n_envs"]

    # Add wrappers to environment
    if "time_limit" in hyperparams:
        wrapper = TimeFeatureWrapper
    elif "atari" in hyperparams:
        wrapper = AtariWrapper
    else:
        wrapper = None

    if "mujoco_img" in hyperparams:
        envs, eval_env = get_mujoco_img_vec_env(args.env_id, n_envs=n_envs, seed=seed)
    elif "mujoco_dict" in hyperparams:
        envs, eval_env = get_mujoco_vec_env(
            args.env_id, n_envs=n_envs, seed=seed, wrapper_class=wrapper,
        )
    else:
        envs = make_vec_env(
            args.env_id,
            n_envs=n_envs,
            seed=seed,
            wrapper_class=wrapper,
            vec_env_cls=vec_class,
            env_kwargs=env_kwargs,
        )
        eval_env = make_vec_env(
            args.env_id, 1, wrapper_class=wrapper,
            env_kwargs=env_kwargs,
        )

    # We want environment with Channels last, some weird pytorch errors related to # of n_envs and n_stack. Seems to work better with channel last env.
    if is_image_space(envs.observation_space):
        print("Image observations...")
        assert not is_image_space_channels_first(
            envs.observation_space
        ), "Environment should be channel last"

        if "frame_stack" in hyperparams:
            envs = VecFrameStack(envs, n_stack=hyperparams["frame_stack"])
            eval_env = VecFrameStack(eval_env, n_stack=hyperparams["frame_stack"])

    # eval_env.seed(seed)
    print(
        "envs.observation_space={}, envs.action_space={}".format(
            envs.observation_space, envs.action_space
        )
    )

    # 2. Instantiate model
    print(f"Device in train_vqvail.py = {device}")

    vqvail_params = eval(hyperparams["vqvail"])
    n_embeddings = vqvail_params["n_embeddings"]
    hidden_size = vqvail_params["hidden_size"]
    embedding_dim = vqvail_params["embedding_dim"]
    embedding_dim = n_envs * embedding_dim

    codebook = VQEmbeddingEMA(
        n_embeddings=n_embeddings, embedding_dim=embedding_dim, device=device
    )
    if is_image_space(envs.observation_space):
        # Uses LfO (Learning from observations).
        model = VQVAEImage(
            envs.observation_space, hidden_size, embedding_dim, codebook, device
        )
    else:
        # Flattened observations and action dimensions
        # Uses LfD (learning from demonstrations).
        num_inputs, num_outputs = get_obs_action_dim(envs)
        print("num_inputs={}, num_outputs={}".format(num_inputs, num_outputs))
        model = VQVAE(
            num_inputs + num_outputs, hidden_size, embedding_dim, codebook, device
        )

    # 3. More wrappers for environment
    # separate norm_env as it is done in https://github.com/HumanCompatibleAI/imitation
    if norm_obs:
        envs = norm_env = VecNormalize(envs, norm_obs=True, norm_reward=False)
    else:
        norm_env = envs
    envs = VecCustomReward(
        model,
        envs,
        train=True,
        obs_only=obs_only,
        reward_factor=reward_factor,
        device=device,
    )

    print("Environment observation shape: ", envs.observation_space)

    # 3. Preprocess expert data
    # Expert needs to be channel first
    expert_traj = modify_expert_data_for_train(expert_traj, envs, obs_only)
    print(f"After modifying, expert_traj: {expert_traj.shape}")
    if (
        is_image_space(envs.observation_space)
        and not is_image_space_channels_first(envs.observation_space)
        and not np.argmin(expert_traj.size()) == 1
    ):
        # Data is NxHxWxC. Need to be transposed to NxCxHxW.
        print("Permuting expert data...")
        expert_traj = expert_traj.permute(0, 3, 1, 2)
        print(f"Expert after permutation : {expert_traj.size()}")

    policy_kwargs = eval(hyperparams["policy_kwargs"])
    policy_kwargs.update({"env": envs})
    print("policy_kwargs: ", policy_kwargs)

    log_path = os.path.join(args.log, args.env_id)
    algo = VQVAIL(
        PPO,
        envs,
        model,
        expert_traj=expert_traj,
        tensorboard_log=log_path,
        device=device,
        verbose=args.verbose,
        seed=seed,
        learning_rate=lr,
        normalize=norm_obs,
        norm_env=norm_env,
        lr_schedule_main=lr_schedule_main,
        lr_schedule_agent=lr_schedule_agent,
        obs_only=obs_only,
        tune=tune,
        ent_decay=ent_decay,
        env_name=args.env_id,
        policy_kwargs=policy_kwargs,
    )

    algo, ep_mean_rewards = algo.learn(total_timesteps=timesteps, eval_env=eval_env,tb_log_name=tb_log_name)

    # save model and evaluation results
    if not tune:
        evaluate_model(algo, eval_env, args.env_id, norm_obs, seed)

    return algo
