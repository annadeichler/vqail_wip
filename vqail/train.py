import argparse
import os

import torch as th
import numpy as np
import gym
import gym_miniworld
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)

from train_models import train_gail, train_vail, train_vqvail
from arguments import get_args
from util import read_hyperparameters, get_device, modify_expert_data_for_train
from env_utils import get_mujoco_vec_env, get_mujoco_img_vec_env, mini_grid_wrapper


ALGOS = {
    "vqail": train_vqvail,
    "vail": train_vail,
    "gail": train_gail,
}
choices = list(ALGOS.keys()).append("all")
seeds = list(range(1000, 20000, 1000))


def set_up_parameters(hyperparams, args):
    # 1. Set up variables for parameters
    args.obs_only = True if "obs_only" in hyperparams and hyperparams["obs_only"] else False
    args.norm_obs = True if "norm_obs" in hyperparams and hyperparams["norm_obs"] else False
    args.reward_factor = (
        hyperparams["reward_factor"] if "reward_factor" in hyperparams else -1.0
    )
    args.lr = hyperparams["learning_rate"] if "learning_rate" in hyperparams else 3e-4
    args.lr_schedule_main = (
        hyperparams["lr_schedule_main"] if "lr_schedule_main" in hyperparams else False
    )
    args.lr_schedule_agent = (
        hyperparams["lr_schedule_agent"]
        if "lr_schedule_agent" in hyperparams
        else False
    )
    args.ent_decay = hyperparams["ent_decay"] if "ent_decay" in hyperparams else 1.0
    args.timesteps = (
        args.timesteps if args.timesteps is not None else hyperparams["total_timesteps"]
    )
    args.env_kwargs = {"chg_box_color": True} if args.chg_box_color else None
    if args.top_view:
        if args.env_kwargs:
            args.env_kwargs["top_view"] = True
        else:
            args.env_kwargs = {"top_view": True}

    if args.chg_tex_train_test:
        if args.env_kwargs:
            args.env_kwargs["chg_tex"] = True
        else:
            args.env_kwargs = {"chg_tex": True}


    if args.num_objs:
        if args.env_kwargs:
            args.env_kwargs["num_objs"] = args.num_objs
        else:
            args.env_kwargs = {"num_objs": args.num_objs}

    args.test_env_kwargs = args.env_kwargs.copy() if args.env_kwargs else None

    if args.chg_tex_test:
        if args.test_env_kwargs:
            args.test_env_kwargs["chg_tex"] = True
        else:
            args.test_env_kwargs = {"chg_tex": True}

    args.n_envs = hyperparams["n_envs"]
    args.num_frame_stack = hyperparams["frame_stack"] if "frame_stack" in hyperparams else 1
    args.log_path = os.path.join(args.log, args.env_id)
    args.tile_size = hyperparams["tile_size"] if "tile_size" in hyperparams else 32
    args.gail_loss = hyperparams["gail_loss"] if "gail_loss" in hyperparams else 'gailfo'

    print("Training env params: {}".format(args.env_kwargs))
    print("Testing env params: {}".format(args.test_env_kwargs))
    
    return args


def create_env(hyperparams, args, seed, tune=False):
    vec_class = SubprocVecEnv if not tune else DummyVecEnv

    # Add wrappers to environment
    wrapper_kwargs = None
    if "time_limit" in hyperparams and hyperparams["time_limit"]:
        wrapper = TimeFeatureWrapper
    elif "atari" in hyperparams and hyperparams["atari"]:
        wrapper = AtariWrapper
    elif "mini_grid" in hyperparams and hyperparams["mini_grid"]:
        wrapper = mini_grid_wrapper
        if "tile_size" in hyperparams:
            wrapper_kwargs = {"tile_size": args.tile_size}
    else:
        wrapper = None

    if "mujoco_img" in hyperparams and hyperparams["mujoco_img"]:
        envs, eval_env = get_mujoco_img_vec_env(args.env_id, n_envs=args.n_envs, seed=seed,
                                                render_dim=args.render_dim)
    elif "mujoco_dict" in hyperparams and hyperparams["mujoco_dict"]:
        envs, eval_env = get_mujoco_vec_env(
            args.env_id, n_envs=args.n_envs, seed=seed, wrapper_class=wrapper
        )
    # elif "mini_grid" in hyperparams and hyperparams["mini_grid"]:
    #     envs, eval_env = get_grid_vec_env(args.env_id, n_envs=args.n_envs,
    #                                       tile_size=args.tile_size, seed=seed,
    #                                       vec_env_cls=vec_class)
    else:
        envs = make_vec_env(
            args.env_id,
            n_envs=args.n_envs,
            seed=seed,
            wrapper_class=wrapper,
            vec_env_cls=vec_class,
            env_kwargs=args.env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        eval_env = make_vec_env(
            args.env_id,
            1,
            wrapper_class=wrapper,
            env_kwargs=args.test_env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )

    # Transpose to channel first for pytorch convolutions
    if is_image_space(envs.observation_space):
        print("Image observations...")   
        
        if "frame_stack" in hyperparams:
            print("Stacking frames...")
            envs = VecFrameStack(envs, n_stack=hyperparams["frame_stack"])
            eval_env = VecFrameStack(eval_env, n_stack=hyperparams["frame_stack"])

        if not is_image_space_channels_first(envs.observation_space):
            print("Transposing images to be channel first")
            envs = VecTransposeImage(envs)

    # 3. More wrappers for environment
    # separate norm_env as it is done in https://github.com/HumanCompatibleAI/imitation
    if args.norm_obs:
        envs = norm_env = VecNormalize(envs, norm_obs=True, norm_reward=False)
    else:
        norm_env = envs

    eval_env.seed(seed)
    print(
        "envs.observation_space={}, envs.action_space={}".format(
            envs.observation_space, envs.action_space
        )
    )

    return envs, eval_env, norm_env


def modify_expert_data(expert_traj, envs, args):
    # 3. Preprocess expert data
    # Expert needs to be channel first
    expert_traj = modify_expert_data_for_train(expert_traj, envs, args.obs_only)
    print(f"After modifying, expert_traj: {expert_traj.shape}")
    if (
        is_image_space(envs.observation_space)
        and not np.argmin(expert_traj.size()) == 1
        and expert_traj.shape[-1] in [1, args.num_frame_stack, 3*args.num_frame_stack]
    ):
        # Data is NxHxWxC. Need to be transposed to NxCxHxW.
        print("Permuting expert data...")
        expert_traj = expert_traj.permute(0, 3, 1, 2)
        print(f"Expert after permutation : {expert_traj.size()}")

    return expert_traj


def main():  # noqa: C901
    args = get_args()

    print(f"Training {args.env_id}...")

    expert_traj = np.load("data/expert_{}.npy".format(args.env_id)).astype(np.float32)
    expert_traj = th.from_numpy(expert_traj)
    print("Expert data: ", expert_traj.shape)

    device = get_device(args.device)
    using_cuda = True if device == th.device("cuda") else False
    print(f"Device = {device}")

    args.world_size = 2

    if args.seed is not None:
        seed = args.seed
        if args.algo == "all":
            for algo, _ in ALGOS.items():
                set_random_seed(seed, using_cuda)
                hyperparams, saved_hyperparams = read_hyperparameters(
                    algo, args.env_id, verbose=args.verbose
                    )
                args = set_up_parameters(hyperparams, args)
                envs, eval_env, norm_env = create_env(hyperparams, args, seed)
                expert_traj = modify_expert_data(expert_traj, envs, args)
                ALGOS[algo].train(
                    envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj, device, seed
                )
        else:
            set_random_seed(seed, using_cuda)
            hyperparams, saved_hyperparams = read_hyperparameters(
                    args.algo, args.env_id, verbose=args.verbose
                    )
            args = set_up_parameters(hyperparams, args)
            envs, eval_env, norm_env = create_env(hyperparams, args, seed)
            expert_traj = modify_expert_data(expert_traj, envs, args)
            ALGOS[args.algo].train(
                envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj, device, seed
            )
    else:
        if args.algo == "all":
            for i in range(args.n_times):
                for algo, _ in ALGOS.items():
                    set_random_seed(seeds[i], using_cuda)
                    hyperparams, saved_hyperparams = read_hyperparameters(
                        algo, args.env_id, verbose=args.verbose
                        )
                    args = set_up_parameters(hyperparams, args)
                    envs, eval_env, norm_env = create_env(hyperparams, args, seeds[i])
                    expert_traj = modify_expert_data(expert_traj, envs, args)
                    ALGOS[algo].train(
                        envs,
                        eval_env,
                        norm_env,
                        args,
                        hyperparams,
                        saved_hyperparams,
                        expert_traj,
                        device,
                        seeds[i],
                    )
        else:
            for i in range(args.n_times):
                set_random_seed(seeds[i], using_cuda)
                hyperparams, saved_hyperparams = read_hyperparameters(
                    args.algo, args.env_id, verbose=args.verbose
                    )
                args = set_up_parameters(hyperparams, args)
                envs, eval_env, norm_env = create_env(hyperparams, args, seeds[i])
                expert_traj = modify_expert_data(expert_traj, envs, args)
                ALGOS[args.algo].train(
                    envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj, device, seeds[i]
                )


if __name__ == "__main__":
    main()
