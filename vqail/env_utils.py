import os
from typing import Any, Dict, Optional, Type, Union, Callable, Tuple
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import dmc2gym
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from wrappers import DictImgObsWrapper, DictObsWrapper


def mini_grid_wrapper(env: gym.Env, tile_size: int = 32) -> gym.Env:
    env = ImgObsWrapper(RGBImgObsWrapper(env, tile_size=tile_size))
    return env

def make_mujoco_img_env(
    env_id: str,
    rank: int,
    wrapper_class: None,
    seed: int = 0,
    render_dim: int = 80,
    monitor_dir: Optional[str] = None,
) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id, reward_type="sparse")
        if wrapper_class:
            env = wrapper_class(env)
        env = DictImgObsWrapper(env, render_dim=render_dim)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        monitor_kwargs = {}
        monitor_path = (
            os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        )
        # Create the monitor folder if needed
        if monitor_path is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=monitor_path, **monitor_kwargs)

        return env

    return _init


def make_dmc_env(
    env_id: str,
    rank: int,
    wrapper_class: None,
    seed: int = 0,
    monitor_dir: Optional[str] = None,
) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        # env = gym.make(env_id, reward_type="sparse")
        env = dmc2gym.make(domain_name=env_id, visualize_reward=False, from_pixels=True, seed=seed)
        if wrapper_class:
            env = wrapper_class(env)
        env = DictObsWrapper(env)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        monitor_kwargs = {}
        monitor_path = (
            os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        )
        # Create the monitor folder if needed
        if monitor_path is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=monitor_path, **monitor_kwargs)

        return env

    return _init

def make_mujoco_env(
    env_id: str,
    rank: int,
    wrapper_class: None,
    seed: int = 0,
    monitor_dir: Optional[str] = None,
) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id, reward_type="sparse")
        if wrapper_class:
            env = wrapper_class(env)
        env = DictObsWrapper(env)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        monitor_kwargs = {}
        monitor_path = (
            os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        )
        # Create the monitor folder if needed
        if monitor_path is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=monitor_path, **monitor_kwargs)

        return env

    return _init


def get_mujoco_img_vec_env(
    env_id,
    n_envs: int = 1,
    seed: int = 0,
    render_dim: int = 80,
    wrapper_class=None,
    vec_env_cls=SubprocVecEnv,
):
    venv = vec_env_cls(
        [make_mujoco_img_env(env_id, i, wrapper_class, seed, render_dim) for i in range(n_envs)]
    )
    eval_env = DictImgObsWrapper(gym.make(env_id, reward_type="sparse"), render_dim=render_dim)
    return venv, eval_env


def get_mujoco_vec_env(
    env_id,
    n_envs: int = 1,
    seed: int = 0,
    wrapper_class=None,
    vec_env_cls=SubprocVecEnv,
):
    venv = vec_env_cls(
        [make_mujoco_env(env_id, i, wrapper_class, seed) for i in range(n_envs)]
    )
    eval_env = DictObsWrapper(wrapper_class(gym.make(env_id, reward_type="sparse")))
    return venv, eval_env

