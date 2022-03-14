import os
import argparse
import functools
import cProfile, pstats
import torch as th
import numpy as np
import gym
import wandb
import gym_miniworld
from stable_baselines3.common.utils import set_random_seed

from train_models import train_gail, train_vail, train_vqvail
from util import read_hyperparameters, get_device


ALGOS = {"gail": train_gail, "vail": train_vail, "vqail": train_vqvail}
choices = list(ALGOS.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id", help="environment ID", type=str, default="CartPole-v1"
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm (gail, vail or vqail)",
        default="vqail",
        type=str,
        required=True,
        choices=choices,
    )
    parser.add_argument(
        "-log",
        "--log",
        help="Log folder for tensorboard",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--timesteps",
        help="override the default timesteps to run (default 10)",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--seed", type=int, default=1000, help="random seed (default: 1000)"
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="verbose 0 or 1 (default: 0)"
    )
    parser.add_argument(
        "--plot-umap",
        default=False,
        action="store_true",
        help="Plot umap projection for vqvail, saved in plots/",
    )
    parser.add_argument("--n-times", type=int, help="Repeat n times", default=1)
    parser.add_argument("--device", help="Device", type=str, default="auto")
    parser.add_argument(
        "--chg-box-color", default=False,
        help="Change box color of miniworld envs",
        action="store_true",
    )

    args = parser.parse_args()

    print(f"Training {args.env_id}...")

    global hyperparams
    global saved_hyperparams
    hyperparams, saved_hyperparams = read_hyperparameters(
        args.algo, args.env_id, verbose=args.verbose
    )

    global expert_traj
    expert_traj = np.load("data/expert_{}.npy".format(args.env_id)).astype(np.float32)
    expert_traj = th.from_numpy(expert_traj)
    print(expert_traj.shape)

    global device
    device = get_device(args.device)
    print(f"Device = {device}")

    tb_log_name = os.path.join(args.log, args.env_id)

    using_cuda = True if device == th.device("cuda") else False

    set_random_seed(args.seed, using_cuda)

    profiler = cProfile.Profile()
    profiler.enable()
    ALGOS[args.algo].train(
        args, hyperparams, saved_hyperparams, expert_traj,
        device, args.seed, tune=False, tb_log_name=tb_log_name
    )
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
    save_path = os.path.join(tb_log_name, 'profile-output')
    stats.dump_stats(save_path)
