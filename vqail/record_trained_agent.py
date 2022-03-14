import argparse
import os
import types

import torch
import gym
import numpy as np
import gym_miniworld
from gym.wrappers import Monitor
from stable_baselines3 import PPO

from util import read_hyperparameters
from train import set_up_parameters, create_env

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    "--env-id", help="environment ID", type=str, default="CartPole-v1"
)
parser.add_argument(
    "--algo",
    help="RL Algorithm (gail, vail or vqvail, or 'all')",
    default="vqail",
    type=str,
    required=True,
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
    help="override the default timesteps to run",
    default=None,
    type=int,
)
parser.add_argument(
    "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "--verbose", type=int, default=1, help="verbose 0 or 1 (default: 1)"
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
parser.add_argument(
    "--render-dim", default=80,
    help="Render dim for fetch pick env", type=int
)
parser.add_argument(
    "--gpu-optimize", default=False,
    help="Optimize GPU by clearing cache and deleting cuda objects",
    action="store_true",
)
parser.add_argument(
    "--save-model-interval", type=int, default=100, help="Save model interval (default: 100)"
)
parser.add_argument(
    "--top-view", default=False,
    help="Render top view observations for miniworld env",
    action="store_true",
)
parser.add_argument(
    "--chg-tex-train-test", default=False,
    help="Change texture of walls/ceilings at train and test time for miniworld env",
    action="store_true",
)
parser.add_argument(
    "--chg-tex-test", default=False,
    help="Change texture of walls/ceilings at test time for miniworld env",
    action="store_true",
)
parser.add_argument(
    "--model-path",
    help="Full model path",
    type=str,
    default=None,
)
parser.add_argument(
    "--log-dir",
    help="Dir for model path",
    type=str,
    default="./outputs",
)
args = parser.parse_args()

if not args.model_path:
    print("Please provide model path --model-path")
    exit(0)

hyperparams, saved_hyperparams = read_hyperparameters(
                        args.algo, args.env_id, verbose=args.verbose
                        )
args = set_up_parameters(hyperparams, args)
_, env, _ = create_env(hyperparams, args, args.seed, tune=True)
env2 = Monitor(gym.make(args.env_id), './video/{}'.format(args.env_id), force=True)
env2.seed(args.seed)

state = env.reset()
_ = env2.reset()
done = False

if args.model_path:
    model = PPO.load(args.model_path, env=env)
else:
    print("Please provide model path --model-path")
    exit(0)

rewards = 0.0
steps = 0
while not done:
    with torch.no_grad():
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        _ = env2.step(action)
        rewards += reward
        steps += 1

env.close()
env2.close()
print("Reward: ", rewards)
print("Steps: ", steps)