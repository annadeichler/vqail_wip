import argparse
import os
import types

import gym
import gym_miniworld
from gym.wrappers import Monitor
import numpy as np
import torch
from vec_env.dummy_vec_env import DummyVecEnv
from envs import VecPyTorch, VecPyTorchFrameStack


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 1, 0)

def make_env(env_id, seed):
    def _thunk():

        env = gym.make(args.env_name)
        env.seed(seed)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk

def make_vec_envs(env_name, seed=1):
    envs = [make_env(env_name, seed)]
    envs = DummyVecEnv(envs)
    envs = VecPyTorch(envs, "cpu")

    if len(envs.observation_space.shape) == 3:
        print('Creating frame stacking wrapper')
        envs = VecPyTorchFrameStack(envs, 4, "cpu")

    return envs


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--env-name', default='MiniWorld-PickupObjs-v0',
                    help='environment to train on (default: MiniWorld-PickupObjs-v0)')
parser.add_argument('--load-dir', default='trained_models/ppo/',
                    help='directory to save agent logs (default: trained_models/ppo/)')
args = parser.parse_args()

env = make_vec_envs(args.env_name, seed=1)
env2 = Monitor(gym.make(args.env_name), './video', force=True)
env2.seed(1)

state = env.reset()
_ = env2.reset()
done = False


actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

actor_critic.eval()
recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

rewards = 0.0
steps = 0
while not done:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            state, recurrent_hidden_states, masks, deterministic=True)
        state, reward, done, info = env.step(action)
        _ = env2.step(action)
        rewards += reward
        steps += 1
    masks.fill_(0.0 if done else 1.0)
env.close()
env2.close()
print("Reward: ", rewards)
print("Steps: ", steps)