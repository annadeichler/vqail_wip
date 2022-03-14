#!/usr/bin/env python3

import os
import numpy as np
import math
import matplotlib.pyplot as plt
import gym
import torch
import gym_miniworld
from gym_miniworld.wrappers import PyTorchObsWrapper, GreyscaleWrapper
from gym_miniworld.entity import TextFrame
from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import DummyVecEnv as dummy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


class VecEnv(ABC):

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    """
    An abstract asynchronous, vectorized environment.
    """
    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: an array of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        #logger.warn('Render not defined for %s'%self)
        pass

class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self):
        self.venv.render()


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if done: 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        return obs, reward, done, info


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.low.shape[0]
        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape)
        self.stackedobs = torch.from_numpy(self.stackedobs).float()
        self.stackedobs = self.stackedobs.to(device)
        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs[:, :-self.shape_dim0] = self.stackedobs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs.fill_(0)
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()


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


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets,
             chg_box_color=False, chg_entity=False, top_view=False):
    def _thunk():
        # chg_box_color and chg_entity are not common to all environments
        if chg_box_color and chg_entity:
            env = gym.make(env_id, chg_box_color=chg_box_color, chg_entity=chg_entity, top_view=top_view)
        elif chg_entity:
            env = gym.make(env_id, chg_entity=chg_entity, top_view=top_view)
        elif chg_box_color:
            env = gym.make(env_id, chg_box_color=chg_box_color, top_view=top_view)
        else:
            env = gym.make(env_id, top_view=top_view)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        #if log_dir is not None:
        #    env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
        #                        allow_early_resets=allow_early_resets)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep, device, allow_early_resets,
                  chg_box_color=False, chg_entity=False, top_view=False):
    envs = [make_env(env_name, seed, i, log_dir, add_timestep,
                     allow_early_resets, chg_box_color, chg_entity, top_view) for i in range(num_processes)]

    envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if len(envs.observation_space.shape) == 3:
        print('Creating frame stacking wrapper')
        envs = VecPyTorchFrameStack(envs, 4, device)
        #print(envs.observation_space)

    return envs


env_name = 'MiniWorld-OneRoom-v0'
seed = 0
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
path = "/content/" #"/tmp/gym/"

print("Testing single gym env observations...")
env = gym.make(env_name, chg_box_color=True)
env.seed(seed)
for i in range(3):
    obs = env.reset()
    print("env shape: ", env.observation_space.shape)
    plt.imshow(obs)
    plt.title("single gym env obs")
    plt.savefig(path + "obs1_{}.png".format(str(i)))

print("Testing vectorized gym env observations...")
envs = make_vec_envs(env_name, seed, 1,
                     0.99, "/tmp/gym/", False, device, allow_early_resets=False,
                     chg_box_color=False, chg_entity=False, top_view=False)
obs = envs.reset()
print("env shape: ", envs.observation_space.shape)
print("obs shape: ", obs.shape)
obs = obs[0].permute(2, 1, 0).int().cpu().numpy()
obs4 = obs[:, :, 9:]
plt.imshow(obs4)
plt.title("vec gym env obs 4")
plt.savefig(path + "obs2_4.png")

print("Testing vectorized gym env observations in top view...")
envs = make_vec_envs(env_name, seed, 1,
                     0.99, "/tmp/gym/", False, device, allow_early_resets=False,
                     chg_box_color=False, chg_entity=False, top_view=True)
obs = envs.reset()
print("env shape: ", envs.observation_space.shape)
print("obs shape: ", obs.shape)
obs = obs[0].permute(2, 1, 0).int().cpu().numpy()
obs4 = obs[:, :, 9:]
plt.imshow(obs4)
plt.title("vec gym env obs 4 top view")
plt.savefig(path + "obs3_4.png")

print("Testing SB3 env observations...")
envs = make_vec_env(
            env_name,
            n_envs=1,
            seed=seed,
            wrapper_class=None,
            vec_env_cls=dummy,
            env_kwargs=None,
        )
envs = VecFrameStack(envs, n_stack=4)
envs = VecTransposeImage(envs)
obs = envs.reset()
print("env shape: ", envs.observation_space.shape)
print("obs shape: ", obs.shape)
obs = torch.LongTensor(obs[0]).permute(2, 1, 0).int().cpu().numpy()
obs4 = obs[:, :, 9:]
plt.imshow(obs4)
plt.title("sb3 vec gym env obs 4")
plt.savefig(path + "obs4_4.png")

print("Testing SB3 env observations top view...")
envs = make_vec_env(
            env_name,
            n_envs=1,
            seed=seed,
            wrapper_class=None,
            vec_env_cls=dummy,
            env_kwargs={"top_view": True},
        )
envs = VecFrameStack(envs, n_stack=4)
envs = VecTransposeImage(envs)
obs = envs.reset()
print("env shape: ", envs.observation_space.shape)
print("obs shape: ", obs.shape)
obs = torch.LongTensor(obs[0]).permute(2, 1, 0).int().cpu().numpy()
obs4 = obs[:, :, 9:]
plt.imshow(obs4)
plt.title("sb3 vec gym env obs 4 top view")
plt.savefig(path + "obs5_4.png")