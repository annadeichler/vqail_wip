from typing import Union
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import gym
from gym import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.preprocessing import (
    preprocess_obs,
    is_image_space_channels_first,
    is_image_space,
)
from util import get_device


class VecCustomReward(VecEnvWrapper):
    def __init__(
        self,
        model,
        venv: VecEnv,
        train: bool = True,
        obs_only: bool = False,
        reward_factor: int = -1,
        device: Union[th.device, str] = "auto",
    ):
        VecEnvWrapper.__init__(self, venv)

        self.model = model
        self.device = get_device(device) if isinstance(device, str) else device
        print(f"Device in custom reward wrapper: {self.device}")
        self.train = train
        self.obs_only = obs_only
        self.reward_factor = reward_factor

    def reset(self):
        obs = self.venv.reset()
        # plt.imshow(th.FloatTensor(obs[0][9:, :, :]).permute(1, 2, 0))
        # plt.savefig("/content/state.png")
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        if self.train:
            rewards = self._update_reward(obs, self._actions, self.venv)

        return obs, rewards, dones, infos

    def _update_reward(self, states, actions, env):
        """Update reward using discriminator."""
        states = th.from_numpy(states)
        states = preprocess_obs(states, env.observation_space)
        if is_image_space(env.observation_space) and not is_image_space_channels_first(
            env.observation_space
        ):
            # Observations are NxHxWxC. Need to be transposed to NxCxHxW.
            states = states.permute(0, 3, 1, 2)

        if self.obs_only:
            state_action = states.to(self.device)
        else:
            actions = th.from_numpy(actions)
            actions = preprocess_obs(actions, env.action_space)
            state_action = th.cat(
                [
                    states.reshape(states.shape[0], -1),
                    actions.reshape(actions.shape[0], -1),
                ],
                1,
            ).to(self.device)

        # model is channel first
        value = self.model(state_action)
        if isinstance(value, tuple):
            value = value[0]

        return self.reward_factor * np.log(value.view(-1).cpu().data.numpy())


class DictImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output.
    """

    def __init__(self, env, render_dim=200):
        super().__init__(env)

        self.render_dim = render_dim

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(render_dim, render_dim, 3), dtype="uint8"
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode="rgb_array", width=self.render_dim, height=self.render_dim
        )
        rgb_img = rgb_img.astype(np.uint8)

        return rgb_img


class DictObsWrapper(gym.core.ObservationWrapper):
    """
    Use only observation output.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = env.observation_space.spaces["observation"]

    def observation(self, obs):
        return obs["observation"]
