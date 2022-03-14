import numpy as np


class ExpertData:
    def __init__(self) -> None:
        self.obs = None
        self.actions = None

    def add(self, obs, action):
        obs = np.array(obs)
        action = np.array(action)

        if self.obs is None:
            self.obs = obs
        else:
            self.obs = np.append(self.obs, obs, axis=0)

        if self.actions is None:
            self.actions = action
        else:
            self.actions = np.append(self.actions, action, axis=0)
