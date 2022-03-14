# import gym
import gym_miniworld
import matplotlib.pyplot as plt
from colabgymrender.recorder import Recorder


env = gym.make('MiniWorld-TMaze-v0')
env = Recorder(env, './video')
first_obs = env.reset()

env.play()
