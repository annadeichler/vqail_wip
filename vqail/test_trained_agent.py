import argparse
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import preprocess_obs
import gym_miniworld
from gym_minigrid.wrappers import *
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1000,
                    help='random seed (default: 1000)')
parser.add_argument('--test-seed', type=int, default=1000,
                    help='random seed (default: 1000)')
parser.add_argument('--env-name', default='MiniWorld-PickupObjs-v0',
                    help='environment to train on (default: MiniWorld-PickupObjs-v0)')
parser.add_argument('--test-env-name', default=None,
                    help='environment to train on (default: None)')
parser.add_argument('--algo', default='VQVAIL',
                    help='algo to train on, GAIL, VAIL or VQVAIL (default: VQVAIL)')
parser.add_argument('--load-dir', default='./outputs/',
                    help='directory to load model (default: ./outputs/)')
parser.add_argument('--eval-episodes', type=int, default=100,
                    help='random seed (default: 100)')
parser.add_argument('--tile-size', type=int, default=8,
                    help='tile size for minigrid (default: 8)')
parser.add_argument(
        "--chg-box-color", default=False,
        help="Change box color of miniworld envs",
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

args = parser.parse_args()
args.env_kwargs = {"chg_box_color": True} if args.chg_box_color else None
if args.chg_tex_train_test or args.chg_tex_test:
    if args.env_kwargs:
        args.env_kwargs["chg_tex"] = True
    else:
        args.env_kwargs = {"chg_tex": True}

print("Testing env params: {}".format(args.env_kwargs))

if not args.test_env_name:
    test_env_name = args.env_name
else:
    test_env_name = args.test_env_name

if 'MiniGrid' in args.env_name:
    env = ImgObsWrapper(RGBImgObsWrapper(gym.make(test_env_name),
                        tile_size=args.tile_size))
else:
    env = make_vec_env(
                test_env_name, 1, wrapper_class=None, seed=args.test_seed,
                env_kwargs=args.env_kwargs,
            )
    env = VecFrameStack(env, 4)

try:
    env = VecTransposeImage(env)
except:
    pass
print('Environment observation space: ', env.observation_space.shape)

model = PPO.load("{}/{}/{}-{}-{}.zip".format(args.load_dir, args.env_name, args.algo, args.seed, args.env_name), env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes)
print('Mean + Std Dev: ', mean_reward, std_reward)

# Enjoy trained agent
obs = env.reset()
print('Observation type: ', type(obs))
rewards = []

for ep in range(10):
    steps = 0
    rew = 0
    while True:
        obs = np.array(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        steps += 1
        reward = reward if isinstance(reward, int) or isinstance(reward, float) else reward[0]
        rew += reward

        if done or steps > 400:
            obs = env.reset()
            rewards.append(rew)
            print("rew : {}, steps : {}".format(rew, steps))
            break

print("Rewards: ", rewards)
print("Mean reward: ", np.mean(rewards))
