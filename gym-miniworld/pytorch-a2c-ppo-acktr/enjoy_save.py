import argparse
import os
import types

import numpy as np
import torch

from vec_env.dummy_vec_env import DummyVecEnv
#from vec_env.vec_normalize import VecNormalize
from envs import VecPyTorch, make_vec_envs

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--episodes', default=5, type=int,
                    help='number of episodes to save')
parser.add_argument('--no-permute', default=False, action="store_true", help='permute data from NxCxWxH to NxWxHxC, else to NxHxWxC, to keep same as env obs space')
parser.add_argument('--chg-box-color', default=False, action="store_true", help='Change box color at random')
parser.add_argument('--chg-entity', default=False, action="store_true", help='Change the entity at random')
parser.add_argument("--reward-threshold", help="Reward threshold", type=float, default=0)
parser.add_argument("--len-threshold", help="Length threshold", type=int, default=None)

args = parser.parse_args()

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None, args.add_timestep, device='cpu',
                            allow_early_resets=False, chg_box_color=args.chg_box_color, chg_entity=args.chg_entity)
print(env.observation_space.shape)

# Get a render function
render_func = None
tmp_env = env
while True:
    if hasattr(tmp_env, 'envs'):
        render_func = tmp_env.envs[0].render
        break
    elif hasattr(tmp_env, 'venv'):
        tmp_env = tmp_env.venv
    elif hasattr(tmp_env, 'env'):
        tmp_env = tmp_env.env
    else:
        break

if hasattr(tmp_env, "envs"):
    max_steps = tmp_env.envs[0].max_episode_steps
else:
    max_steps = None

if not args.len_threshold:
    args.len_threshold = max_steps

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

actor_critic.eval()

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()
print("Observation shape: ", obs.numpy().shape)
traj = None

ep_rews, ep_lens = [], []

for ep in range(args.episodes):
    steps = 0
    rew = 0
    ep_traj = obs.numpy()
    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=False)

        # Observation, reward and next obs
        obs, reward, done, _ = env.step(action)
        ep_traj = np.append(ep_traj, obs.numpy(), axis=0)

        rew += reward[0]
        steps += 1

        if done:
            if rew >= args.reward_threshold and max_steps and steps < max_steps and args.len_threshold and steps < args.len_threshold:
                print(f"episode {ep}, reward {rew}, steps {steps}")
                ep_rews.append(rew)
                ep_lens.append(steps)

                if traj is None:
                    traj = ep_traj.copy()
                else:
                    traj = np.append(traj, ep_traj, axis=0)

            obs = env.reset()
            break

if traj is not None:
    if args.no_permute:
        print("before permute to N x H x W x C. : ", traj.shape)
        traj = torch.from_numpy(traj).permute(0, 3, 2, 1).numpy()
    else:
        # observation shape (N x C x W x H) different from original env observation space (N x C x H x W).
        print("before permute to N x W x H x C. : ", traj.shape)
        # traj = torch.from_numpy(traj).permute(0, 3, 2, 1).numpy()

        # Permute to N x W x H x C
        traj = torch.from_numpy(traj).permute(0, 2, 3, 1).numpy()

    print(traj.shape)
    path = os.path.join(args.load_dir, "expert_{}.npy".format(args.env_name))
    with open(path, "wb") as f:
        np.save(f, traj)

    print("Avg Reward: {}".format(np.average(ep_rews)))
    print("Avg Length: {}".format(np.average(ep_lens)))