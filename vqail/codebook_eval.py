# from io import SEEK_END
from  models import VQVAE, VQVAEImage,VQEmbeddingEMA,VQVAIL
from typing import Union
import os
import json
from collections import OrderedDict
import torch
import gym
import gym_miniworld

import gym_minigrid
import argparse
import yaml
from yaml.loader import SafeLoader
from util import *

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv

from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from env_utils import get_mujoco_vec_env, get_mujoco_img_vec_env, mini_grid_wrapper
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from wrappers import VecCustomReward

WANDB_DIR = "./wandb/"
RESULTS_DIR="./results/"
D_ALGO= {"gail":"GAILAlgorithm","vail":"VAILAlgorithm","vqail":"VQAILAlgorithm"}

def get_checkpoints(rdir,rdata,run_id):
    base_dir = os.path.join(rdir,rdata['env_id'])
    run_dir = [os.path.join(base_dir,f) for f in os.listdir(base_dir) if run_id in f][0]
    ckp_dir = os.path.join(run_dir,'checkpoints')
    d={int(c.split('-')[-1].strip('.pt')):os.path.join(run_dir,'checkpoints',c) for c in os.listdir(ckp_dir)}

    return OrderedDict(sorted(d.items()))

def get_seed(rdir,rdata,run_id):
    base_dir = os.path.join(rdir,rdata['env_id'])
    run_dir = [os.path.join(base_dir,f) for f in os.listdir(base_dir) if run_id in f][0]

    return int(os.path.basename(os.path.normpath(run_dir)).split('_')[2])

def get_p(d,p):
    return d['config'][p]['value']

def read_json_file(filename):
    with open(filename, 'r') as f:
        cache = f.read()
        data = eval(cache)
    return data

def get_wandb_data(run_id):
    data={}
    fdir = [f for f in os.listdir(WANDB_DIR) if run_id in f][0]
    pconfig = os.path.join(WANDB_DIR,fdir,'files','config.yaml')
    with open(pconfig) as f:
        data['config'] = yaml.load(f, Loader=SafeLoader)        
    data['ep_rew_mean']=json.load(open(os.path.join(WANDB_DIR,fdir,'files','wandb-summary.json')))['ep_rew_mean']
    jargs=json.load(open(os.path.join(WANDB_DIR,fdir,'files','wandb-metadata.json')))['args']
    data['algo'] = jargs[jargs.index('--algo')+1]
    data['env_id']  = jargs[jargs.index('--env-id')+1]
    data['reg']  = jargs[jargs.index('--reg')+1]
    return data



def modify_expert_data(expert_traj, envs, hypes):
    # 3. Preprocess expert data
    # Expert needs to be channel first
    obs_only=hypes.get('obs_only',False)
    num_frame_stack =hypes.get('num_fram_stack',False)
    expert_traj = modify_expert_data_for_train(expert_traj, envs, obs_only)
    print(f"After modifying, expert_traj: {expert_traj.shape}")
    if (
        is_image_space(envs.observation_space)
        and not np.argmin(expert_traj.size()) == 1
        and expert_traj.shape[-1] in [1, num_frame_stack, 3*num_frame_stack]
    ):
        # Data is NxHxWxC. Need to be transposed to NxCxHxW.
        print("Permuting expert data...")
        expert_traj = expert_traj.permute(0, 3, 1, 2)
        print(f"Expert after permutation : {expert_traj.size()}")

    return expert_traj

def create_env(d_hypes,d_run,seed,env_kwargs={},tune=False):
    vec_class = SubprocVecEnv if not tune else DummyVecEnv
    wrapper_kwargs = None
    n_envs=d_hypes.get('n_envs',1)
    tile_size = d_hypes.get('tile_size',32)
    norm_obs =d_hypes.get('norm_obs',False)   

    if "time_limit" in d_hypes and d_hypes["time_limit"]:
        wrapper = TimeFeatureWrapper
    elif "mini_grid" in d_hypes and d_hypes["mini_grid"]:
        wrapper = mini_grid_wrapper
        if "tile_size" in d_hypes:
            wrapper_kwargs = {"tile_size":tile_size}
    else:
        wrapper = None

    if "mujoco_img" in d_hypes and d_hypes["mujoco_img"]:
        envs, eval_env = get_mujoco_img_vec_env(args.env_id, n_envs=n_envs, seed=seed,render_dim=args.render_dim)
    elif "mujoco_dict" in d_hypes and d_hypes["mujoco_dict"]:
        envs, eval_env = get_mujoco_vec_env(
            args.env_id, n_envs=n_envs, seed=seed, wrapper_class=wrapper
        )
    else:
        envs = make_vec_env(
            d_run['env_id'],
            n_envs=n_envs,
            seed=seed,
            wrapper_class=wrapper,
            vec_env_cls=vec_class,
            env_kwargs=env_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
    # Transpose to channel first for pytorch convolutions
    if is_image_space(envs.observation_space):
        print("Image observations...")   
        if "frame_stack" in d_hypes:
            print("Stacking frames...")
            envs = VecFrameStack(envs, n_stack=d_hypes["frame_stack"])
            # eval_env = VecFrameStack(eval_env, n_stack=d_hypes["frame_stack"])

        if not is_image_space_channels_first(envs.observation_space):
            print("Transposing images to be channel first")
            envs = VecTransposeImage(envs)

    # 3. More wrappers for environment
    # separate norm_env as it is done in https://github.com/HumanCompatibleAI/imitation
    if norm_obs:
        envs = norm_env = VecNormalize(envs, norm_obs=True, norm_reward=False)
    else:
        norm_env = envs

    print(
        "envs.observation_space={}, envs.action_space={}".format(
            envs.observation_space, envs.action_space
        )
    )
    return envs, norm_env

def load_model(args,envs,d_run,d_hypes,d_ckpt,verbose=False):

    n_embeddings = get_p(d_run,"n_embeddings")
    hidden_size = get_p(d_run, "hidden_size")
    embedding_dim = get_p(d_run,"embedding_dim")
    embedding_dim = d_hypes["n_envs"] * embedding_dim
    regularization = d_run["reg"]
    
    codebook = VQEmbeddingEMA(
        n_embeddings=n_embeddings, embedding_dim=embedding_dim,
        regularization=regularization, device=device
    )
    # Uses LfO (Learning from observations).
    model = VQVAEImage(
        envs.observation_space, hidden_size, embedding_dim, codebook, device
    )
    ckpt_path=d_ckpt[next(reversed(d_ckpt))]
    if verbose:
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    d_checkpoint = torch.load(ckpt_path,map_location=device)
    print(d_checkpoint.keys())
    d_model = d_checkpoint['model_state_dict']
    model.load_state_dict(d_model)

    return model


        
# def run_inference(model,expert_traj):

#        expert_state_action = expert_traj[
#             np.random.randint(0, self.expert_traj.shape[0], num_samples), :
#         ]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        help="wandb sweep id of tuning",
        type=str,
        required=False,
    )
    args=parser.parse_args()
    run_id=args.run_id
    run_id = '216rxrhd'

    device=get_device("cuda")
    run_data= get_wandb_data(run_id)
    hypes, saved_hyperparams=read_hyperparameters(run_data['algo'], run_data['env_id'])
    
    run_data["norm_obs"] = True if "norm_obs" in hypes and hypes["norm_obs"] else False
    d_ckpt = get_checkpoints(RESULTS_DIR,run_data,run_id)
    seed=get_seed(RESULTS_DIR,run_data,run_id)


    envs,norm_env = create_env(hypes, run_data, seed)
    expert_traj = np.load("data/expert_{}.npy".format(run_data["env_id"])).astype(np.float32)
    expert_traj = th.from_numpy(expert_traj)
    expert_traj = modify_expert_data(expert_traj, envs, hypes)
    print(expert_traj.shape)

    model = load_model(args,envs,run_data,hypes,d_ckpt,True)


    # envs = VecCustomReward(
    #     model,
    #     envs,
    #     train=True,
    #     obs_only=args.obs_only,
    #     reward_factor=args.reward_factor,
    #     device=device,
    # )

    policy_kwargs = eval(hypes["policy_kwargs"])
    policy_kwargs.update({"env": envs})
    algo = VQVAIL(
        PPO,
        envs,
        model,
        expert_traj=expert_traj,
        device=device,
        seed=seed,
        normalize=hypes.get('norm_obs',False),
        norm_env=norm_env,
        obs_only=hypes.get('obs_only',False),
        env_name=run_data["env_id"],
        policy_kwargs=policy_kwargs,
    )
    print(expert_traj.shape)


    algo.train_vqvae(algo.ts)

 
    print(model)

    