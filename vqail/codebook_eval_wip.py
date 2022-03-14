from  models import VQVAE, VQVAEImage,VQEmbeddingEMA,VQVAIL
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from env_utils import get_mujoco_vec_env, get_mujoco_img_vec_env, mini_grid_wrapper
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from wrappers import VecCustomReward
from util import (
    get_obs_action_dim,
    save_eval_rewards,
    evaluate_model,
)
from collections import OrderedDict

import torch
import yaml
from yaml.loader import SafeLoader


import argparse
import functools
from pprint import pprint
import torch as th
import numpy as np
import gym
import os
import gym_miniworld
import wandb
import json
from stable_baselines3.common.utils import set_random_seed
import torch.nn as nn

from train_models import train_gail, train_vail, train_vqvail
from train import set_up_parameters, create_env, modify_expert_data
from util import read_hyperparameters, get_device, read_sweepconfig
from arguments import get_args


DIR_CONFIG = './final_configs'
ALGOS = {"gail": train_gail, "vail": train_vail, "vqail": train_vqvail}
choices = list(ALGOS.keys()).append("all")

WANDB_DIR = "./wandb/"
RESULTS_DIR="./results/"
D_ALGO= {"gail":"GAILAlgorithm","vail":"VAILAlgorithm","vqail":"VQAILAlgorithm"}


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
    try:
        data['num_objs']  = jargs[jargs.index('--num_objs')+1]
    except: ValueError
    data['seed_config'] = jargs[jargs.index('--seed')+1]
    data['reg']  = jargs[jargs.index('--reg')+1]
    data['seed'] = get_seed(RESULTS_DIR,data,run_id)
    return data



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


def update_config_hypes(hyperparams,config):
    
    hyperparams["learning_rate"] = config["learning_rate"]
    hyperparams["gail_loss"] = config["gail_loss"]
    try:
        hyperparams["reward_factor"] = config["reward_factor"]
    except: KeyError
    hyperparams.update({"policy_kwargs": eval(hyperparams["policy_kwargs"])})
    hyperparams["policy_kwargs"]["n_steps"] = config["n_steps"]
    hyperparams["policy_kwargs"]["ent_coef"] = config["ent_coef"]
    hyperparams["policy_kwargs"]["batch_size"] = config["batch_size"]
    hyperparams["policy_kwargs"]["learning_rate"] = config["policy_lr"]

    try:
        hyperparams["policy_kwargs"]["gae_lambda"] = config["gae_lambda"]
    except: KeyError

    hyperparams.update({"policy_kwargs": str(hyperparams["policy_kwargs"])})

    if args.algo == 'gail':
        hyperparams.update({"gail": eval(hyperparams["gail"])})
        hyperparams["gail"]["discrim_hidden_size"] = config["discrim_hidden_size"]
        hyperparams.update({"gail": str(hyperparams["gail"])})
    elif args.algo == 'vail':
        hyperparams.update({"vail": eval(hyperparams["vail"])})
        hyperparams["vail"]["latent_size"] = config["latent_size"]
        hyperparams.update({"vail": str(hyperparams["vail"])})
    elif args.algo == 'vqail':
        hyperparams.update({"vqvail": eval(hyperparams["vqvail"])})
        hyperparams["vqvail"]["embedding_dim"] = config["embedding_dim"]
        hyperparams["vqvail"]["hidden_size"] = config["hidden_size"]
        hyperparams["vqvail"]["n_embeddings"] = config["n_embeddings"]
        hyperparams.update({"vqvail": str(hyperparams["vqvail"])})
    
def get_p(d,p):
    return d['config'][p]['value']

def train(args,fp,expert_traj,device,config_dict):
    run_id = 'qwt7ekfv'

    device=get_device("cuda")
    run_data= get_wandb_data(run_id)
    hypes, saved_hyperparams=read_hyperparameters(run_data['algo'], run_data['env_id'])
    
    run_data["norm_obs"] = True if "norm_obs" in hypes and hypes["norm_obs"] else False
    d_ckpt = get_checkpoints(RESULTS_DIR,run_data,run_id)
    ckpt_path=d_ckpt[next(reversed(d_ckpt))]

    seed=get_seed(RESULTS_DIR,run_data,run_id)



    # seeds = np.random.randint(3,1000,10)
    # seed = np.random.randint(3,1000,1)

    # for seed in seeds:
    seed=int(seed)
    # tune_run_id=fp.split('_')[-2]
    # project_name = "_".join(["final_train",args.env_id,args.algo,tune_run_id])
    # print(project_name)
    # with wandb.init(project=project_name,config=config_dict) as run:
    update_config_hypes(hyperparams,config_dict)
    name = args.algo.upper() + "Algorithm"
    # tb_log_name = '_'.join([name , run.id, str(seed), str(args.cuda_id)])
    using_cuda = True if device == th.device("cuda") else False
    pprint(saved_hyperparams)
    args = set_up_parameters(hyperparams, args)
    set_random_seed(seed, using_cuda)
    envs, eval_env, norm_env = create_env(hyperparams, args, seed, tune=True)
    expert_traj = modify_expert_data(expert_traj, envs, args)
    
    # 2. Instantiate model
    print(f"Device in train_vqvail.py = {device}")

    vqvail_params = eval(hyperparams["vqvail"])
    # n_embeddings = vqvail_params["n_embeddings"]
    n_embeddings = get_p(run_data,"n_embeddings")
    print("EMBEDDING SIZE - " + str(n_embeddings))
    hidden_size = get_p(run_data,"hidden_size")
    embedding_dim = get_p(run_data,"embedding_dim")
    embedding_dim = hypes['n_envs'] * embedding_dim
    regularization = "expire codes"

    codebook = VQEmbeddingEMA(
        n_embeddings=n_embeddings, embedding_dim=embedding_dim,
        regularization=regularization, device=device
    )
    if is_image_space(envs.observation_space):
        # Uses LfO (Learning from observations).
        model = VQVAEImage(
            envs.observation_space, hidden_size, embedding_dim, codebook, device
        )
    else:
        # Flattened observations and action dimensions
        # Uses LfD (learning from demonstrations).
        num_inputs, num_outputs = get_obs_action_dim(envs)
        print("num_inputs={}, num_outputs={}".format(num_inputs, num_outputs))
        model = VQVAE(
            num_inputs + num_outputs, hidden_size, embedding_dim, codebook, device
        )

    envs = VecCustomReward(
        model,
        envs,
        train=True,
        obs_only=args.obs_only,
        reward_factor=args.reward_factor,
        device=device,
    )

    print("Environment observation shape: ", envs.observation_space)

    policy_kwargs = eval(hyperparams["policy_kwargs"])
    policy_kwargs.update({"env": envs})
    print("policy_kwargs: ", policy_kwargs)

    
    algo = VQVAIL(
        PPO,
        envs,
        model,
        expert_traj=expert_traj,
        tensorboard_log=args.log_path,
        device=device,
        verbose=args.verbose,
        seed=seed,
        learning_rate=args.lr,
        normalize=args.norm_obs,
        norm_env=norm_env,
        lr_schedule_main=args.lr_schedule_main,
        lr_schedule_agent=args.lr_schedule_agent,
        obs_only=args.obs_only,
        ent_decay=args.ent_decay,
        env_name=args.env_id,
        gpu_optimize=args.gpu_optimize,
        gail_loss=args.gail_loss,
        policy_kwargs=policy_kwargs,
    )
    print(expert_traj.shape)
    # algo.get_latents(algo.ts,True)

    d_checkpoint = torch.load(ckpt_path,map_location=device)
    d_model = d_checkpoint['model_state_dict']
    model.load_state_dict(d_model)


    real_z,real_q_z =  algo.get_latents(algo.ts,True)
    # print(fake_q_z.shape)

def load_config(rdata):
    env_id = rdata["env_id"]
    cdir= os.path.join(DIR_CONFIG,env_id)
    print([f for f in os.listdir(cdir)])
    seed=rdata["seed_config"]
    fp = [f for f in os.listdir(cdir) if args.algo in f and str(seed) in f ][0]
    with open(f"./final_configs/{env_id}/{fp}", "r") as f:
        config = json.load(f)
    return config,fp


def main(hyperparams,d_run, expert_traj, device):
    print("running training")
    config,fp=load_config(d_run)
    train(args,fp, expert_traj, device,config['parameters'])

if __name__ == "__main__":
    args = get_args()


    run_id = 'qwt7ekfv'
    device="cuda"
    device=get_device(device)
    print(f"Device = {device}")
    run_data= get_wandb_data(run_id)
    hyperparams, saved_hyperparams = read_hyperparameters(
       run_data['algo'], run_data['env_id'], verbose=True
    )
    if "num_objects" in run_data:
        expert_traj = np.load("data/expert_{}_{}.npy".format(run_data['env_id'],str(run_data["num_objs"]))).astype(np.float32)
    else:
        expert_traj = np.load("data/expert_{}.npy".format(run_data['env_id'])).astype(np.float32)
    expert_traj = th.from_numpy(expert_traj)    

    main(hyperparams, run_data, expert_traj, device)
