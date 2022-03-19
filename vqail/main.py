import argparse
import os
import json
from stable_baselines3.common.utils import set_random_seed
from arguments import get_args
from train_models import train_gail, train_vail, train_vqvail

from util import get_device, read_hypes, update_config_hypes, set_random_seed
from train_utils import  set_up_parameters, create_env, modify_expert_data
import torch as th
import numpy as np 
import wandb

DIR_CONFIG = './train_configs'
ALGOS = {"gail": train_gail, "vail": train_vail, "vqail": train_vqvail}
# choices = list(ALGOS.keys()).append("all")


def load_expert_data(args):

    if args.num_objs!=0:
        expert_traj = np.load("data/expert_{}_{}.npy".format(args.env_id,str(args.num_objs))).astype(np.float32)
    else:
        expert_traj = np.load("data/expert_{}.npy".format(args.env_id)).astype(np.float32)
    expert_traj = th.from_numpy(expert_traj)
    print(expert_traj.shape)
    return expert_traj

def load_configs(args):

    hyperparams = read_hypes(
        args.algo, args.env_id, verbose=args.verbose)

    if args.config_id!="":
        cdir = os.path.join(DIR_CONFIG,args.env_id)
        fp = [f for f in os.listdir(cdir) if args.algo in f and str(args.config_id) in f ][0]
        with open(f"./train_configs/{args.env_id}/{fp}", "r") as f:
            config = json.load(f)
        config['parameters']['run_id']=fp.split('_')[-1].strip('.json')
        hyperparams=update_config_hypes(args,hyperparams,config['parameters'])

    device = get_device(args.device)
    print(f"Device = {device}")
    return hyperparams,device,config
    
def train(args,hyperparams,device,expert_traj,seed,run_id):

    name = args.algo.upper() + "Algorithm"
    tb_log_name = '_'.join([name , run_id, str(seed), str(args.cuda_id)])
    using_cuda = True if device == th.device("cuda") else False
    set_random_seed(seed, using_cuda)
    args = set_up_parameters(hyperparams, args)
    set_random_seed(seed, using_cuda)
    envs, eval_env, norm_env = create_env(hyperparams, args, int(seed), tune=True)
    expert_traj = modify_expert_data(expert_traj, envs, args)
    ALGOS[args.algo].train(
    envs, eval_env, norm_env, args, hyperparams, {}, expert_traj,
    device, int(seed), tune=args.log_wandb,tb_log_name=tb_log_name,run_id=run_id,cuda_id=args.cuda_id
    )

def main(args):
    hypes,device,config=load_configs(args)
    expert_traj = load_expert_data(args)
    print(f"Training {args.env_id}...")
    seeds = np.random.randint(3,1e4,args.n_seeds)
    for seed in seeds:
        if args.log_wandb==True:
            project_name = "_".join(["train",args.env_id,args.algo,args.tag])
            with wandb.init(project=project_name,config=config['parameters']) as run:
                train(args,hypes,device,expert_traj,seed,"",run.id)
        else:
            train(args,hypes,device,expert_traj,seed,"")
    

if __name__ == "__main__":
    args = get_args()
    main(args)


