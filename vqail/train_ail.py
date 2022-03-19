import argparse
import functools
from pprint import pprint
import torch as th
import numpy as np
import gym
import gym_miniworld
import wandb
from stable_baselines3.common.utils import set_random_seed
import torch.nn as nn

from train_models import train_gail, train_vail, train_vqvail
from train import set_up_parameters, create_env, modify_expert_data
from util import read_hyperparameters, get_device, read_sweepconfig
from arguments import get_args




ALGOS = {"gail": train_gail, "vail": train_vail, "vqail": train_vqvail}
choices = list(ALGOS.keys()).append("all")

def update_config_hypes(hyperparams,config):
    
    hyperparams["learning_rate"] = config["learning_rate"]
    hyperparams["gail_loss"] = config["gail_loss"]
    try:
        hyperparams["reward_factor"] = config["reward_factor"]
    except: KeyError
    try:
        hyperparams["cnn_version"] = config["cnn_version"]
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
    

def train(args,seeds,expert_traj,device,config_dict):
    project_name = "_".join(["tuning",args.env_id,args.algo, args.tag])
    with wandb.init(project=project_name,config=config_dict) as run:
        update_config_hypes(hyperparams,config_dict)
        for seed in seeds:
            name = args.algo.upper() + "Algorithm"
            # tb_log_name = '_'.join([name , run.id, str(seed), str(config["n_steps"])])
            tb_log_name = '_'.join([name , run.id, str(seed), str(args.cuda_id)])

            using_cuda = True if device == th.device("cuda") else False
            args = set_up_parameters(hyperparams, args)
            set_random_seed(seed, using_cuda)
            envs, eval_env, norm_env = create_env(hyperparams, args, seed, tune=True)
            expert_traj = modify_expert_data(expert_traj, envs, args)
            # return
            ALGOS[args.algo].train(
            envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj,
            device, seed, tune=True,tb_log_name=tb_log_name,run_id=run.id,cuda_id=args.cuda_id
            )


def tune(args, seeds, expert_traj, device):
    
    project_name = "_".join(["tuning",args.env_id,args.algo, args.tag])
    with wandb.init(project=project_name) as run:
        for seed in seeds:
            seed=int(seed)
            print("starging trainin on " + str(seed))
            config = wandb.config
            
            update_config_hypes(hyperparams,config)

            

            name = args.algo.upper() + "Algorithm"
            # tb_log_name = '_'.join([name , run.id, str(seed), str(config["n_steps"])])
            tb_log_name = '_'.join([name , run.id, str(seed), str(args.cuda_id)])

            using_cuda = True if device == th.device("cuda") else False

            args = set_up_parameters(hyperparams, args)
            set_random_seed(seed, using_cuda)
            envs, eval_env, norm_env = create_env(hyperparams, args, seed, tune=True)
            expert_traj = modify_expert_data(expert_traj, envs, args)
            ALGOS[args.algo].train(
                envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj,
                device, seed, tune=True,tb_log_name=tb_log_name,run_id=run.id,cuda_id=args.cuda_id
            )


def main(hyperparams, args, expert_traj, device):

    sweep_config = read_sweepconfig(args.env_id, args.algo, args.sweep_set, verbose=args.verbose)
    for k,v in sweep_config['parameters'].items():
        if  isinstance(v, list):
            sweep_config["parameters"].update({k:dict(values=v)})

    seeds=np.random.randint(3,1e4,1)

    sweep_config.update({"name": "_".join([args.env_id, args.algo, args.tag])})

    print("running tuning")
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config)
    wandb_train_func = functools.partial(tune, args, seeds, expert_traj, device)
    wandb.agent(sweep_id, function=wandb_train_func, count=args.count)


if __name__ == "__main__":
    args = get_args()

    print(f"Training {args.env_id}...")

    hyperparams, saved_hyperparams = read_hyperparameters(
        args.algo, args.env_id, verbose=args.verbose
    )
    if args.num_objs!=0:
        expert_traj = np.load("data/expert_{}_{}.npy".format(args.env_id,str(args.num_objs))).astype(np.float32)
    else:
        expert_traj = np.load("data/expert_{}.npy".format(args.env_id)).astype(np.float32)
    expert_traj = th.from_numpy(expert_traj)

    device = get_device(args.device)
    print(f"Device = {device}")

    main(hyperparams, args, expert_traj, device)
