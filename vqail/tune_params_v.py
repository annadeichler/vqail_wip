import argparse
import functools
from pprint import pprint
import torch as th
import numpy as np
import gym
import gym_miniworld
import wandb
from stable_baselines3.common.utils import set_random_seed

from train_models import train_gail, train_vail, train_vqvail
from train import set_up_parameters, create_env, modify_expert_data
from util import read_hyperparameters, get_device
from arguments import get_args


ALGOS = {"gail": train_gail, "vail": train_vail, "vqail": train_vqvail}
choices = list(ALGOS.keys()).append("all")


def train(args, seeds, expert_traj, device):
    project_name = "_".join(["tuning",args.env_id,args.algo, args.tag])
    with wandb.init(project=project_name) as run:
        for seed in seeds:
            config = wandb.config
            hyperparams["learning_rate"] = config["learning_rate"]
            hyperparams["gail_loss"] = config["gail_loss"]
            hyperparams["reward_factor"] = config["reward_factor"]
            hyperparams.update({"policy_kwargs": eval(hyperparams["policy_kwargs"])})
            hyperparams["policy_kwargs"]["n_steps"] = config["n_steps"]
            hyperparams["policy_kwargs"]["ent_coef"] = config["ent_coef"]
            hyperparams["policy_kwargs"]["batch_size"] = config["batch_size"]
            hyperparams["policy_kwargs"]["learning_rate"] = config["policy_lr"]
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

            name = args.algo.upper() + "Algorithm"
            # tb_log_name = '_'.join([name , run.id, str(seed), str(config["n_steps"])])
            tb_log_name = '_'.join([name , run.id, str(seed)])

            using_cuda = True if device == th.device("cuda") else False

            pprint(saved_hyperparams)

            args = set_up_parameters(hyperparams, args)
            set_random_seed(seed, using_cuda)
            envs, eval_env, norm_env = create_env(hyperparams, args, seed, tune=True)
            expert_traj = modify_expert_data(expert_traj, envs, args)
            ALGOS[args.algo].train(
                envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj,
                device, seed, tune=True,tb_log_name=tb_log_name,run_id=run.id,cuda_id=args.cuda_id
            )


def main(hyperparams, args, expert_traj, device):
    sweep_config = {
        "name": "_".join([args.env_id, args.algo, args.tag]),
        "method": "random",
        "metric": {"name": "ep_rew_mean", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"min": 0.00001, "max": 0.001},
            "reward_factor": dict(values=[1,-1]),
            "gail_loss": dict(values=["gailfo", "agent"]),
            "ent_coef": {"min": 0.001, "max": 0.01},
            'policy_lr': {"min": 0.00001, "max": 0.001},
            'n_steps': dict(values=[128, 256]),
            'batch_size': dict(values=[64, 128]),
            "embedding_dim": dict(values=[8,16, 32]), # VQ
            "hidden_size": dict(values=[64, 128, 256]), 
            # "latent_size": dict(values=[16, 64, 128, 256]), # vail
            "n_embeddings": dict(values=[16, 64, 128, 256]) # VQ
        },
    }

    if args.algo == 'gail':
        sweep_config["parameters"].update({"discrim_hidden_size": dict(values=[64, 100]),})
    elif args.algo == 'vail':
        sweep_config["parameters"].update({"latent_size": dict(values=[3, 32, 128]),
                                           "hidden_size": dict(values=[64, 100])
                                          })
    elif args.algo == 'vqail':
        sweep_config["parameters"].update({
            "embedding_dim": dict(values=[8, 16, 32]),
            "n_embeddings": dict(values=[3, 8, 32, 64]),
            "hidden_size": dict(values=[64, 100])
        })

    sweep_id = wandb.sweep(sweep_config)
    seeds = [int(s) for s in args.seeds]
    wandb_train_func = functools.partial(train, args, seeds, expert_traj, device)
    print(args.seeds)
    wandb.agent(sweep_id, function=wandb_train_func, count=args.count)


if __name__ == "__main__":
    args = get_args()

    print(f"Training {args.env_id}...")

    hyperparams, saved_hyperparams = read_hyperparameters(
        args.algo, args.env_id, verbose=args.verbose
    )

    expert_traj = np.load("data/expert_{}.npy".format(args.env_id)).astype(np.float32)
    expert_traj = th.from_numpy(expert_traj)
    print(expert_traj.shape)

    device = get_device(args.device)
    print(f"Device = {device}")

    main(hyperparams, args, expert_traj, device)
