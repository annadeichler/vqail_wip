import argparse
import functools
import torch as th
import torch
import torch.nn as nn
import numpy as np
import gym
import gym_miniworld
import wandb
from stable_baselines3.common.utils import set_random_seed

from train_models import train_gail, train_vail, train_vqvail
from util import read_hyperparameters, get_device


ALGOS = {"gail": train_gail, "vail": train_vail, "vqail": train_vqvail}

def train(algo,seeds):
    project_name = "_".join(["tuning",args.env_id,algo])
    with wandb.init(project=project_name) as run:
        for seed in seeds:
            config = wandb.config
            # config_str = '_'.join(['_'.join([k,str(round(v,4))]) for k,v in config._items.items() if  not isinstance(v, dict)])
            config = wandb.config
            hyperparams["learning_rate"] = config["learning_rate"]
            # hyperparams["n_epochs"] = config["n_epochs"]

            # TODO: Can evaluate in train instead of in train_model files.
            hyperparams.update({"policy_kwargs": eval(hyperparams["policy_kwargs"])})
            hyperparams.update({"gail": eval(hyperparams["gail"])})
            hyperparams.update({"vail": eval(hyperparams["vail"])})
            hyperparams.update({"vqvail": eval(hyperparams["vqvail"])})

            hyperparams["policy_kwargs"]["n_steps"] = config["n_steps"]
            hyperparams["policy_kwargs"]["ent_coef"] = config["ent_coef"]
            hyperparams["vail"]["latent_size"] = config["latent_size"]
            hyperparams["vqvail"]["embedding_dim"] = config["embedding_dim"]
            # hyperparams["gail"]["discrim_hidden_size"] = config["discrim_hidden_size"]
            # hyperparams["vqvail"]["hidden_size"] = config["hidden_size"]
            hyperparams["vqvail"]["n_embeddings"] = config["n_embeddings"]

            hyperparams.update({"policy_kwargs": str(hyperparams["policy_kwargs"])})
            hyperparams.update({"gail": str(hyperparams["gail"])})
            hyperparams.update({"vail": str(hyperparams["vail"])})
            hyperparams.update({"vqvail": str(hyperparams["vqvail"])})

            name = algo.upper() + "Algorithm"
            tb_log_name = '_'.join([name ,run.id,str(seed)])
            using_cuda = True if device == th.device("cuda") else False

            set_random_seed(seed, using_cuda)
            ALGOS[algo].train(
                args, hyperparams, saved_hyperparams, expert_traj, device, seed, tune=False,tb_log_name=tb_log_name
            )

def main(hyperparams):
    # sweep_config = {
    #     "name": args.algo,
    #     "method": "random",
    #     "metric": {"name": "ep_rew_mean", "goal": "maximize"},
    #    "parameters": {
    #         "learning_rate": {"min": 0.0001, "max": 0.001},
    #         "n_steps": dict(values=[128,512,1024]),
    #         "ent_coef": {"min": 0.001, "max": 0.01},
    #         "embedding_dim": dict(values=[2, 8, 32]),
    #         "hidden_size": dict(values=[64, 100, 128]),
    #         "latent_size": dict(values=[2, 4, 16,  256]),
    #         "n_embeddings": dict(values=[32, 64, 128, 256, 512]),
    #         "n_epochs": dict(values=[4, 10]),
    #         "discrim_hidden_size": dict(values=[32, 64, 128]),
    #     },
    # }

    sweep_config = {
        "name": args.algo,
        "method": "random",
       "metric": {"name": "ep_rew_mean", "goal": "maximize"},
       "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.001},
            # "n_steps": dict(values=[128,512]),
            "ent_coef": {"min": 0.001, "max": 0.01},
            "embedding_dim": dict(values=[16, 32]), # VQ
            "latent_size": dict(values=[32, 128,  256]),
            "n_embeddings": dict(values=[32, 128, 512])
        },
    }

    sweep_id = wandb.sweep(sweep_config)
    seeds = [int(s) for s in args.seeds]
    wandb_train_func = functools.partial(train,args.algo,seeds)
    print(args.seeds)
    wandb.agent(sweep_id, function=wandb_train_func, count=args.count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id", help="environment ID", type=str, default="CartPole-v1"
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm (gail, vail or vqvail, or 'all')",
        default="vqvail",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--count",
        help="count parameter for wandb",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-log",
        "--log",
        help="Log folder for tensorboard",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--timesteps",
        help="override the default timesteps to run",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--seeds", 
        help="list of seeds to run tuning on", 
        nargs="*", 
        default=[1000, 2000, 3000],
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="verbose 0 or 1 (default: 1)"
    )
    parser.add_argument(
        "--plot-umap",
        type=bool,
        help="Plot umap projection for vqvail, saved in plots/",
        default=False,
    )
    parser.add_argument(
        "--chg-box-color", default=False,
        help="Change box color of miniworld envs",
        action="store_true",
    )
    parser.add_argument("--n-times", type=int, help="Repeat n times", default=1)
    parser.add_argument("--device", help="Device", type=str, default="auto")
    parser.add_argument(
        "--gpu-optimize", default=False,
        help="Optimize GPU memory",
        action="store_true",
    )
    args = parser.parse_args()

    print(f"Training {args.env_id}...")

    global hyperparams
    global saved_hyperparams
    hyperparams, saved_hyperparams = read_hyperparameters(
        args.algo, args.env_id, verbose=args.verbose
    )

    global expert_traj
    expert_traj = np.load("data/expert_{}.npy".format(args.env_id)).astype(np.float32)
    expert_traj = th.from_numpy(expert_traj)
    # print(expert_traj.shape)

    global device
    device = get_device(args.device)
    print(f"Device = {device}")

    main(hyperparams)
