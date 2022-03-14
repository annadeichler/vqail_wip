from typing import Dict
import torch.nn as nn
import torch.cuda


from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import (
    is_image_space,
)

from models import VQVAIL, VQEmbeddingEMA, VQVAE, VQVAEImage
from plots import plot_embeddings
from wrappers import VecCustomReward
from util import (
    get_obs_action_dim,
    save_eval_rewards,
    evaluate_model,
)


def train(envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj, device, seed, tune=False,tb_log_name="VQAILAlgorithm",run_id="",cuda_id = ""):
    # 2. Instantiate model
    print(f"Device in train_vqvail.py = {device}")

    vqvail_params = eval(hyperparams["vqvail"])
    n_embeddings = vqvail_params["n_embeddings"]
    hidden_size = vqvail_params["hidden_size"]
    embedding_dim = vqvail_params["embedding_dim"]
    embedding_dim = args.n_envs * embedding_dim
    regularization = args.reg
    cnn_version = hyperparams['cnn_version'] # correction here, was args, possibly false runs
    codebook = VQEmbeddingEMA(
        n_embeddings=n_embeddings, embedding_dim=embedding_dim,
        regularization=regularization, device=device
    )
    
    if is_image_space(envs.observation_space):
        # Uses LfO (Learning from observations).
        model = VQVAEImage(
            cnn_version,envs.observation_space, hidden_size, embedding_dim, codebook, device
        )
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        print("created VQ")
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
    print("created env")
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
        tune=tune,
        ent_decay=args.ent_decay,
        env_name=args.env_id,
        gpu_optimize=args.gpu_optimize,
        gail_loss=args.gail_loss,
        policy_kwargs=policy_kwargs,
    )

    algo, ep_mean_rewards, log_path = algo.learn(total_timesteps=args.timesteps, eval_env=eval_env,tb_log_name=tb_log_name, save_model_interval=args.save_model_interval,run_id=run_id,cuda_id=cuda_id)

    # save model and evaluation results
    if not tune:
        evaluate_model(algo, eval_env, args.env_id, args.norm_obs, hyperparams, log_path, seed)
        # fake_z_quantized, real_z_quantized = algo.get_latents(args.timesteps*args.n_envs)
        # print(fake_z_quantized.shape, real_z_quantized.shape)

        if args.plot_umap:
            try:
                plot_embeddings(algo, args.env_id, log_path, seed)
            except Exception as e:
                print(e)
                print("Check if you have the correct library installed, pip install umap-learn")

    # cleanup subproc code
    envs.close()

    return algo
