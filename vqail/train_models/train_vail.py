from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import (
    is_image_space,
)

from models import VDB, VAIL, VDBImage
from wrappers import VecCustomReward
from util import (
    get_obs_action_dim,
    save_eval_rewards,
    evaluate_model,
)


def train(envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj, device, seed, tune=False,tb_log_name="VAILAlgorithm",run_id="",cuda_id=""):
    # 2. Instantiate model
    print(f"Device in train_vqvail.py = {device}")

    vail_params = eval(hyperparams["vail"])
    hidden_size = vail_params["hidden_size"]
    latent_size = vail_params["latent_size"]

    if is_image_space(envs.observation_space):
        # Uses LfO (Learning from observations).
        model = VDBImage(envs.observation_space, hidden_size, latent_size, device)
    else:
        # Flattened observations and action dimensions
        # Uses LfD (learning from demonstrations).
        num_inputs, num_outputs = get_obs_action_dim(envs)
        print("num_inputs={}, num_outputs={}".format(num_inputs, num_outputs))
        model = VDB(num_inputs + num_outputs, hidden_size, latent_size, device)

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

    algo = VAIL(
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

    algo, ep_mean_rewards, log_path = algo.learn(total_timesteps=args.timesteps,eval_env=eval_env,tb_log_name=tb_log_name, save_model_interval=args.save_model_interval,run_id=run_id,cuda_id=cuda_id)

    # save model and evaluation results
    if not tune:
        evaluate_model(algo, eval_env, args.env_id, args.norm_obs, hyperparams, log_path, seed)

    # cleanup subproc code
    envs.close()

    return algo
