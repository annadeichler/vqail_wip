from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import (
    is_image_space,
)

from models import Discriminator, DiscriminatorImage, GAIL
from wrappers import VecCustomReward
from util import (
    get_obs_action_dim,
    save_eval_rewards,
    evaluate_model,
)


def train(envs, eval_env, norm_env, args, hyperparams, saved_hyperparams, expert_traj, device, seed, tune=False,tb_log_name="GAILAlgorithm",run_id="",cuda_id=""):
    # 2. Instantiate model
    print(f"Device in train_vqvail.py = {device}")

    discrim_hidden_size = eval(hyperparams["gail"])["discrim_hidden_size"]

    if is_image_space(envs.observation_space):
        # Uses LfO (Learning from observations).
        model = DiscriminatorImage(envs.observation_space, discrim_hidden_size, device)
    else:
        # Flattened observations and action dimensions
        # Uses LfD (learning from demonstrations).
        num_inputs, num_outputs = get_obs_action_dim(envs)
        print("num_inputs={}, num_outputs={}".format(num_inputs, num_outputs))
        model = Discriminator(num_inputs + num_outputs, discrim_hidden_size, device)

    envs = VecCustomReward(
        model,
        envs,
        train=True,
        obs_only=args.obs_only,
        reward_factor=args.reward_factor,
        device=device,
    )

    print("Environment observation shape: ", envs.observation_space.shape)

    policy_kwargs = eval(hyperparams["policy_kwargs"])
    policy_kwargs.update({"env": envs})
    print("policy_kwargs: ", policy_kwargs)

    algo = GAIL(
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

    algo, ep_mean_rewards, log_path = algo.learn(total_timesteps=args.timesteps, eval_env=eval_env,tb_log_name=tb_log_name, save_model_interval=args.save_model_interval,run_id=run_id,cuda_id="")

    # save model and evaluation results
    if not tune:
        evaluate_model(algo, eval_env, args.env_id, args.norm_obs, hyperparams, log_path, seed)

    # cleanup subproc code
    envs.close()

    return algo
