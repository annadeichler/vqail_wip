import argparse
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import gym
import gym_miniworld


choices = ["GAIL", "VAIL", "VQVAIL"]

parser = argparse.ArgumentParser(description="RL")
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument(
    "--env-id",
    default="MiniWorld-TMaze-v0",
    help="environment to train on (default: MiniWorld-TMaze-v0)",
)
parser.add_argument(
    "--algo",
    help="RL Algorithm (GAIL, VAIL or VQVAIL)",
    default="VQVAIL",
    type=str,
    choices=choices,
)
parser.add_argument(
    "--algo-seed", type=int, default=1000, help="random seed (default: 1000)"
)
parser.add_argument(
    "--domain-rand", type=bool, help="Domain randomization", default=False
)
parser.add_argument(
    "--chg-box-color", type=bool, help="Change color of box", default=False
)

args = parser.parse_args()

seed = args.seed
env_id = args.env_id
algo_seed = args.algo_seed
algo = args.algo
domain_rand = args.domain_rand
chg_box_color = args.chg_box_color

eval_env = make_vec_env(
    env_id,
    1,
    wrapper_class=None,
    env_kwargs={"domain_rand": domain_rand, "chg_box_color": chg_box_color},
)
eval_env.seed(seed)

eval_env = VecFrameStack(eval_env, n_stack=4)

model = PPO.load("outputs/{}/{}-{}-{}".format(env_id, algo, algo_seed, env_id))

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
