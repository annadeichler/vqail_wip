import os
import pathlib
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline


plt.style.use("seaborn-colorblind")

expert_metrics = {
    "MiniGrid-Empty-Random-6x6-v0": (0.97, 0.007),
    "MiniGrid-Empty-8x8-v0": (0.93, 0.008),
    "MiniWorld-OneRoom-v0": (0.93, 0.02),
    "MiniWorld-Hallway-v0": (0.964, 0.16),
    "MiniWorld-TMaze-v0": (0.894, 0.004),
    "MiniWorld-YMaze-v0": (0.877, 0.15),
    "MiniWorld-TMazeLeft-v0": (0.954, 0.005),
    "MiniWorld-TMazeRight-v0": (0.941, 0.008),
    "Pendulum-v0": (
        -169.887,
        104.904,
    ),  # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
    "CartPole-v1": (
        500.0,
        0.0,
    ),  # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
    "Acrobot-v1": (
        -73.506,
        18.201,
    ),  # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
}


def get_log(path, fs):
    logs = []
    print(f"fs: {fs}")
    for f in fs:
        d = np.load(path + f)
        # This is a hack to remove nan's.
        if np.isnan(d[0, 1]):
            d[0, 1] = d[3, 1]
            d[1, 1] = d[3, 1]
            d[2, 1] = d[3, 1]
        d = np.unique(d, axis=0)
        logs.append(pd.DataFrame(d, columns=["Step", "Value"]))

    for i in range(len(logs)):
        logs[i].set_index("Step", inplace=True)

    rew_log = logs[0].copy()
    for i in range(1, len(logs)):
        rew_log["Value " + str(i)] = logs[i]["Value"]

    rew_log["Avg Reward"] = rew_log.loc[:, list(rew_log.columns)].mean(axis=1)
    rew_log["Std Reward"] = rew_log.loc[:, list(rew_log.columns)].std(axis=1)

    # 95% confidence interval
    rew_log["Size"] = list(range(1, rew_log.shape[0] + 1))
    rew_log["Conf Interval"] = 1.96 * rew_log["Std Reward"] / np.sqrt(len(logs))

    print(rew_log.head(4))

    return rew_log


def plot_log(logs, labels, fname, env, expert_rew, expert_std, save=True):
    plt.clf()
    for i in range(len(logs)):
        plt.plot(logs[i].index, logs[i]["Avg Reward"], label=labels[i])
        plt.fill_between(
            logs[i].index,
            logs[i]["Avg Reward"] - logs[i]["Conf Interval"],
            logs[i]["Avg Reward"] + logs[i]["Conf Interval"],
            alpha=0.5,
        )
    if expert_rew is not None:
        plt.plot(logs[i].index, [expert_rew] * logs[i].shape[0], linestyle="dashed")

    plt.legend(loc="upper left")
    plt.ylabel("Return")
    plt.title(env[:-3])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if save:
        plt.savefig("plots/{}.png".format(fname))
    else:
        plt.show()


def plot_reward(env, save=True):
    expert_rew, expert_std = expert_metrics[env]
    path = "outputs/{}/".format(env)
    labels = ["GAIL", "VAIL", "VQAIL (Ours)"]

    savepath = "mean_reward_{}".format(env)

    files = os.listdir(path)
    files = [f for f in files if pathlib.Path(f).suffix == ".npy"]
    print(files)

    gail_files = [f for f in files if "_GAIL" in f]
    vail_files = [f for f in files if "_VAIL" in f]
    vqvail_files = [f for f in files if "_VQVAIL" in f]

    rew_gail = get_log(path, gail_files)
    rew_vail = get_log(path, vail_files)
    rew_vqvail = get_log(path, vqvail_files)

    plot_log(
        [rew_gail, rew_vail, rew_vqvail],
        labels,
        savepath,
        env,
        expert_rew,
        expert_std,
        save,
    )


def plot_vail(env, save=True):
    path1 = "outputs/{}/".format(env)
    path2 = "outputs/vail_exp/{}/32_latents/".format(env)
    path3 = "outputs/vail_exp/{}/256_latents/".format(env)

    labels = ["3 Latents", "32 Latents", "256 Latents", "3 Latents VQAIL"]

    savepath = "mean_reward_vail_{}".format(env)

    files = os.listdir(path1)
    files = [f for f in files if pathlib.Path(f).suffix == ".npy"]
    vail_files = [f for f in files if "_VAIL" in f]

    files32 = os.listdir(path2)
    files = [f for f in files32 if pathlib.Path(f).suffix == ".npy"]
    vail_files2 = [f for f in files if "_VAIL" in f]

    files256 = os.listdir(path2)
    files = [f for f in files256 if pathlib.Path(f).suffix == ".npy"]
    vail_files3 = [f for f in files if "_VAIL" in f]

    files3vq = os.listdir(path1)
    files = [f for f in files3vq if pathlib.Path(f).suffix == ".npy"]
    vail_files4 = [f for f in files if "_VQVAIL" in f]

    rew_vail1 = get_log(path1, vail_files)
    rew_vail2 = get_log(path2, vail_files2)
    rew_vail3 = get_log(path3, vail_files3)
    rew_vail4 = get_log(path1, vail_files4)

    plot_log(
        [rew_vail1, rew_vail2, rew_vail3, rew_vail4],
        labels,
        savepath,
        env,
        None,
        None,
        save,
    )


def plot_embeddings(vqvail, env_id, log_dir, seed):
    # modified from https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    import umap

    proj = umap.UMAP(n_neighbors=3, min_dist=0.1, metric="cosine").fit_transform(
        vqvail.vqvae.codebook.ema_weight.data.cpu()
    )
    figure = plt.figure()
    ax = figure.gca()
    ax.scatter(proj[:, 0], proj[:, 1], alpha=0.3)

    log_path = os.path.join(log_dir, "umap")
    Path(log_path).mkdir(exist_ok=True)
    print(f"Saving embeddings at {log_path}")
    path = os.path.join(log_path, "Umap-VQAIL-{}-{}.png".format(env_id, seed))
    plt.savefig(path)

    return figure


def plot_smooth_curve(dfs):
    # Dataset

    for i, d in enumerate(dfs):
        x = d[:, 0]
        y = d[:, 1]

        X_Y_Spline = make_interp_spline(x, y)

        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)

        # Plotting the Graph
        plt.plot(X_, Y_, label=str(i))
        plt.xlabel("X")
        plt.ylabel("Y")
    plt.legend()
    plt.show()


def get_mean(env):
    path = "outputs/{}/".format(env)
    n = list(range(1000, 6000, 1000))
    labels = ["GAIL", "VAIL", "VQVAIL"]

    # https://stats.stackexchange.com/questions/209585/the-average-of-mean-and-standard-deviation
    for i in labels:
        rews, stds = [], []
        for seed in n:
            p = os.path.join(path, i)
            f = open("{}-{}-{}.txt".format(p, str(seed), env)).read()
            flist = f.split(",")
            rew = float(flist[1].split(":")[1].rstrip().lstrip())
            std = float(flist[2].split(":")[1].rstrip().lstrip())
            rews.append(rew)
            stds.append(std)
        print(rews)
        print(stds)
        print(i, np.average(rews), np.sqrt(0.2 * np.sum(np.square(stds))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name", help="environment name", type=str, default="cartpole"
    )
    parser.add_argument(
        "--func",
        help="Plot Function (reward, vail or mean)",
        default="reward",
        type=str,
        required=True,
        choices=["reward", "vail", "mean"],
    )
    args = parser.parse_args()
    if args.func == "vail":
        plot_vail(args.env_name)
    elif args.func == "mean":
        get_mean(args.env_name)
    else:
        plot_reward(args.env_name)
