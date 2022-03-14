import argparse

ALGOS = ["vqail", "vail", "gail", "all"]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id", help="environment ID", type=str, default="CartPole-v1"
    )
    parser.add_argument(
        "--tune", help="tune or train", type=str2bool, default=True,
    )
    parser.add_argument(
        "--sweep_set", help="which json", type=str, default="",
    )
    parser.add_argument(
        "--cnn_version", help="which cnn to use ", type=str, default="",
    )

    parser.add_argument(
        "--algo",
        help="RL Algorithm (gail, vail or vqvail, or 'all')",
        default="vqvail",
        type=str,
        required=True,
        choices=ALGOS,
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
        "--seed", type=int, default=None, help="random seed (default: None)"
    )
    parser.add_argument(
        "--config_id", type=str, default=None, help="config to load"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="verbose 0 or 1 (default: 1)"
    )
    parser.add_argument(
        "--plot-umap",
        default=False,
        action="store_true",
        help="Plot umap projection for vqvail, saved in plots/",
    )
    parser.add_argument("--n-times", type=int, help="Repeat n times", default=1)
    parser.add_argument("--device", help="Device", type=str, default="auto")
    parser.add_argument(
        "--chg-box-color", default=False,
        help="Change box color of miniworld envs",
        action="store_true",
    )
    parser.add_argument(
        "--render-dim", default=80,
        help="Render dim for fetch pick env", type=int
    )
    parser.add_argument(
        "--gpu-optimize", default=False,
        help="Optimize GPU by clearing cache and deleting cuda objects",
        action="store_true",
    )
    parser.add_argument(
        "--save-model-interval", type=int, default=100, help="Save model interval (default: 100)"
    )
    parser.add_argument(
        "--cuda-id", type=int, default=1, help="cuda id on gpu machines"
    )
    parser.add_argument(
        "--top-view", default=False,
        help="Render top view observations for miniworld env",
        action="store_true",
    )
    parser.add_argument(
        "--chg-tex-train-test", default=False,
        help="Change texture of walls/ceilings at train and test time for miniworld env",
        action="store_true",
    )
    parser.add_argument(
        "--chg-tex-test", default=False,
        help="Change texture of walls/ceilings at test time for miniworld env",
        action="store_true",
    )
    parser.add_argument(
    "--reg",
    help="Regularization type for VQ. Valid types are: ortho_loss (default), mask_codes and expire_codes",
    type=str,
    default="ortho_loss",
    )

    # for tuning
    parser.add_argument(
    "--seeds", 
    help="list of seeds to run tuning on", 
    nargs="*", 
    default=[1000],
    )
    parser.add_argument(
    "--tag",
    help="tag for the sweep during tuning",
    type=str,
    default="",
    )

    parser.add_argument(
    "--num_objs",
    help="pick up objects ",
    type=int,
    default=0,
    )
    parser.add_argument(
        "--count", default=50,
        help="Count of runs in sweep", type=int
    )

    args = parser.parse_args()

    return args


    # ./train_ail.sh 0 lglg_code MiniGrid-DoorKey-5x5-v0 vqail  600 expire_codes minigrid_1 