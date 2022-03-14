import subprocess
import os
import argparse


def main(args):
    env_id=args.env_id
    cu=args.cuda_id
    algo=args.algo
    files=os.listdir(os.path.join("./train_configs/",args.env_id))
    files=[f.strip('.json') for f in files]
    config_ids=[f.replace(args.env_id+"_"+args.algo+"_","") for f in files]
    config_ids=[ c for c in config_ids if args.tag in c]
    print(config_ids)
    for c in config_ids:
        print(c)
        cmd = 'CUDA_VISIBLE_DEVICES={} xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 run_final.py --cuda-id {} --env-id {} --algo {} --config_id {} --timesteps 500 --reg expire_codes --tag {} '.format(cu,cu,env_id,algo,c,args.tag)
        # print(cmd)
        subprocess.call(cmd, shell=True)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id", help="environment ID", type=str, default="MiniGrid-DoorKey-5x5-v0"
    )
    parser.add_argument(
        "--cuda_id", help=" ID", type=str, default=0
    )
    parser.add_argument(
        "--algo", help="algorithm imitation", type=str, default="vqail"
    )
    parser.add_argument(
        "--tag", help="environment ID", type=str, default="MiniGrid-DoorKey-5x5-v0"
    )
    args = parser.parse_args()

    print(args)
    main(args)