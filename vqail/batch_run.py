import subprocess
import os
import argparse
from collections import OrderedDict
from itertools import islice
import numpy as np 

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
def main(args):
    env_id=args.env_id
    cu=args.cuda_id
    algo=args.algo
    files=os.listdir(os.path.join("./train_configs/",args.env_id))
    files=[f.strip('.json') for f in files]
    config_ids=[f.replace(args.env_id+"_"+args.algo+"_","") for f in files]
    if args.filter_tag!="":
        tag = args.filter_tag
    else:
        tag = args.tag
    config_ids=np.array([ c for c in config_ids if tag in c])
    print(config_ids)
    d={c.split('_')[-1]:c for c in config_ids}
    od = OrderedDict(sorted(d.items()))

    # config = list(split(range(len(config_ids)), 3))[args.id]
    splits=np.array_split(range(len(config_ids)), args.chunks)
    
    # print(splits[2])
    configs = config_ids[np.array(splits[args.id])]
    for c in configs:
        print(c)
        cmd = 'CUDA_VISIBLE_DEVICES={} xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 run_final.py --cuda-id {} --env-id {} --algo {} --config_id {} --timesteps {} --reg expire_codes --tag {} '.format(cu,cu,env_id,algo,c,args.timesteps,args.tag)
        print(cmd)
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
        "--tag", help="environment ID", type=str, default="smsm"
    )
    parser.add_argument(
        "--timesteps", help="ts", type=int, default=750
    )
    parser.add_argument(
        "--filter_tag", help="ts", type=str, default=""
    )
    parser.add_argument(
        "--id", help="ts", type=int, default=0
    )
    parser.add_argument(
        "--chunks", help="ts", type=int, default=1
    )
    args = parser.parse_args()

    print(args)
    main(args)