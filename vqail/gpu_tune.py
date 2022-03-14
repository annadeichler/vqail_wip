import multiprocessing
import subprocess
import tune_params_seed
import os
import argparse

#list of gpu ids to use (image base should be different, state based can be same)
# gpu_id_list = [3,4,5]
# gpu_id_list =[3,4,5]


# def function(x):
#     # assign each process to own GPU
#     cpu_name = multiprocessing.current_process().name
#     cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
#     gpu_id = gpu_id_list[cpu_id]
#     print(gpu_id)
#     os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
#     cmd = 'mpiexec -n 1 python3 tune_params_seed.py --env-id MiniWorld-PickupObjs-v0 --algo {:s} --n-times 1 --seeds 100 200 300 --timesteps 1500 --device cuda'.format(x)

#     subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id", help="environment ID", type=str, default="CartPole-v1"
    )
    args = parser.parse_args()
    parser.add_argument(
        "--algo", help="vail/gail/vqail", type=str, default="CartPole-v1"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
