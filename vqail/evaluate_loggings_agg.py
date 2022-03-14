import glob
import os
import re
from numpy.lib.npyio import save
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
import functools
import argparse


pd.set_option('display.max_columns', None)


def get_loggings_df(fp):
    def parse_tfevent(tfevent):
        if len(tfevent.summary.value[0].histo.bucket)!=0:
            return dict(
                wall_time=tfevent.wall_time,
                step=tfevent.step,
                name = tfevent.summary.value[0].tag+'_hist',
                value={"bucket_limit":tfevent.summary.value[0].histo.bucket_limit,
                        "bucket": tfevent.summary.value[0].histo.bucket}
            )
        else:
            return dict(
                wall_time=tfevent.wall_time,
                name=tfevent.summary.value[0].tag,
                step=tfevent.step,
                value=float(tfevent.summary.value[0].simple_value),
            )
    df = pd.DataFrame([parse_tfevent(e) for e in summary_iterator(fp) if len(e.summary.value)])
    logged_vals = df['name'].unique()
    dfs=[]
    for v in logged_vals:
        dfs.append(df.loc[df['name'] == v].rename(columns={'value': v}).drop(columns=['wall_time','name']))
    merge = functools.partial(pd.merge, on=['step'])
    result = functools.reduce(merge, dfs)
    hkeys = [v for v in logged_vals if 'hist' in v]
    for v in hkeys:
        result=pd.concat([result.drop([v], axis=1),pd.json_normalize(result[v]).add_prefix(v+'_')],axis=1)

    return result,logged_vals
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id", help="environment ID", type=str, default="Hopper-v2"
    )
    parser.add_argument(
        "--logdir", help="Input path", type=str, default='./results/'
    )
    parser.add_argument(
        "--savedir", help="Save path", type=str, default='./outputs/'
    )
    parser.add_argument(
        "--tuning", help="", type=str, default=True
    )

    args = parser.parse_args()
    logging_dir = os.path.join(args.logdir,args.env_id)
    event_paths = glob.glob(os.path.join(logging_dir, "*","event*"))
    print(event_paths)

    runs = set()
    nsteps = set()
    algos = set()
    for ep in event_paths:
        filename = re.sub(logging_dir+"/", "", os.path.split(ep)[0])
        algo_name, run_id, seed, n_steps, _ = filename.split("_")
        runs.add(run_id)
        nsteps.add(str(n_steps))
        algos.add(algo_name)
    print("Runs: ", runs)
    print("n_steps: ", n_steps)
    print("algos: ", algos)

    for a in algos:
        for nstep in nsteps:
            print(a, nstep)
            df_all = []
            for run in runs:
                dfs = []
                for ep in event_paths:
                    filename = re.sub(logging_dir+"/", "", os.path.split(ep)[0])
                    algo_name, run_id, seed, n_steps, _ = filename.split("_")
                    if run_id == run and n_steps == nstep and algo_name == a:
                        df_logs,logged_vals=get_loggings_df(ep)
                        if not dfs:
                            dfs.append(df_logs["step"])
                        dfs.append(df_logs["rollout/ep_rew_mean"])
                if not dfs:
                    continue
                df = pd.concat(dfs, axis=1)
                df[run+"_mean_ep_rew_mean"] = df.iloc[:, 1:].mean(axis=1)

                if not df_all:
                    df_all.append(df["step"])
                df_all.append(df[run+"_mean_ep_rew_mean"])

            df_all = pd.concat(df_all, axis=1)
            save_dir = os.path.join(args.savedir,args.env_id)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, a+"_"+nstep+".csv")
            print(save_path, df_all.shape)
            df_all.to_csv(save_path)

    # for ep in event_paths:
    #     df_logs,logged_vals=get_loggings_df(ep)
    #     hist_keys = [k for k in logged_vals if 'logit'  in k]
    #     print(df_logs)
    #     # print(df_logs['vail/train/logits_gen'])
