import glob
import os
from itertools import groupby
from typing import DefaultDict
from collections import defaultdict
import pandas as pd
from heapq import heapify, heappush, heappop
from yaml import events
from tensorflow.python.summary.summary_iterator import summary_iterator
import functools
import argparse
import yaml
import numpy as np 
import json

def get_loggings_df(fp):
    def parse_tfevent(tfevent):
        if len(tfevent.summary.value[0].histo.bucket)!=0:
            tag = "/".join(tfevent.summary.value[0].tag.split('/')[1:])
            return dict(
                wall_time=tfevent.wall_time,
                step=tfevent.step,
                name = tag +'_hist',
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
    # print(lenlogged_vals)
    if(len(logged_vals)<2): return None
    dfs=[]
    for v in logged_vals:
        dfs.append(df.loc[df['name'] == v].rename(columns={'value': v}).drop(columns=['wall_time','name']))
    # print(dfs)
    merge = functools.partial(pd.merge, on=['step'])
    result = functools.reduce(merge, dfs)
    # print(result.columns)
    hkeys = [v for v in logged_vals if 'hist' in v]
    for v in hkeys:
        result=pd.concat([result.drop([v], axis=1),pd.json_normalize(result[v]).add_prefix(v+'_')],axis=1)
    return result

def get_config(wandb_paths,id):
    cpath = [p for p in wandb_paths if id in p][0]
    with open(cpath) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_config(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def filter_id(a): return a.split('/')[3].split('_')[1]
def filter_algo(a): return a.split('/')[3].split('_')[0]
def key_func(x,d,a): return d[x][a]['value']

def get_k_best(events_groups,d_conf,k=5):
    topk=[]
    heapify(topk)
    best = {}
    s = ["seed3", "seed2","seed1"]    
    for id,paths in events_groups.items():
        config= d_conf[id]
        # print(paths)
        res=[get_loggings_df(p) for p in paths]
        # if None in res:
            # print("error")
            # continue
        res = pd.concat(res, axis=1, keys=s[:len(paths)])
        res=res.swaplevel(0, 1, axis=1).sort_index(axis=1)
        # try:
        rewards=res['rollout/ep_rew_mean']
         
        lasn  = rewards.tail(int(0.15*len(rewards))).mean(axis=1).mean(axis=0)
        if len(topk) < k:  
            heappush(topk, lasn)
            best[lasn]={'id':id,'results':res}
        else:
            if lasn>topk[0]:
                last=(heappop(topk))
                del best[last]
                heappush(topk, lasn)
                best[lasn]={'id':id,'results':res}
    return best

def save_results(env,algo,attr,best_results,wandb_paths):
    keys = defaultdict(dict)
    for k,v in best_results.items():
        keys[v['id']]={}
        cpath = [p for p in wandb_paths if v['id'] in p][0]
        with open(cpath) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        keys[v['id']]['config'] = config
        keys[v['id']]['mean_reward'] = k
        v['results'].to_csv("{}_{}_{}_{}.csv".format(env,algo,attr,v['id']))
    print(keys.keys())
    
    with open("{}_{}_{}.json".format(env,algo,attr), 'w') as f:
        json.dump(keys, f)

def save_results(outdir,results,d_conf,env,algo,attr):
    keys=defaultdict(dict)
    for k,v in results.items():
        id = v['id']
        keys[id]={}
        keys[id]['config'] = d_conf[id]
        keys[id]['mean_reward'] = k 
        v['results'].to_csv(os.path.join(outdir,"{}_{}_{}_{}.csv".format(env,algo,attr,id)))
    with open(os.path.join(outdir,"{}_{}_{}.json".format(env,algo,attr)), 'w') as f:
        json.dump(keys, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id", help="environment ID", type=str, default="MiniWorld-PickupObjs-v0"
    )
    parser.add_argument(
        "--logdir", help="tensorboard logdir", type=str, default='./results/'
    )
    parser.add_argument(
        "--k", help="get best k mean rewards", type=str, default=5
    )
    parser.add_argument(
        "--tuning", help="", type=str, default=True
    )
    parser.add_argument(
        "--out_dir", default="./results_tuning/"
    )
    args = parser.parse_args()
    outdir = os.path.join(args.out_dir,args.env_id)
    logging_dir = os.path.join(args.logdir,args.env_id)
    if not os.path.exists(outdir):os.makedirs(outdir)
    wandb_dir = './wandb/'
    wb_paths = glob.glob(os.path.join(wandb_dir, "*/","*","config*"))
    d_id_conf = {p.split('/')[2].split('-')[-1]: get_config(p) for p in wb_paths}
    event_paths = glob.glob(os.path.join(logging_dir, "*","event*"))
    event_paths.sort()
    print(len(event_paths))
    algos_grouped = {j:list(i) for j, i in groupby(event_paths,lambda a: filter_algo(a))}
    algos=list(algos_grouped.keys())

    for algo in algos:
        print(algo)
        d = defaultdict(dict)
        attr = 'n_steps'
        events_group = {j:list(i) for j, i in groupby(algos_grouped[algo], lambda a: filter_id(a))}
        [d.setdefault(key_func(k,d_id_conf,attr), {}).update({k:l}) for k,l in events_group.items()]
        attr_vals=list(d.keys())
        print(d.keys())
        for k in attr_vals:
            print(k)
            # print(d.keys())
            best_results=get_k_best(d[k],d_id_conf,3)
            save_results(args.out_dir,best_results,d_id_conf,args.env_id,algo,k)
   
    # things for histo data...
    # for event_path in event_paths:
    #         id = event_path.split('/')[3].split('_')[1]
    #         result,logged_vals=get_loggings_df(event_paths[0]) 
    #         path = [p)for p in wb_paths if id in p][0]
    #         print(event_path)
    #         with open(path) as f:
    #             config = yaml.load(f, Loader=yaml.FullLoader)
            # bucket_limits=result['train/logits_expert_hist_bucket_limit']
            # buckets=result['train/logits_expert_hist_bucket']
  

 