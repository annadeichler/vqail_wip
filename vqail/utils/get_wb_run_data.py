import os,sys
from re import L
import argparse
import json,io,yaml
import ruamel.yaml  
import yaml
import collections.abc
from collections import OrderedDict
import ast 


TUNED_HYPES_DIR='../final_configs/'
WANDB_BASE_DIR = '../wandb/'
HYPES_DIR = '../hyperparams/'
wandb_keys=['wandb_version','_wandb']
HYPES_ALGO = {'vail':'vail','gail':'gail','vqail':'vqvail'}

def first(s):
    return next(iter(s))

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def write_yaml(data,algo):
    ryaml = ruamel.yaml.YAML()
    ryaml.indent(mapping=4, sequence=6, offset=10)
    # ryaml.version = (1, 2)
    with open(os.path.join(HYPES_DIR,"hyperparams_" +str(algo)+".yml"), 'w') as fp:
        ryaml.dump(data, fp)


def update_hypes(env,algo,data):
    d_policy= {}
    fpath = os.path.join(HYPES_DIR,'hyperparams_' + algo + '.yml')
    with open(fpath) as f:
        hypes_all = yaml.load(f, Loader=yaml.FullLoader)
    hypes = hypes_all[env]
    policy_lr = data.pop('policy_lr',None)
    ent_coef = data.pop('ent_coef',None)
    # d_policy = ast.literal_eval(hypes['policy_kwargs'])
    if policy_lr!=None:
        d_policy['learning_rate'] = policy_lr
    if ent_coef!=None:
        d_policy['ent_coef'] = ent_coef
    hypes['policy_kwargs'] = str(dict(d_policy))
    d_algo =ast.literal_eval(hypes[HYPES_ALGO[algo]])
    for k in d_algo:
        if k in data:
            d_algo[k] = data.pop(k)
    hypes[HYPES_ALGO[algo]]=str(d_algo)
    update(hypes,data)
    hypes_all[env] = hypes
    hypes_flatten={}
    print(data)
    # write_yaml(hypes_all,algo)
    

def get_run_data(r,wb_files):
    d={}
    p = [os.path.join(WANDB_BASE_DIR,w) for w in wb_files if r in w][0]
    d['ep_rew_mean']=json.load(open(os.path.join(p,'files','wandb-summary.json')))['ep_rew_mean']
    jargs=json.load(open(os.path.join(p,'files','wandb-metadata.json')))['args']
    algo = jargs[jargs.index('--algo')+1]
    env  = jargs[jargs.index('--env-id')+1]
    path = os.path.join(p,'files',"config.yaml")
    with open(path) as f:
        data_loaded = yaml.load(f, Loader=yaml.FullLoader)
    d['parameters']={k:v['value'] for k,v in data_loaded.items() if k not in wandb_keys}
    return d,algo,env


def get_results(args):
    dir = os.path.join(WANDB_BASE_DIR, "sweep-"+args.sweep_id)
    print(dir)
    fnames=os.listdir(dir)
    wb_files=os.listdir(WANDB_BASE_DIR)
    runs= [f.strip('.yaml').split('-')[1] for f in fnames]
    d=OrderedDict()
    for r in runs:
        try: 
            data,algo,env=  get_run_data(r,wb_files)
            [d.update({r:data})]
        except KeyError:print(r)
    d = dict(sorted(d.items(), key=lambda x: x[1]['ep_rew_mean'],reverse=True))
    return d,env,algo
    # tuned_hypes = d[first(d)]['parameters']
    # update_hypes(env,algo,d[first(d)]["tuned_hypes"])

def main(args):
    d,env,algo= get_results(args)
    out_name = "{}_{}_{}_{}.json".format(env,algo,args.sweep_id,args.seed)
    print(out_name)
    fpath=os.path.join(TUNED_HYPES_DIR,out_name)
    print(fpath)
    with open(fpath, 'w') as fp:
        json.dump(d[first(d)], fp, sort_keys=True, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_id",
        help="wandb sweep id of tuning",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config_id",
        help="wandb sweep id of tuning",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    wb_files=os.listdir(WANDB_BASE_DIR)
    data,algo,env=  get_run_data(args.run_id,wb_files)
    out_name = "{}_{}_{}_{}.json".format(env,algo,args.run_id,args.config_id)
    fpath=os.path.join(TUNED_HYPES_DIR,out_name)
    with open(fpath, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
    print(data)
