import os,sys
from re import L
import argparse
import json,io,yaml
import ruamel.yaml  
import yaml
import collections.abc
from collections import OrderedDict
import ast 
import numpy as np 
import random

# TUNED_HYPES_DIR='../final_configs/'
TUNED_HYPES_DIR='../train_configs/'
WANDB_BASE_DIR = '../wandb/'
HYPES_DIR = '../hyperparams/'
wandb_keys=['wandb_version','_wandb']
HYPES_ALGO = {'vail':'vail','gail':'gail','vqail':'vqvail'}

_true_set = {'yes', 'true', 't', 'y', '1'}
_false_set = {'no', 'false', 'f', 'n', '0'}


def str2bool(value, raise_exc=False):
    if isinstance(value, str) or sys.version_info[0] < 3 and isinstance(value, basestring):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None

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
    d['run_id'] = r
    d['ep_rew_mean']=json.load(open(os.path.join(p,'files','wandb-summary.json')))['ep_rew_mean']
    jargs=json.load(open(os.path.join(p,'files','wandb-metadata.json')))['args']
    algo = jargs[jargs.index('--algo')+1]
    env  = jargs[jargs.index('--env-id')+1]
    tag=""
    try: 
        tag  = jargs[jargs.index('--tag')+1]
    except: KeyError
    path = os.path.join(p,'files',"config.yaml")
    with open(path) as f:
        data_loaded = yaml.load(f, Loader=yaml.FullLoader)
    d['parameters']={k:v['value'] for k,v in data_loaded.items() if k not in wandb_keys}
    return d,algo,env,tag


def get_results(args):
    dir = os.path.join(WANDB_BASE_DIR, "sweep-"+args.sweep_id)
    print(dir)
    fnames=os.listdir(dir)
    wb_files=os.listdir(WANDB_BASE_DIR)
    runs= [f.strip('.yaml').split('-')[1] for f in fnames]
    print(runs)
    d=OrderedDict()
    for r in runs:
        try: 
            data,algo,env,tag=  get_run_data(r,wb_files)
            [d.update({r:data})]
        except KeyError:print(r)
    d = dict(sorted(d.items(), key=lambda x: x[1]['ep_rew_mean'],reverse=True))
    return d,env,algo,tag
    # tuned_hypes = d[first(d)]['parameters']
    # update_hypes(env,algo,d[first(d)]["tuned_hypes"])

def sample_results(args,ids_vail,ids_vqail):
        ids_vail = random.sample(ids_vail,5)
        ids_vqail = random.sample(ids_vqail,5)
        print(ids_vail)
        print(ids_vqail)
        wb_files=os.listdir(WANDB_BASE_DIR)

        d_vl=OrderedDict()
        for r in ids_vail:
            try: 
                data,algo,env=  get_run_data(r,wb_files)
                [d_vl.update({r:data})]
            except KeyError:print(r)
        d_vq=OrderedDict()
        for r in ids_vqail:
            try: 
                data,algo,env=  get_run_data(r,wb_files)
                [d_vq.update({r:data})]
                print(data['parameters']['ent_coef'])
            except KeyError:print(r)

        r_vl=([v['ep_rew_mean'] for k,v in d_vl.items()])
        r_vq= ([v['ep_rew_mean'] for k,v in d_vq.items()])
        print(r_vl)
        print(r_vq)
        print(np.mean(r_vl))
        print(np.mean(r_vq))

def save_config(d_run,algo,env,tag):
    print(d_run[0])
    out_name = "{}_{}_{}_{}_{}.json".format(env,algo,tag,args.sweep_id,d_run[0])
    print(out_name)
    fpath=os.path.join(TUNED_HYPES_DIR,env,out_name)
    print(fpath)
    with open(fpath, 'w') as fp:
            json.dump(d_run[1], fp, sort_keys=True, indent=4)



def main(args):

    if args.update==True:
    
        print("updating")
        d,env,algo,tag= get_results(args)
        for i in range(len(d)):
            d_run=list(d.items())[i]
            if d_run[1]['ep_rew_mean']<args.rew_thr or  np.isnan(d_run[1]['ep_rew_mean']):
                continue
            print(d_run[1]['ep_rew_mean'])
            save_config(d_run,algo,env,tag)


        # out_name = "{}_{}_{}_{}.json".format(env,algo,args.sweep_id,)
        # print(out_name)
        # fpath=os.path.join(TUNED_HYPES_DIR,out_name)
        # print(fpath)
        # print(args.update)
        # with open(fpath, 'w') as fp:
        #     json.dump(d[first(d)], fp, sort_keys=True, indent=4)

    else:

        ids_vail = ["39lw7ip8",
        "oepyxhug",
        "yaxnfjyg",
        "sbdqk0yv",
        "3agtizei",
        "bfjs3zyn",
        "3iijz9x7",
        "2eomti7d",
        "1s4df37l",
        "2gs9wqm"]    
        ids_vqail=[
        "1elgc6p5",
        "a72exycu",
        "3nred44d",
        "1r2914tc",
        "1cnbj6wm",
        "1gaqa882",
        "jnccxx85",
        "3l9x67vy",
        "2xhowtqm",
        "df87mbw4",
        "33sccpkm",
        "34w9xso0",
        "2lj4gpvh",
        "3p95x2da"]
        sample_results(args,ids_vail,ids_vqail)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sweep_id",
        help="wandb sweep id of tuning",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--seed",
        help="wandb sweep id of tuning",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--update",
        help="update config file",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--rew_thr",
        help="reward threshold",
        type=float
    )
    args = parser.parse_args()
    main(args)
