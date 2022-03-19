tag="smsm"
cuda_id=$1
env=$2
algo=$4
ts=$5
tag=$6
flist=$( ls ./train_configs/$2 | grep -- $3)

for i in $flist; do
    var=$i
    prefix="Mini"
    var=${var/#$env}
    var=${var/#_}
    var=${var/#$algo}
    var=${var/#_}
    suffix=".json"
    config_id=${var/%$suffix}
    echo $config_id

    CUDA_VISIBLE_DEVICES=$cuda_id  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset"  python3.8 main.py  --cuda-id $cuda_id --env-id $env --algo $algo  --config_id  $config_id --timesteps $ts  --reg expire_codes   --tag $tag 
    
done




