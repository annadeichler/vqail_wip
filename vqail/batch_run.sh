tag="lglg"
env=$1
algo=$2
cuda_id=$3
ts=$4
flist=$( ls ./train_configs/$1 | grep -- $tag )
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

    CUDA_VISIBLE_DEVICES=$cuda_id  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset"  python3.8 run_final.py  --cuda-id $3 --env-id $1 --algo $2  --config_id  $config_id --timesteps $ts  --reg expire_codes   --tag $tag 
    for i in 1000; do
        echo "start"
    done
done




