CUDA_VISIBLE_DEVICES=$1  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset"  python3.8 run_final.py  --cuda-id $1 --env-id $2 --algo $3  --config_id  $4 --timesteps $5  --reg $6   --tag $7 
#  --tag $7 --cnn_version $8 --num_objs 2
# 