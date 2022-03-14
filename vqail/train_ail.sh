CUDA_VISIBLE_DEVICES=$1 xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 train_ail.py --cuda-id $1 --sweep_set $2 --env-id $3 --algo $4  --timesteps $5 --reg $6   --tag $7  --tune true 

