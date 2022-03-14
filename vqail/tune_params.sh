CUDA_VISIBLE_DEVICES=$1  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 tune_params.py --cuda-id $1 --env-id  $2  --algo $3 --timesteps $4 --reg $5 --tag $5 --tune $6 --seeds 1000


CUDA_VISIBLE_DEVICES=1  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 train_ail.py --cuda-id 1 --env-id  MiniWorld-TMazeLeft-v0  --algo vqail --timesteps 500 --reg expire_codes  --tag expire_code_broad --tune true --seeds 1000


 CUDA_VISIBLE_DEVICES=2  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 train_ail.py --cuda-id 2 --env-id MiniWorld-PickupObjs-v0 --algo vqail --timesteps 1000 --reg expire_codes --seeds 1000 --tune false --tag num_objs_2_1 --num_objs 2

 CUDA_VISIBLE_DEVICES=0  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 train_ail.py --cuda-id 1 --env-id  MiniWorld-TMazeLeft-v0  --algo vqail --timesteps 500 --reg ortho_loss  --tag ortho_loss_set_0 --tune true --sweep_set 0

  CUDA_VISIBLE_DEVICES=1  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 train_ail.py --cuda-id 1 --env-id MiniWorld-PickupObjs-v0 --algo vqail --timesteps 500 --reg ortho_loss  --tune true --tag num_objs_2_ortho --num_objs 2 --sweep_set 2


  CUDA_VISIBLE_DEVICES=0  xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3.8 train_ail.py --cuda-id 0 --env-id MiniGrid-DoorKey-5x5-v0 --algo gail --timesteps 500 -   --tune true --sweep_set 1
