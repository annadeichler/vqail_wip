
if [ $# -eq 1 ]
    then
        DIR="/home/$USER/vqail/"
    else
        DIR=$1
fi

if [ $# -eq 2 ]
    then
        MEM="6g"
    else
        MEM=$2
fi

xhost +local:root
Xvfb :1 -screen 0 1024x768x24 +extension GLX +render -noreset >> xsession.log 2>&1 &
env DISPLAY=:1 docker run \
  --rm \
  -it \
  --gpus all \
  --shm-size 8G \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
 -v ${PWD}/display/Xauthority:/tmp/.Xauthority \
 -v /home/deichler/vqail/:/vqail \
  vqail \
  bash