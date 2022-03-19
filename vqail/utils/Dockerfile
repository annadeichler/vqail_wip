FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install sudo 

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list


ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev python3.8 python3-pip python3-setuptools \
    libgtk3.0 libsm6 python3-venv cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.8-dev \
    libboost-python-dev libtinyxml-dev bash python3-tk \

    wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev graphviz graphviz-dev patchelf
RUN pip3 install pip --upgrade

RUN rm -rf /var/lib/apt/lists/*


RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.8-dev python3.8 python3-pip

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.8 /usr/bin/python
RUN ln -s /env/bin/pip3.8 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest
RUN  update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN python -m pip install --upgrade --force-reinstall pip

RUN pip install --upgrade pip
RUN pip install pymongo
RUN pip install numpy scipy pyyaml matplotlib ruamel.yaml networkx tensorboardX pygraphviz
RUN pip  install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-torch-1.8.0+cu111.html
RUN pip install torch-geometric
RUN pip install gym
RUN pip install gym[atari]
RUN pip install pybullet cffi
RUN pip install seaborn
RUN pip install tensorflow

RUN pip install six beautifulsoup4 termcolor num2words
RUN pip install lxml tabulate coolname lockfile glfw
RUN pip install Cython
RUN pip install sacred
RUN pip install imageio

RUN pip install xmltodict

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN pip install stable-baselines3[extra] sb3_contrib wandb

RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz  -P /$user/.mujoco/ \
    && tar -xvf /$user/.mujoco/mujoco210-linux-x86_64.tar.gz  --no-same-owner  -C /$user/.mujoco/  \
    && rm /$user/.mujoco/mujoco210-linux-x86_64.tar.gz  

ENV LD_LIBRARY_PATH /$user/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$user/.mujoco/mujoco210/bin
RUN export MJLIB_PATH=/$user/.mujoco/mujoco210/bin/libmujoco210.so

RUN export MUJOCO_GL="glfw" \
 &&  export MJLIB_PATH=/$user/.mujoco/mujoco210/bin/libmujoco200.dylib \
 &&  export MJKEY_PATH=/$user/.mujoco/mujoco210/mjkey.txt \
 &&  export LD_LIBRARY_PATH=/$user/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
 &&  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$user/.mujoco/mujoco210/bin \
 &&  export MUJOCO_PY_MUJOCO_PATH=/$user/.mujoco/mujoco210/ \
 &&  export MUJOCO_PY_MJPRO_PATH=/$user/.mujoco/mujoco210/ \
 &&  export MUJOCO_PY_MJKEY_PATH=/$user/.mujoco/mujoco210/mjkey.txt \
 && pip install git+https://github.com/openai/mujoco-py

ENV MUJOCO_PY_MUJOCO_PATH=/$user/.mujoco/mujoco210/ 
RUN  apt-get update \
       &&  apt install nano \
       &&  apt-get install -y libglfw3 libglew2.0 libglew-dev 

RUN  apt-get install -y libglew-dev 
RUN python -m pip install --upgrade --force pip
ENV MJLIB_PATH=/$user/.mujoco/mujoco210/bin/libmujoco210.so

RUN pip install dm_control gym_miniworld
RUN sudo apt install xvfb   

ENV PYTHONPATH "${PYTHONPATH}:/vqail/gym-minigrid/"
ENV PYTHONPATH "${PYTHONPATH}:/vqail/gym-miniworld/"
RUN export PYTHONPATH=$PYTHONPATH:$HOME/vqail/gym-miniworld

RUN pip install atari-py==0.2.5
RUN git config --global url."https://".insteadOf git://

RUN pip install git+git://github.com/denisyarats/dmc2gym.git

# COPY  ../utils/  /go/src/app
# RUN mkdir -p /vqail/gym-minigrid

# COPY requirements.txt /vqail/gym-minigrid

# RUN pip install -r /usr/src/app/requirements.txt

# COPY . /usr/src/app
# WORKDIR /usr/src/app