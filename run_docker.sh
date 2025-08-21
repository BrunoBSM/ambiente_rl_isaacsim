#!/bin/bash

xhost +

WORKPATH=$(pwd)

docker run --rm --name isaac-sim-container --entrypoint bash -it --runtime=nvidia --gpus all --ipc=host --network=host \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -e DISPLAY \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v ${WORKPATH}:/isaac-sim/ambiente_rl_isaacsim \
  ambiente-rl-isaacsim:4.5.0