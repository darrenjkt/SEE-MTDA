#!/bin/bash

# Local paths
#DATA_PATH="/media/darren/dcsky/datasets"
#CODE_PATH="/home/darren/Dropbox/code/perception/projects/ST3D"

# Monolith paths
DATA_PATH="/mnt/big-data/darren/data"
CODE_PATH="/mnt/big-data/darren/code/ST3D"

GPU_NAME="g2080ti"

declare -A GPUS
GPUS["g2080ti"]="0"
GPUS["g1080ti"]="1"
GPUS["g1080"]="2"

GPU_ID=${GPUS[$GPU_NAME]}
NAME="darrenst3d_${GPU_NAME}"
echo "Starting docker container $NAME - gpu:$GPU_NAME, ID:${GPUS[$GPU_NAME]}"

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

VOLUMES="       --volume=$DATA_PATH/kitti/3d_object_detection:/ST3D/data/kitti
                --volume=$DATA_PATH/nuscenes:/ST3D/data/nuscenes
                --volume=$DATA_PATH/waymo:/ST3D/data/waymo
                --volume=$DATA_PATH/baraja:/ST3D/data/baraja
                --volume=$CODE_PATH/pcdet:/ST3D/pcdet
                --volume=$CODE_PATH/tools:/ST3D/tools
                --volume=$CODE_PATH/output:/ST3D/output"

VISUAL="        --env=DISPLAY
                --env=QT_X11_NO_MITSHM=1
                --volume=/tmp/.X11-unix:/tmp/.X11-unix"

xhost +local:docker

docker  run -d -it --rm \
        $VOLUMES \
        $ENVS \
        $VISUAL \
        --runtime=nvidia \
        --name=$NAME \
        --gpus $GPU_ID \
        --privileged \
        --net=host \
        --workdir=/ST3D \
	darrenjkt/st3d:v0.3.0

