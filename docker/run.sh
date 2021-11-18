#!/bin/bash

# Local paths
#DATA_PATH="/media/darren/dcsky/datasets"
#CODE_PATH="/home/darren/Dropbox/code/perception/projects/SEE-MTDA"

# Monolith paths
DATA_PATH="/mnt/big-data/darren/data"
CODE_PATH="/mnt/big-data/darren/code/SEE-MTDA"

GPU_NAME="g1080ti"

declare -A GPUS
GPUS["g2080ti"]="0"
GPUS["g1080ti"]="1"
GPUS["g1080"]="2"

GPU_ID=${GPUS[$GPU_NAME]}
NAME="darren_see-mtda_${GPU_NAME}"
echo "Starting docker container $NAME - gpu:$GPU_NAME, ID:${GPUS[$GPU_NAME]}"

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

VOLUMES="       --volume=$DATA_PATH/kitti/3d_object_detection:/SEE-MTDA/data/kitti
                --volume=$DATA_PATH/nuscenes:/SEE-MTDA/data/nuscenes
                --volume=$DATA_PATH/waymo:/SEE-MTDA/data/waymo
                --volume=$DATA_PATH/baraja:/SEE-MTDA/data/baraja
                --volume=$CODE_PATH:/SEE-MTDA"

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
        --workdir=/SEE-MTDA \
        anonymoustofu/see-mtda:see-v1.0
        # darrenjkt/mmdetection:v2.14.0-1
	

