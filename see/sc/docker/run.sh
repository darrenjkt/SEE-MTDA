#!/bin/bash

# Local paths
#DATA_PATH="/media/darren/dcsky/datasets"
DATA_PATH="/media/darren/Samsung_T5/"
CODE_PATH="/home/darren/Nextcloud/professional/code/projects/meshcloud"

# Monolith paths
# DATA_PATH="/mnt/big-data/darren/data"
# CODE_PATH="/mnt/big-data/darren/code/meshcloud"

GPU_ID=0

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

VOLUMES="       --volume=$DATA_PATH/kitti/3d_object_detection:/meshcloud/data/kitti
                --volume=$DATA_PATH/nuscenes:/meshcloud/data/nuscenes
		--volume=$DATA_PATH/waymo:/meshcloud/data/waymo
		--volume=$DATA_PATH/baraja:/meshcloud/data/baraja
                --volume=$CODE_PATH:/meshcloud"

VISUAL="	--env=DISPLAY
		--env=QT_X11_NO_MITSHM=1
		--volume=/tmp/.X11-unix:/tmp/.X11-unix"

xhost +local:docker

# Ctrl-P-Q will detach container without killing it
docker  run -it -d --rm \
        $VOLUMES \
        $ENVS \
        $VISUAL \
        --runtime=nvidia \
        --gpus device=$GPU_ID \
        --privileged \
        --net=host \
        --workdir=/meshcloud \
	darrenjkt/meshcloud:v1.1

