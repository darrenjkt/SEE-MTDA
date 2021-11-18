#!/bin/bash

# Local paths
#DATA_PATH="/media/darren/dcsky/datasets"
DATA_PATH="/media/darren/Samsung_T5"
CODE_PATH="/home/darren/Nextcloud/professional/code/projects/mmdetection"

# Monolith paths
#DATA_PATH="/mnt/big-data/darren"
#CODE_PATH="/home/darren/code/mmdetection"

GPU_ID=0

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

VOLUMES="       --volume=$DATA_PATH/kitti/3d_object_detection:/mmdetection/data/kitti
                --volume=$DATA_PATH/nuscenes:/mmdetection/data/nuscenes
                --volume=$DATA_PATH/waymo:/mmdetection/data/waymo
                --volume=$DATA_PATH/baraja:/mmdetection/data/baraja
                --volume=$CODE_PATH/tools:/mmdetection/tools
                --volume=$CODE_PATH/mmdet:/mmdetection/mmdet
                --volume=$CODE_PATH/checkpoints:/mmdetection/checkpoints"

# Ctrl-P-Q will detach container without killing it
docker  run -it -d --rm \
        $VOLUMES \
        $ENVS \
        --gpus device=$GPU_ID \
        --privileged \
        --net=host \
        --workdir=/mmdetection \
	darrenjkt/mmdetection:v2.14.0-1
