#!/bin/bash

DATA_PATH="/mnt/big-data/darren/data"
CODE_PATH="/mnt/big-data/darren/code/SEE-MTDA"

while getopts ":i:g:" flag
do
        case "${flag}" in
        i) image=${OPTARG};;    
        g) gpus=${OPTARG};;    
        \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1;;
        :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1;;  
        esac
done

# Setting GPUs
if [ "$gpus" == "" ]; then
        GPU_ID="all"     
else
        GPU_ID=$gpus  
fi
ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

# Modify these paths as necessary to mount the data
VOLUMES="       --volume=$DATA_PATH/kitti/3d_object_detection:/SEE-MTDA/data/kitti
                --volume=$DATA_PATH/nuscenes:/SEE-MTDA/data/nuscenes
                --volume=$DATA_PATH/waymo:/SEE-MTDA/data/waymo
                --volume=$DATA_PATH/baraja:/SEE-MTDA/data/baraja
                --volume=$CODE_PATH:/SEE-MTDA"

# Setup visualization for point cloud demos
VISUAL="        --env=DISPLAY
                --env=QT_X11_NO_MITSHM=1
                --volume=/tmp/.X11-unix:/tmp/.X11-unix"

# Start docker image
xhost +local:docker

echo "Running the docker image for see-mtda:${image} [GPUS: ${GPU_ID}]"
docker_image="anonymoustofu/see-mtda:${image}-v1.0"

docker  run -d -it --rm \
$VOLUMES \
$ENVS \
$VISUAL \
--runtime=nvidia \
--gpus $GPU_ID \
--privileged \
--net=host \
--workdir=/SEE-MTDA \
$docker_image   