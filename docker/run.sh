#!/bin/bash

# Edit these paths. The volume mounting below assumes your datasets are e.g. data/Baraja, data/KITTI etc.
DATA_PATH="/media/darren/dcsky/datasets"
CODE_PATH="/home/darren/Nextcloud/professional/code/projects/SEE-MTDA"

# -i options are "see" and "detector". 
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

# -i options are "see" and "detector". 
if [ "$image" != "see" ] && [ "$image" != "detector" ]; then
        echo "Invalid docker image. Please specify: see or detector" >&2
        exit 1
fi

# Setting GPUs.
# Accepts e.g. -g 0 for single gpu or -g 0,1,2 for multiple gpus
if [ "$gpus" == "" ]; then
        echo "Please choose which GPUs to use (e.g. -g 0 or -g 0,1,2)" >&2
        exit 1  
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
                --volume=$CODE_PATH/see:/SEE-MTDA/see
                --volume=$CODE_PATH/detector/tools:/SEE-MTDA/detector/tools
                --volume=$CODE_PATH/detector/output:/SEE-MTDA/detector/output
                --volume=$CODE_PATH/model_zoo:/SEE-MTDA/model_zoo"

# Setup visualization for point cloud demos
VISUAL="        --env=DISPLAY
                --env=QT_X11_NO_MITSHM=1
                --volume=/tmp/.X11-unix:/tmp/.X11-unix"

# Start docker image
xhost +local:docker

echo "Running the docker image for see-mtda:${image} [GPUS: ${GPU_ID}]"
docker_image="darrenjkt/see-mtda:${image}-v1.0"
# docker_image="darrenjkt/openpcdet:v0.3.0"

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
