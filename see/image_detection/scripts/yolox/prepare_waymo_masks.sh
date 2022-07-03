#!/bin/bash

model="yolox"

declare -a cameras=('FRONT' 'FRONT_LEFT' 'FRONT_RIGHT' 'SIDE_LEFT' 'SIDE_RIGHT')
for cam in "${cameras[@]}"
do 
	echo "Getting masks for $cam"

	python /SEE-MTDA/see/generate_masks.py \
	--config "/SEE-MTDA/see/mmdetection/configs/$model/yolox_x_8x8_300e_coco.py.py" \
	--checkpoint "/SEE-MTDA/model_zoo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth" \
	--data_dir "/SEE-MTDA/data/waymo/custom_1000/image_lidar_projections/image/$cam" \
	--output_json "/SEE-MTDA/data/waymo/custom_1000/image_lidar_projections/masks/$model/$cam.json"

done