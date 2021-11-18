#!/bin/bash

declare -a cameras=("image_2" "image_3")

for cam in "${cameras[@]}"
do 
	echo "Getting masks for $cam"

	python /mmdetection/tools/see_masks/create_masks.py \
	--config "/mmdetection/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py" \
	--checkpoint "/mmdetection/checkpoints/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth" \
	--data_dir "/mmdetection/data/kitti/training/$cam" \
	--output_json "/mmdetection/data/kitti/training/masks/htc/$cam.json"

done