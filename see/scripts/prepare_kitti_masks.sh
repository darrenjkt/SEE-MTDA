#!/bin/bash

# cameras=("image_2" "image_3") if you want to get masks for both cameras
declare -a cameras=("image_2")

for cam in "${cameras[@]}"
do 
	echo "Getting masks for $cam"

	python /SEE-MTDA/see/mmdetection/tools/see_masks/create_masks.py \
	--config "/SEE-MTDA/see/mmdetection/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py" \
	--checkpoint "/SEE-MTDA/see/mmdetection/checkpoints/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth" \
	--data_dir "/SEE-MTDA/data/kitti/training/$cam" \
	--output_json "/SEE-MTDA/data/kitti/training/masks/htc/$cam.json"

done