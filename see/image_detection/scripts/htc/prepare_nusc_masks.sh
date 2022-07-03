#!/bin/bash

# cameras=("image_2" "image_3") if you want to get masks for both cameras
declare -a cameras=('CAM_FRONT' 'CAM_FRONT_RIGHT' 'CAM_BACK_RIGHT' 'CAM_BACK' 'CAM_BACK_LEFT' 'CAM_FRONT_LEFT'
)
for cam in "${cameras[@]}"
do 
	echo "Getting masks for $cam"

	python /SEE-MTDA/see/generate_masks.py \
	--config "/SEE-MTDA/see/mmdetection/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py" \
	--checkpoint "/SEE-MTDA/model_zoo/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth" \
	--data_dir "/SEE-MTDA/data/nuscenes/custom_t4025-v3980/samples/$cam" \
	--output_json "/SEE-MTDA/data/nuscenes/custom_t4025-v3980/masks/htc/$cam.json" \
	--instance_mask

done