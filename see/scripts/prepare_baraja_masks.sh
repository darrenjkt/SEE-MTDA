#!/bin/bash

declare -a cameras=("front")

for cam in "${cameras[@]}"
do 
	echo "Getting masks for camera: $cam"

	python /SEE-MTDA/see/generate_masks.py \
	--config "/SEE-MTDA/see/mmdetection/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py" \
	--checkpoint "/SEE-MTDA/model_zoo/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth" \
	--data_dir "/SEE-MTDA/data/baraja/test/image/$cam" \
	--output_json "/SEE-MTDA/data/baraja/test/masks/htc/$cam.json"

done