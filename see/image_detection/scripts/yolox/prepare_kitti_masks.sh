#!/bin/bash

model="yolox"

# cameras=("image_2" "image_3") if you want to get masks for both cameras
declare -a cameras=("image_2")

for cam in "${cameras[@]}"
do 
	echo "Getting masks for $cam"

	python /SEE-MTDA/see/generate_masks.py \
	--config "/SEE-MTDA/see/mmdetection/configs/$model/yolox_x_8x8_300e_coco.py" \
	--checkpoint "/SEE-MTDA/model_zoo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth" \
	--data_dir "/SEE-MTDA/data/kitti/training/$cam" \
	--output_json "/SEE-MTDA/data/kitti/training/masks/$model/$cam.json" 

done