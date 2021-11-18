import os
import cv2
import glob
import time
import mmcv
import json
import datetime
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pycocotools import mask
import setproctitle
from mask_utils import mask2polygon, segm2json, xyxy2xywh
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

INFO = {
    "description": "SEE Masks",
    "url": "",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "darrenjkt",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 0,
        'name': 'person',
        'supercategory': 'person',
    },
    {
        'id': 1,
        'name': 'bicycle',
        'supercategory': 'vehicle',
    },
    {
        'id': 2,
        'name': 'car',
        'supercategory': 'vehicle',
    },
]

coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }


def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' ')):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": date_captured,
    }

    return image_info

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get segmentation masks for image dataset and save to COCO json annotation format')
    parser.add_argument('--config', required=True, help='Model config file path')
    parser.add_argument('--checkpoint', required=True, help='Trained model checkpoint file')
    parser.add_argument('--data_dir', required=True, help='folder of images for feeding into seg network')
    parser.add_argument('--output_json', required=True, help='folder to save json annotation files')
    args = parser.parse_args()

    print("\n----- Args -----")
    for arg in vars(args):
        print (f'- {arg}: {getattr(args, arg)}')
    print("\n")

    return args
    
if __name__ == "__main__":

    start_time = time.time()
    args = parse_args()
    
    img_list = sorted(glob.glob(args.data_dir + "/*"))
    print(f'Found {len(img_list)} images in {args.data_dir}')
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    
    for idx, imgname in enumerate(img_list):        
        tic = time.time()
        num_images = len(img_list)        
        setproctitle.setproctitle(f'Masks:{idx/num_images*100:4.3}%')

        # Read image and create image_infos
        img = mmcv.imread(imgname)
        image_id = int(os.path.basename(imgname).split('.')[0])
        image_info = create_image_info(image_id=image_id, 
                                       file_name=os.path.basename(imgname), 
                                       image_size=img.shape)
        coco_output["images"].append(image_info)

        # Get segmentation mask
        result = inference_detector(model, img)
        segm_json_result = segm2json(result, score_thresh=0.3)
        num_predicted = len(segm_json_result)

        # Convert to COCO json format    
        for iidx, segm in enumerate(segm_json_result):

            category_info = {'id': segm["category_id"], 'is_crowd': 0}
            instance_id = f'1{image_id:0{len(str(num_images))}}{iidx:0{len(str(num_predicted))}}'

            seg_mask = segm['segmentation']
            binary_mask_encoded = mask.encode(np.asfortranarray(seg_mask.astype(np.uint8)))
            area = mask.area(binary_mask_encoded)        

            annotation_info = {
                "id": instance_id,
                "image_id": image_id,
                "category_id": category_info["id"],
                "iscrowd": category_info["is_crowd"],
                "area": area.tolist(),
                "bbox": segm['bbox'], 
                "segmentation": mask2polygon(seg_mask),
            } 

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

        print(f'{idx}/{num_images} | Time taken: {(time.time() - tic):.4}s')

    # Save JSON file
    os.makedirs(os.path.dirname(args.output_json),exist_ok=True)
    with open(args.output_json, 'w+') as output_json_file:
        json.dump(coco_output, output_json_file)

    print(f'Finished.\nTime taken: {time.time() - start_time}s')
