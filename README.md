# SEE-MTDA

Code release for the paper **See Eye to Eye: A Lidar-Agnostic 3D Detection Framework for Unsupervised Multi-Target Domain Adaptation**

![pipeline](./docs/pipeline.png)

Our code is based on [OpenPCDet v0.3.0](https://github.com/open-mmlab/OpenPCDet/tree/v0.3.0) with DA configurations adopted from [ST3D](https://github.com/CVMI-Lab/ST3D). 

## Detectors

### Source: Waymo
| Source | Model | Method | Download | 
|--------|------:|:------:|:--------:|
| Waymo | SECOND-IoU | SEE | [model](https://drive.google.com/file/d/1AP436Sq8XKM6sU8MchHgUKTKKgVavvQl/view?usp=sharing) |
| Waymo | PV-RCNN | SEE | [model](https://drive.google.com/file/d/1oaRA-LZelDKfU8eFii_h6VYeoYnXjAix/view?usp=sharing) |

Due to the [Waymo Dataset License Agreement](https://waymo.com/open/terms/) we can't provide the Source-only models for the above. You should achieve a similar performance by training with the default configs. 

### Source: nuScenes
| Source | Model | Method | Download | 
|--------|------:|:------:|:--------:|
| nuScenes | SECOND-IoU | Source-only | [model](https://drive.google.com/file/d/1ZDJqBWJzM-cfCYj_nrtRMNSXauv5enUz/view?usp=sharing) | 
| nuScenes | SECOND-IoU | SEE | [model](https://drive.google.com/file/d/1NkjttovNoNvktSFwJu-RCc6Qv4ErRsTf/view?usp=sharing) |
| nuScenes | PV-RCNN | Source-only | [model](https://drive.google.com/file/d/1vDEErtKlRWdmDM0bqaQhq9iQApR0Hl6C/view?usp=sharing) | 
| nuScenes | PV-RCNN | SEE | [model](https://drive.google.com/file/d/1NBBClCyapwf5vEds_XDGJqUV68RpIwJx/view?usp=sharing) |

## Installation
Please refer to INSTALL.md for installation instructions.

## Dataset Preparation
Please refer to DATASET_PREPARATION.md instructions on downloading and preparing datasets. 

## Usage

#### 1. Object Isolation
Get instance masks for all images. 
```
bash see/mmdetection/tools/see_masks/prepare_baraja_masks.sh
```

#### 2. Surface Completion and Point Sampling
Create meshes using instance masks and sample from the meshes
```
python create_meshes.py --cfg_file cfgs/BAR-DM-ORH005.yaml
```

#### 3. Point Cloud Detector 
For training, see the following example. Replace the cfg file with any of the other cfg files in the `tools/cfgs` folder. 
```
cd detector/tools
python train.py --cfg_file cfgs/source-waymo/secondiou/see/secondiou_ros_custom1000_GM-ORH005.yaml
```

For testing, download the models from the links above and modify the cfg and ckpt paths below. For the cfg files, please link to the yaml files in the output folder instead of the ones in the `tools/cfg` folder. 
```
python test.py --cfg_file /ST3D/output/source-waymo/secondiou/see/secondiou_ros_custom1000_GM-ORH005/default/secondiou_ros_custom1000_GM-ORH005_eval-baraja100.yaml \
--ckpt /ST3D/output/SEE-models/w-k_secondiou_see_6552.pth \
--batch_size 1
```
