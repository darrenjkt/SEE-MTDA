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
