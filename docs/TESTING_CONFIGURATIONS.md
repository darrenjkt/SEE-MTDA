# Testing Configurations
We provide locations to the configuration files of all the different evaluation tasks for our reported results in the paper (in the model hyperlink). Please modify the `--cfg file` parameter accordingly in the `python test.py` command. The models of Waymo - KITTI and Waymo - Baraja are the same. Likewise for nuScenes - KITTI and nuScenes - Baraja. We've included the model download links for convenience. Please modify the ``--ckpt`` parameter accordingly for the model pth file. 

**Waymo - KITTI**
|  Model | Method | Car@R40 | Download |
|--------|:------:|:--------:|:--------:|
| [SECOND-IoU](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-waymo/secondiou/baseline/secondiou_custom1000/default/secondiou_custom1000.yaml) | Source-Only | 11.92 | - |
| [SECOND-IoU](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-waymo/secondiou/see/secondiou_ros_custom1000_GM-ORH005/default/secondiou_ros_custom1000_GM-ORH005_eval-DM-ORH005.yaml) | SEE | 65.52 | [model](https://drive.google.com/file/d/1AP436Sq8XKM6sU8MchHgUKTKKgVavvQl/view?usp=sharing) | 
| [PV-RCNN](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-waymo/pvrcnn/baseline/pvrcnn_custom1000/default/pvrcnn_custom1000.yaml) | Source-Only | 41.54 | - |
| [PV-RCNN](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-waymo/pvrcnn/see/pvrcnn_ros_custom1000_GM-ORH005/default/pvrcnn_ros_custom1000_GM-ORH005_eval-DM-ORH005.yaml) | SEE | 79.39 | [model](https://drive.google.com/file/d/1oaRA-LZelDKfU8eFii_h6VYeoYnXjAix/view?usp=sharing) |

**nuScenes - KITTI**
|  Model | Method | Car@R40 | Download |
|--------|:------:|:--------:|:------:|
| [SECOND-IoU](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-nuscenes/secondiou/baseline/secondiou_custom4025/default/secondiou_custom4025.yaml) | Source-Only | 16.39 | [model](https://drive.google.com/file/d/1ZDJqBWJzM-cfCYj_nrtRMNSXauv5enUz/view?usp=sharing) | 
| [SECOND-IoU](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-nuscenes/secondiou/see/secondiou_ros_GM-ORH005/default/secondiou_ros_GM-ORH005_eval-DM-ORH005.yaml) | SEE | 56.00 | [model](https://drive.google.com/file/d/1NkjttovNoNvktSFwJu-RCc6Qv4ErRsTf/view?usp=sharing) |
| [PV-RCNN](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-nuscenes/pvrcnn/baseline/pvrcnn_custom4025/default/pvrcnn_custom4025.yaml) | Source-Only | 48.03 | [model](https://drive.google.com/file/d/1vDEErtKlRWdmDM0bqaQhq9iQApR0Hl6C/view?usp=sharing) | 
| [PV-RCNN](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-nuscenes/pvrcnn/see/pvrcnn_ros_GM-ORH005/default/pvrcnn_ros_GM-ORH005_eval-DM-ORH005.yaml) | SEE | 72.51 | [model](https://drive.google.com/file/d/1NBBClCyapwf5vEds_XDGJqUV68RpIwJx/view?usp=sharing) |

**Waymo - Baraja**
| Model | Method | Car@R40 | Download |
|--------|:------:|:------:|:------:|
| [SECOND-IoU](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-waymo/secondiou/baseline/secondiou_custom1000/default/secondiou_custom1000_eval-baraja100.yaml) | Source-Only | 49.96 | - |
| [SECOND-IoU](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-waymo/secondiou/see/secondiou_ros_custom1000_GM-ORH005/default/secondiou_ros_custom1000_GM-ORH005_eval-baraja100.yaml) | SEE | 73.79 | [model](https://drive.google.com/file/d/1AP436Sq8XKM6sU8MchHgUKTKKgVavvQl/view?usp=sharing) | 
| [PV-RCNN](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-waymo/pvrcnn/baseline/pvrcnn_custom1000/default/pvrcnn_custom1000_eval-baraja100.yaml) | Source-Only | 76.14 | - |
| [PV-RCNN](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-waymo/pvrcnn/see/pvrcnn_ros_custom1000_GM-ORH005/default/pvrcnn_ros_custom1000_GM-ORH005_eval-baraja100.yaml) | SEE | 79.13 | [model](https://drive.google.com/file/d/1oaRA-LZelDKfU8eFii_h6VYeoYnXjAix/view?usp=sharing) |

**nuScenes - Baraja**

| Model | Method | Car@R40 | Download |
|--------|:------:|:------:|:------:|
| [SECOND-IoU](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-nuscenes/secondiou/baseline/secondiou_custom4025/default/secondiou_custom4025_eval-baraja100.yaml) | Source-Only | 1.02 | [model](https://drive.google.com/file/d/1ZDJqBWJzM-cfCYj_nrtRMNSXauv5enUz/view?usp=sharing) | 
| [SECOND-IoU](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-nuscenes/secondiou/see/secondiou_ros_GM-ORH005/default/secondiou_ros_GM-ORH005_eval-baraja100.yaml) | SEE | 34.54 | [model](https://drive.google.com/file/d/1NkjttovNoNvktSFwJu-RCc6Qv4ErRsTf/view?usp=sharing) |
| [PV-RCNN](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-nuscenes/pvrcnn/baseline/pvrcnn_custom4025/default/pvrcnn_custom4025_eval-baraja100.yaml) | Source-Only | 10.85 | [model](https://drive.google.com/file/d/1vDEErtKlRWdmDM0bqaQhq9iQApR0Hl6C/view?usp=sharing) | 
| [PV-RCNN](https://github.com/anonymoustofu/SEE-MTDA/blob/main/detector/output/source-nuscenes/pvrcnn/see/pvrcnn_ros_GM-ORH005/default/pvrcnn_ros_GM-ORH005_eval-baraja100.yaml) | SEE | 64.34 | [model](https://drive.google.com/file/d/1NBBClCyapwf5vEds_XDGJqUV68RpIwJx/view?usp=sharing) |
