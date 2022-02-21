# SEE-MTDA

Being a multi-target domain adaptation approach, we enable **any** novel SOTA detectors to become an **agnostic** model for different lidar sensors without requiring any form of training. The same model trained with SEE can support any kind of scan pattern. We provide download links for all the models that were used in the paper. 

We aim to support novel SOTA detectors as they emerge in order to provide accessibility for those who do not have the training or manual labelling resources. We believe that this will be a great contribution to supporting novel and non-conventional lidars that differ from the popular lidars used in the research context.

![pipeline](./docs/pipeline.png)

This project builds upon the progress of other outstanding codebases in the computer vision community. We acknowledge the works of the following codebases in our project: 
- Our instance segmentation code uses [mmdetection](https://github.com/open-mmlab/mmdetection).
- Our detector code is based on [OpenPCDet v0.3.0](https://github.com/open-mmlab/OpenPCDet/tree/v0.3.0) with DA configurations adopted from [ST3D](https://github.com/CVMI-Lab/ST3D). 

## Model Zoo
Please place all downloaded models into the `model_zoo` folder. See the model zoo [readme](https://github.com/darrenjkt/SEE-MTDA/blob/readme-edits/model_zoo/README.md) for more details. All models were trained with a single 2080Ti for approximately 30-40hrs with our nuScenes and Waymo subsets. 

### Instance Segmentation Models
Pre-trained instance segmentation models can be obtained from the model zoo of [mmdetection](https://github.com/open-mmlab/mmdetection). Our paper uses 
Hybrid Task Cascade (download model [here](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth)).

### Source: Waymo
| Model | Method | Download | 
|------:|:------:|:--------:|
| [SECOND-IoU](https://github.com/darrenjkt/SEE-MTDA/blob/main/detector/tools/cfgs/source-waymo/secondiou/see/secondiou_ros_custom1000_GM-ORH005.yaml) | SEE | [model](https://drive.google.com/file/d/1AP436Sq8XKM6sU8MchHgUKTKKgVavvQl/view?usp=sharing) |
| [PV-RCNN](https://github.com/darrenjkt/SEE-MTDA/blob/main/detector/tools/cfgs/source-waymo/pvrcnn/see/pvrcnn_ros_custom1000_GM-ORH005.yaml) | SEE | [model](https://drive.google.com/file/d/1oaRA-LZelDKfU8eFii_h6VYeoYnXjAix/view?usp=sharing) |

Due to the [Waymo Dataset License Agreement](https://waymo.com/open/terms/) we can't provide the Source-only models for the above. You should achieve a similar performance by training with the default configs and training subsets.

### Source: nuScenes
| Model | Method | Download | 
|------:|:------:|:--------:|
| [SECOND-IoU](https://github.com/darrenjkt/SEE-MTDA/blob/main/detector/tools/cfgs/source-nuscenes/secondiou/baselines/secondiou_custom4025.yaml) | Source-only | [model](https://drive.google.com/file/d/1ZDJqBWJzM-cfCYj_nrtRMNSXauv5enUz/view?usp=sharing) | 
| [SECOND-IoU](https://github.com/darrenjkt/SEE-MTDA/blob/main/detector/tools/cfgs/source-nuscenes/secondiou/see/secondiou_ros_GM-ORH005.yaml) | SEE | [model](https://drive.google.com/file/d/1NkjttovNoNvktSFwJu-RCc6Qv4ErRsTf/view?usp=sharing) |
| [PV-RCNN](https://github.com/darrenjkt/SEE-MTDA/blob/main/detector/tools/cfgs/source-nuscenes/pvrcnn/baselines/pvrcnn_custom4025.yaml) | Source-only | [model](https://drive.google.com/file/d/1vDEErtKlRWdmDM0bqaQhq9iQApR0Hl6C/view?usp=sharing) | 
| [PV-RCNN](https://github.com/darrenjkt/SEE-MTDA/blob/main/detector/tools/cfgs/source-nuscenes/pvrcnn/see/pvrcnn_ros_GM-ORH005.yaml) | SEE | [model](https://drive.google.com/file/d/1NBBClCyapwf5vEds_XDGJqUV68RpIwJx/view?usp=sharing) |

## Installation
This repo is structured in 2 parts: see and detector. For each part, we have provided a separate docker image which can be obtained as follows:
- `docker pull darrenjkt/see-mtda:see-v1.0`
- `docker pull darrenjkt/see-mtda:detector-v1.0`

We have provided a [script](https://github.com/darrenjkt/SEE-MTDA/blob/main/docker/run.sh) to run the necessary docker images for each part. Please edit the folder names for mounting local volumes into the docker image. We currently do not provide other installation methods. If you'd like to install natively, please refer to the Dockerfile for [see](https://github.com/darrenjkt/SEE-MTDA/blob/main/docker/see/Dockerfile) or [detector](https://github.com/darrenjkt/SEE-MTDA/blob/main/docker/detector/Dockerfile) for more information about installation requirements. 

Note that `nvidia-docker2` is required to run these images. We have tested that this works with NVIDIA Docker: 2.5.0 or higher. Additionally, unless you are training the models, the tasks below require at least 1 GPU with a minimum of 5GB memory. This may vary if using different types of image instance segmentation networks or 3D detectors. 

## Dataset Preparation
Please refer to [DATASET_PREPARATION](https://github.com/darrenjkt/SEE-MTDA/blob/main/docs/DATASET_PREPARATION.md) for instructions on downloading and preparing datasets. 

## Usage
In this section, we provide instructions specifically for the [Baraja Spectrum-Scan™](https://drive.google.com/file/d/16_azaVGiMVycGH799FX2RyRIWHrslU0R/view?usp=sharing) Dataset as an example of adoption to a novel industry lidar. Please modify the configuration files as necessary to train/test for different datasets. 

### 1. SEE
In this phase, we isolate the objects, create meshes and sample from them. Firstly, run the docker image as follows. For instance segmentation, a single GPU is sufficient. If you simply wish to train/test the Source-only baseline, then you can skip this "SEE" section. 
```
# Run docker image for SEE
bash docker/run.sh -i see -g 0

# Enter docker container. Use `docker ps` to find the name of the newly created container from the above command.
docker exec -it ${CONTAINER_NAME} /bin/bash
```
**a) Instance Segmentation**: Get instance masks for all images. If you are using the baraja dataset, we've provided the masks in the download link. Feel free to skip this part. We also provide a script for generating KITTI masks. 
```
bash see/scripts/prepare_baraja_masks.sh
```
**b) Transform to Canonical Domain**: Once we have the masks, we can isolate the objects and transform them into the canonical domain. We have multiple configurations for nuScenes, Waymo, KITTI and Baraja datasets. We code "DM" (Det-Mesh) as using instance segmentation (target domain) and "GM" (GT-Mesh) as using ground truth boxes (source domain) to transform the objects. Here is an example for the Baraja dataset.
```
cd see
python surface_completion.py --cfg_file sc/cfgs/BAR-DM-ORH005.yaml
```

### 2. Point Cloud Detector 
To run train/test the detector, run the following docker image with our provided script. If you'd like to specify gpus to use in the docker container, you can do so with e.g. `-g 0` or `-g 0,1,2`. For testing, a single gpu is sufficient. With docker, this should work out-of-the-box; however, if there are issues finding libraries, please refer to [INSTALL.md](https://github.com/darrenjkt/SEE-MTDA/blob/main/docs/INSTALL.md).
```
# Run docker image for Point Cloud Detector
bash docker/run.sh -i detector -g 0

# Enter docker container. Use `docker ps` to find the newly created container from the above image.
docker exec -it ${CONTAINER_NAME} /bin/bash
```
**a) Training**: see the following example. Replace the cfg file with any of the other cfg files in the `tools/cfgs` folder (cfg file links provided in Model Zoo). Number of epochs, batch sizes and other training related configurations can be found in the cfg file. 
```
cd detector/tools
python train.py --cfg_file cfgs/source-waymo/secondiou/see/secondiou_ros_custom1000_GM-ORH005.yaml
```

**b) Testing**: download the models from the links above and modify the cfg and ckpt paths below. For the cfg files, please link to the yaml files in the output folder instead of the ones in the `tools/cfg` folder. Here is an example for the Baraja dataset. 
```
cd detector/tools
python test.py --cfg_file /SEE-MTDA/detector/output/source-waymo/secondiou/see/secondiou_ros_custom1000_GM-ORH005/default/secondiou_ros_custom1000_GM-ORH005_eval-baraja100.yaml \
--ckpt /SEE-MTDA/model_zoo/waymo_secondiou_see_6552.pth \
--batch_size 1
```
The location of further testing configuration files for the different tasks can be found [here](https://github.com/anonymoustofu/SEE-MTDA/blob/main/docs/TESTING_CONFIGURATIONS.md). These should give similar results as our paper. Please modify the cfg file path and ckpt model in the command above accordingly. 

