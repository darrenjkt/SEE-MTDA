# SEE-MTDA

This is the official codebase for 

[RAL 2022] See Eye to Eye: A Lidar-Agnostic 3D Detection Framework for Unsupervised Multi-Target Domain Adaptation.

[Video](https://www.youtube.com/watch?v=iw-AKNLUfNQ) | [Arxiv](https://arxiv.org/abs/2111.09450) | [IEEE](https://ieeexplore.ieee.org/document/9804815)

**An extension of this work which uses deep learning based shape completion can be found here: https://github.com/darrenjkt/SEE-VCN

### Abstract
Sampling discrepancies between different manufacturers and models of lidar sensors result in inconsistent representations of objects. This leads to performance degradation when 3D detectors trained for one lidar are tested on other types of lidars. Remarkable progress in lidar manufacturing has brought about advances in mechanical, solid-state, and recently, adjustable scan pattern lidars. For the latter, existing works often require fine-tuning the model each time scan patterns are adjusted, which is infeasible. We explicitly deal with the sampling discrepancy by proposing a novel unsupervised multi-target domain adaptation framework, SEE, for transferring the performance of state-of-the-art 3D detectors across both fixed and flexible scan pattern lidars without requiring fine-tuning of models by end-users. Our approach interpolates the underlying geometry and normalises the scan pattern of objects from different lidars before passing them to the detection network. We demonstrate the effectiveness of SEE on public datasets, achieving state-of-the-art results, and additionally provide quantitative results on a novel high-resolution lidar to prove the industry applications of our framework.

![pipeline](./docs/pipeline.png)

In the figure below, source-only denotes the approach where there is no domain adaptation. For the 2nd and 3rd columns, the same trained model is used for SEE. From below, we demonstrate that SEE-trained detectors give tighter bounding boxes and less false positives. 
![qualitative_github](https://user-images.githubusercontent.com/39115809/166407813-eaed28ef-079d-447d-b9d1-8ed36b858369.png)


## Model Zoo
Please place all downloaded models into the `model_zoo` folder. See the model zoo [readme](https://github.com/darrenjkt/SEE-MTDA/blob/main/model_zoo/README.md) for more details. All models were trained with a single 2080Ti for approximately 30-40hrs with our nuScenes and Waymo subsets. 

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
In this section, we provide instructions specifically for the Baraja Spectrum-Scan™ ([Download link](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/EbBLKPoamxJGh6gmTAAv9hgBqo0w_d7JrHOfCzitZ8xI5Q?e=cP3uwH)) Dataset as an example of adoption to a novel industry lidar. Please modify the configuration files as necessary to train/test for different datasets. 

### 1. SEE
In this phase, we isolate the objects, create meshes and sample from them. Firstly, run the docker image as follows. For instance segmentation, a single GPU is sufficient. If you simply wish to train/test the Source-only baseline, then you can skip this "SEE" section. 
```
# Run docker image for SEE
bash docker/run.sh -i see -g 0

# Enter docker container. Use `docker ps` to find the name of the newly created container from the above command.
docker exec -it ${CONTAINER_NAME} /bin/bash
```
**a) Instance Segmentation**: Get instance masks for all images using the HTC model ([link](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth)). If you are using the baraja dataset, we've provided the masks in the download link. Feel free to skip this part. We also provide a script for generating KITTI masks. 
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
The location of further testing configuration files for the different tasks can be found [here](https://github.com/darrenjkt/SEE-MTDA/blob/main/docs/TESTING_CONFIGURATIONS.md). These should give similar results as our paper. Please modify the cfg file path and ckpt model in the command above accordingly. 

# Reference
If you find our work useful in your research, please consider citing our paper:
```
@article{tsai2022see,
  title={See Eye to Eye: A Lidar-Agnostic 3D Detection Framework for Unsupervised Multi-Target Domain Adaptation},
  author={Tsai, Darren and Berrio, Julie Stephany and Shan, Mao and Worrall, Stewart and Nebot, Eduardo},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE}
}
```

# Acknowledgement
This project builds upon the progress of other outstanding codebases in the computer vision community. We acknowledge the works of the following codebases in our project: 
- Our instance segmentation code uses [mmdetection](https://github.com/open-mmlab/mmdetection).
- Our detector code is based on [OpenPCDet v0.3.0](https://github.com/open-mmlab/OpenPCDet/tree/v0.3.0) with DA configurations adopted from [ST3D](https://github.com/CVMI-Lab/ST3D). 
