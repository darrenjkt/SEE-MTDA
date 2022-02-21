# Dataset Preparation

### Baraja Spectrum-Scan™ Dataset
Please download the [Baraja Spectrum-Scan™ Dataset](https://drive.google.com/file/d/16_azaVGiMVycGH799FX2RyRIWHrslU0R/view?usp=sharing) and organise the downloaded files as follows:
```
SEE-MTDA
├── data
│   ├── baraja
│   │   │── ImageSets
│   │   │── test
│   │   │   ├──pcd & masks & image & calib & label
│   │   │── infos
│   │   │   ├──baraja_infos_test.pkl
├── see
├── detector
...
```
All the files should already be provided without needing extra steps. We have provided the Hybrid Task Cascade segmentation masks as well as the infos required. If you'd like to regenerate the infos pkl file, you can run:
```python 
python -m pcdet.datasets.baraja.meshed_baraja_dataset create_baraja_infos tools/cfgs/dataset_configs/baraja_dataset_meshed.yaml
```

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:
```
SEE-MTDA
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
│   │   │── infos_openpcdetv0.3.0
│   │   │   ├──kitti_train_infos.pkl
│   │   │   ├──kitti_val_infos.pkl
│   │   │   ├──kitti_trainval_infos.pkl
├── see
├── detector
...
```

* Generate the data infos by running the following command and placing it into the folder structure above: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
Or download from our pre-generated infos [here](https://drive.google.com/drive/folders/1BanTsv8zWqmL7W1C_QXpultB0b01tnh3?usp=sharing).

### NuScenes Dataset

* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and 
organize the downloaded files as follows: 
```
SEE-MTDA
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
│   │   │   │── infos_openpcdetv0.3.0
│   │   │   │   ├──nuscenes_infos_10sweeps_train.pkl
│   │   │   │   ├──nuscenes_infos_10sweeps_val.pkl
├── see
├── detector
...
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```

* Generate the data infos by running the following command (it may take several hours): 
```python 
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \ 
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
```
In SEE-MTDA, we use a subset of the nuScenes. We sorted the scenes by number of cars in the scene and selected the top 100 scenes, leading to 4025 frames. We modified the generated infos file from above to reflect this subset. 

### Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/). We only need the training data for Waymo.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows:  
```
SEE-MTDA
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
├── see
├── detector
...
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-2-0 --user
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data` to see how many records that have been processed): 
```python 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```
In SEE-MTDA, we use a subset of the Waymo where we selected the 100th frame from each sequence, leading to 1000 frames. We modified the waymo_processed_data and infos accordingly. Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 
