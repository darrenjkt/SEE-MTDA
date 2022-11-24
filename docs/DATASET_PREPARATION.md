# Dataset Preparation

### Baraja Spectrum-Scan™ Dataset
Please download the [Baraja Spectrum-Scan™ Dataset](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/EbBLKPoamxJGh6gmTAAv9hgBqo0w_d7JrHOfCzitZ8xI5Q?e=cP3uwH) and organise the downloaded files as follows:
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

* Please download our [Nuscenes subset](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/Ea6MW4jPciVDttZx2iXVaOoBRZHnz1uAMMKiI2yATtbiHw?e=vflIjI) (39GB Compressed) and 
organize the downloaded files as follows: 
```
SEE-MTDA
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
│   │   │   │── infos_meshed_NUS-GM-ORH005
├── see
├── detector
...
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```
In SEE-MTDA, we use a subset of the nuScenes. We sorted the scenes by number of cars in the scene and selected the top 100 scenes, leading to 4025 frames. We modified the generated infos file from above to reflect this subset. The official dataset can be obtained from [here](https://www.nuscenes.org/download).

### Waymo Open Dataset
* Please download our [Waymo subset](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/EUseiL7sd-tLt3DwzcLq0YUB1rhlam-PORdgjIbOc9zBMA?e=EuJ4I9) (3.9GB Compressed) and organize the downloaded files as follows: 
```
SEE-MTDA
├── data
│   ├── waymo
│   │   │── infos_openpcdetv0.3.0
│   │   │── waymo_meshed_WAY-GM-ORH005
│   │   ...
├── see
├── detector
...
```
In SEE-MTDA, we use a subset of the Waymo where we selected the 100th frame from each sequence, leading to 1000 frames. We modified the waymo_processed_data and infos accordingly. Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. The official Waymo Open Dataset can be obtained at this [link](https://waymo.com/open/download/).
