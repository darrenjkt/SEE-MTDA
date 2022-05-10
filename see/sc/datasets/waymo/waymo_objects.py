import os
import json
import sc.datasets.shared_utils as shared_utils
from PIL import Image, ImageEnhance
from pathlib import Path
import glob
import pickle
import time
import open3d as o3d
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import math
import itertools
tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# Order of enum taken from dataset.proto on official waymo-dataset github
LIDAR_CHANNELS = ['TOP','FRONT','SIDE_LEFT','SIDE_RIGHT','REAR']
CAMERA_CHANNELS = ['FRONT','FRONT_LEFT','FRONT_RIGHT','SIDE_LEFT','SIDE_RIGHT']
idx2lidar = {v+1:k for v, k in enumerate(LIDAR_CHANNELS)}
idx2camera = {v+1:k for v, k in enumerate(CAMERA_CHANNELS)}

class WaymoObjects:

    def __init__(self, root_dir, dataset_cfg, extra_tag):
        
        self.root_dir = Path(root_dir)
        self.dataset_cfg = dataset_cfg        
        self.extra_tag = extra_tag       
        self.mask_dir = self.root_dir / 'image_lidar_projections' / 'masks' / dataset_cfg.DET2D_MODEL if dataset_cfg.DET2D_MASK else None
        self.camera_channels = dataset_cfg.CAMERA_CHANNELS if dataset_cfg.DET2D_MASK else []
        self.masks = self.__load_masks() if dataset_cfg.DET2D_MASK else None
        self.idx2tokens = self.__load_tokens()
        self.record_files = glob.glob(str(self.root_dir.parent / "raw_data/*.tfrecord"))
        self.infos = self.__load_infos()
        self.classes = dataset_cfg.CLASSES
        self.dataset_name = dataset_cfg.NAME

    def __load_masks(self):
        print("Loading masks...")
        masks = {}
        for channel in self.camera_channels:
            mask_json = self.mask_dir / f"{channel}.json"
            masks[channel] = COCO(mask_json)
        return masks

    def __len__(self):
        return len(self.infos)

    def __load_infos(self):
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / 'waymo_infos_train.pkl'), 'rb') as trainpkl:
            train_infos = pickle.load(trainpkl)
        return train_infos
    
    def __load_tokens(self):
        idx2tokens = {}
        for camera in self.camera_channels:
            tokens = self.masks[camera].getImgIds(catIds=[])
            idx2tokens[camera] = {k:v for k,v in enumerate(tokens)}
        return idx2tokens

    def get_infos(self, idx):
        return self.infos[idx]

    def find_info_idx(self, infos, seq, fid):
        
        for i, dic in enumerate(infos):

            if dic['point_cloud']['lidar_sequence'] == seq and int(dic['point_cloud']['sample_idx']) == int(fid):
                return i
        return -1

    def update_infos(self, save_dir):
        saved_files = glob.glob(f'{str(save_dir)}/*/*.pcd')
        
        frame_ids = [int(Path(fname).stem) for fname in saved_files]
        seqs = [Path(fname).parent.stem for fname in saved_files]
        rel_pcds = ['/'.join(fname.split('/')[-3:]) for fname in saved_files]
        seq_id_path = zip(seqs, frame_ids, rel_pcds)
        
        for seq, fid, path in seq_id_path:

            infos_idx = self.find_info_idx(self.infos, seq, fid)
            if infos_idx != -1:
                self.infos[infos_idx]['meshed_lidar_path'] = path

                seq_output = open(str(save_dir / seq / f'{seq}.pkl'), 'wb')
                pickle.dump([self.infos[infos_idx]], seq_output)
            
        savepath = self.root_dir / f'infos_meshed_{self.extra_tag}'
        savepath.mkdir(parents=True, exist_ok=True)
        
        output = open(str(savepath / 'waymo_infos_train.pkl'), 'wb')
        pickle.dump(self.infos, output)
        
        print(f"Saved updated infos: {str(savepath / 'waymo_infos_train.pkl')}")

    def get_pointcloud(self, idx, disable_nlz_flag=False, tanhnorm=False):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']
        path = self.root_dir / 'waymo_processed_data' / sequence_name /f'{sample_idx:04}.npy'
        point_features = np.load(path)
        points_all, NLZ_flag = point_features[:,0:5], point_features[:, 5]
        if disable_nlz_flag:            
            points_all = points_all[NLZ_flag == -1]            
        if tanhnorm:
            points_all[:, 3] = np.tanh(points_all[:,3])

        return points_all[:,:3]

    
    def get_image(self, idx, camera_channel, brightness=1):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']

        img_path = self.root_dir / 'image_lidar_projections' / 'image' / camera_channel / f'{sequence_name}_{sample_idx:04}.png'
        img = Image.open(img_path).convert("RGB")

        # change brightness if desired. 1 is to keep as original
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        return np.array(img)

    def map_pointcloud_to_image(self, idx, camera_channel):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']

        imgpc_path = self.root_dir / 'image_lidar_projections' / 'image_pc' / camera_channel / f'{sequence_name}_{sample_idx:04}.npy'
        pts_img = np.load(imgpc_path)
        fovinds_path = self.root_dir / 'image_lidar_projections' / 'fov_inds' / camera_channel / f'{sequence_name}_{sample_idx:04}.npy'
        fov_inds = np.load(fovinds_path)

        pc_lidar = self.get_pointcloud(idx)[fov_inds,:]

        imgfov = {"pc_lidar": pc_lidar,
                  "pts_img": pts_img,
                  "pc_cam": None,
                  "fov_inds": fov_inds}
        return imgfov
    
    def get_camera_instances(self, idx, channel):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']
        image_id = f'{sequence_name}_{sample_idx:04}'

        ann_ids = self.masks[channel].getAnnIds(imgIds=[image_id])
        instances = self.masks[channel].loadAnns(ann_ids)
        instances = sorted(instances, key=lambda x: x['area'], reverse=True)
        return instances

    def get_mask_instance_clouds(self, idx, camera_channel, use_bbox=True):
        image = self.get_image(idx, camera_channel=camera_channel)
        imgfov = self.map_pointcloud_to_image(idx, camera_channel=camera_channel)
        instances = self.get_camera_instances(idx, camera_channel)
        instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                    instances, 
                                                    imgfov, 
                                                    use_bbox=use_bbox)
        filtered_icloud = [x for x in instance_pts['lidar_xyzls'] if len(x) != 0]
        return filtered_icloud
    
    def render_pointcloud_in_image(self, idx, camera_channel, mask=False, use_bbox=False, min_dist=1.0, point_size=5, brightness=1):
        
        image = self.get_image(idx, camera_channel=camera_channel)
        imgfov = self.map_pointcloud_to_image(idx, camera_channel=camera_channel)

        if mask == True:
            instances = self.get_camera_instances(idx, camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov,
                                                        use_bbox=use_bbox)
            try:
                # For waymo we already concatenated the depth
                all_instance_uvd = np.vstack(instance_pts['img_uv'])
                shared_utils.draw_lidar_on_image(all_instance_uvd, image, instances=instances, clip_distance=min_dist, point_size=point_size)
            except:
                print('No points in mask; drawing whole pointcloud instead')
                shared_utils.draw_lidar_on_image(imgfov['pts_img'], image, instances=None, clip_distance=min_dist, point_size=point_size)
        else:                        
            shared_utils.draw_lidar_on_image(imgfov['pts_img'], image, instances=None, clip_distance=min_dist, point_size=point_size)
    

