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
    # Note: TFRecords is a streaming reader and has no random access.
    # I.e. we can't index it but rather are forced to iterate through it
    # If GT-MESH, we'll use the processed_waymo_data npy pointcloud files
    # If DET-MESH, we'll iterate once through the tfrecords, save all images
    # then use the saved images with npy point cloud to get masks
    
    def __init__(self, root_dir, dataset_cfg, extra_tag):
        
        self.root_dir = Path(root_dir)
        self.dataset_cfg = dataset_cfg        
        self.extra_tag = extra_tag       
        self.mask_dir = self.root_dir / 'masks' / dataset_cfg.DET2D_MODEL if dataset_cfg.DET2D_MASK else None
        self.camera_channels = dataset_cfg.CAMERA_CHANNELS if dataset_cfg.DET2D_MASK else []
        self.masks = self.__load_masks() if dataset_cfg.DET2D_MASK else None
        self.idx2tokens = self.__load_tokens()
        self.record_files = glob.glob(str(self.root_dir) + "raw_data/*.tfrecord")
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

    def get_pointcloud(self, idx):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']
        path = self.root_dir / 'waymo_processed_data' / sequence_name /f'{sample_idx:04}.npy'
        point_features = np.load(path)
        points_all, NLZ_flag = point_features[:,0:5], point_features[:, 5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:,3])
        return points_all[:,:3]

    def get_record_and_frame(self, idx, channel):
        # This is not ideal; TFRecord takes a long time to iterate through
        # We should not need to index the TFRecord for the processing. This 
        # is more for testing purposes
        record_frame_num = self.idx2tokens[channel][idx].split('+')
        record_fname = self.root_dir / f'segment-{record_frame_num[0]}_with_camera_labels.tfrecord'
        
        try:
            record = tf.data.TFRecordDataset(str(record_fname), compression_type='')
        except:
            print('Unable to retrieve record, possibly due to lack of GPU memory')
            return None
        
        # This iteration takes a long time
        for f_idx, data in enumerate(record):
            if f_idx == idx:     
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                return record, frame        
        return None
    
    def map_pointcloud_to_image(self, frame, camera_channel):
        """
        Project lidar to image frame and keep only those in image FOV. 
        This was taken from Waymo tutorial directly
        """
        points, cp_points = self.get_pointcloud_from_frame(frame)
        camera_idx = CAMERA_CHANNELS.index(camera_channel)
        image_record = frame.images[camera_idx]
        cp_points_all_concat = np.concatenate([cp_points, points], axis=-1)
        cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

        # The distance between lidar points and vehicle frame origin.
        points_all_tensor = tf.norm(points, axis=-1, keepdims=True)
        cp_points_all_tensor = tf.constant(cp_points, dtype=tf.int32)

        mask = tf.equal(cp_points_all_tensor[..., 0], image_record.name)

        cp_points_all_tensor = tf.cast(tf.gather_nd(
            cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))
        pc_lidar = tf.cast(tf.gather_nd(points, tf.where(mask)), dtype=tf.float32)
        projected_points_all_from_raw_data = tf.concat(
            [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
        
        imgfov = {"pc_lidar": np.asarray(pc_lidar),
                  "pts_img": np.floor(projected_points_all_from_raw_data).astype(int),
                  "pc_cam": None,
                  "fov_inds": mask}
        return imgfov
    
    def get_mask_instance_clouds(self, frame, camera_channel):
        image = self.get_image_from_frame(frame, channel=camera_channel)
        imgfov = self.map_pointcloud_to_image(frame, camera_channel=camera_channel)
        instances = self.get_camera_instances_from_frame(frame, camera_channel)
        instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], instances, imgfov)
        filtered_icloud = [x for x in instance_pts['lidar_xyzls'] if len(x) != 0]
        return filtered_icloud
    
    def render_pointcloud_in_image(self, frame, camera_channel, mask=False, min_dist=1.0, point_size=5, brightness=1):
        
        image = self.get_image_from_frame(frame, channel=camera_channel)
        imgfov = self.map_pointcloud_to_image(frame, camera_channel=camera_channel)

        if mask == True:
            instances = self.get_camera_instances_from_frame(frame, camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], instances, imgfov['pts_img'], imgfov['pc_lidar'], None)
            try:
                # For waymo we already concatenated the depth
                all_instance_uvd = np.vstack(instance_pts['img_uv'])
                shared_utils.draw_lidar_on_image(all_instance_uvd, image, instances=instances, clip_distance=min_dist, point_size=point_size)
            except:
                print('No points in mask; drawing whole pointcloud instead')
                shared_utils.draw_lidar_on_image(imgfov['pts_img'], image, instances=None, clip_distance=min_dist, point_size=point_size)
        else:                        
            shared_utils.draw_lidar_on_image(imgfov['pts_img'], image, instances=None, clip_distance=min_dist, point_size=point_size)
    

    def get_camera_instances_from_frame(self, frame, channel):
        image_token = f'{frame.context.name}+{channel}+{frame.timestamp_micros}'
        ann_ids = self.masks[channel].getAnnIds(imgIds=[image_token])
        instances = self.masks[channel].loadAnns(ann_ids)
        instances = sorted(instances, key=lambda x: x['area'], reverse=True)
        return instances
    
    def get_image_from_frame(self, frame, channel, brightness=1):
        camera_images = {idx2camera[v.name]:v for k,v in enumerate(frame.images)}
        image = np.asarray(tf.image.decode_jpeg(camera_images[channel].image, dct_method="INTEGER_ACCURATE"))
        
        if brightness != 1:
            pil_image = Image.fromarray(np.uint8(image))
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
            return np.array(pil_image)
        else:
            return image
        
    def get_pointcloud_from_frame(self, frame):
        # Pointclouds are stored as range images so we need to convert
        (range_images, camera_projections,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud( frame,
                                                                            range_images,
                                                                            camera_projections,
                                                                            range_image_top_pose)
        # Join all lidars into one point cloud
        return np.concatenate(points, axis=0), np.concatenate(cp_points, axis=0)