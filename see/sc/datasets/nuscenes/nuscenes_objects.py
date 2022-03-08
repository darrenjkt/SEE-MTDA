import os
import time
import numpy as np
import pickle
import glob
from pathlib import Path
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import sc.datasets.shared_utils as shared_utils

CAMERA_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
class2idx = {'pedestrian':0, 'car':2}

class NuscenesObjects:
    def __init__(self, root_dir, dataset_cfg, extra_tag):
        
        self.root_dir = Path(root_dir)
        self.dataset_cfg = dataset_cfg        
        self.extra_tag = extra_tag       
        self.custom_dataset = dataset_cfg.CUSTOM
        self.version = 'v1.0-trainval' if self.custom_dataset else os.path.basename(self.root_dir)
        self.nusc = NuScenes(version=self.version, dataroot=self.root_dir, verbose=True)
        self.mask_dir = self.root_dir / 'masks' / dataset_cfg.DET2D_MODEL if dataset_cfg.DET2D_MASK else None
        self.camera_channels = dataset_cfg.CAMERA_CHANNELS if dataset_cfg.DET2D_MASK else []
        self.masks = self.__load_masks() if dataset_cfg.DET2D_MASK else None
        self.sample_records = self.__get_sample_records()
        self.infos = self.__load_infos()
        self.classes = dataset_cfg.CLASSES
        self.nsweeps = dataset_cfg.LIDAR_NSWEEPS
        self.dataset_name = dataset_cfg.NAME
        self.ground_zplane=-1.8 # For extending convex hull to ground
                
        print(f"NuScenesObjects initialised!")
    
    def __len__(self):
        """
        Returns number of samples in the dataset. This is the number of
        all scenes joined together
        """
        return len(self.sample_records)
    
    def __load_masks(self):
        print("Loading masks...")
        masks = {}
        for channel in self.camera_channels:
            mask_json = self.mask_dir / f"{channel}.json"
            masks[channel] = COCO(mask_json)
        return masks

    def __get_sample_records(self):
        """
        Grab all the sample records.
        Commented out the val split cause we can just test on the same train split.
        Since we train on another dataset like waymo/kitti
        """
        print("Loading sample records...")
        if self.custom_dataset:
            ftrain = open(str(self.root_dir / 'custom_train_split.txt'), 'r').read()
            # fval = open(str(self.root_dir / 'custom_val_split.txt'), 'r').read()
            scene_names = ftrain.split('\n')
            # scene_names.extend(fval.split('\n'))
            self.scenes = [nusc_scene for nusc_scene in self.nusc.scene if nusc_scene['name'] in scene_names]
        else: 
            self.scenes = self.nusc.scene
        
        frame_num = 0
        sample_records = {}
        for scene in self.scenes:
            current_sample_token = scene['first_sample_token']
            while(current_sample_token != ''):
                current_sample = self.nusc.get('sample', current_sample_token)
                sample_records[frame_num] = current_sample
                current_sample_token = current_sample['next']
                frame_num += 1
        
        return sample_records
    
    def __load_infos(self):
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / 'nuscenes_infos_10sweeps_train.pkl'), 'rb') as trainpkl:
            train_infos = pickle.load(trainpkl)
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / 'nuscenes_infos_10sweeps_val.pkl'), 'rb') as valpkl:
            val_infos = pickle.load(valpkl)
        return {'train': train_infos, 'val': val_infos}
    
    def find_info_idx(self, infos, token):
        for i, dic in enumerate(infos):        
            if dic['token'] == token:
                return i
        return -1
    
    def get_infos(self, idx):
        sample_record = self.sample_records[idx]
        sample_token = sample_record['token']
        tinfos_idx = self.find_info_idx(self.infos['train'], sample_token)
        vinfos_idx = self.find_info_idx(self.infos['val'], sample_token)
        if tinfos_idx != -1:
            return self.infos['train'][tinfos_idx]
        elif vinfos_idx != -1:
            return self.infos['val'][vinfos_idx]
        else:
            print("Sample token not found in infos")
            return None
    
    def update_infos(self, save_meshed_dir, nsweeps=0):
        print('Updating pre-generated infos with meshed paths...')
        if nsweeps == 0:
            nsweeps = self.nsweeps
            
        saved_files = glob.glob(f'{str(save_meshed_dir)}/*')
        
        s_tok = [pcd.split('#')[-2].split('/')[-1] for pcd in saved_files]
        rel_pcds = ['/'.join(pcd.split('/')[-3:]) for pcd in saved_files]
        tok_path = dict(zip(s_tok, rel_pcds))
        
        for sample_token, path in tok_path.items():
            
            tinfos_idx = self.find_info_idx(self.infos['train'], sample_token)
            vinfos_idx = self.find_info_idx(self.infos['val'], sample_token)

            if tinfos_idx != -1:
                self.infos['train'][tinfos_idx]['meshed_lidar_path'] = path
            elif vinfos_idx != -1:
                self.infos['val'][vinfos_idx]['meshed_lidar_path'] = path
            else:
                print(f"Sample token {sample_token} doesn't exist in infos, double check your code")
                print(f"rel_pcd path is {path}")
                
        savepath = self.root_dir / f'infos_meshed_{self.extra_tag}'
        savepath.mkdir(parents=True, exist_ok=True)
        toutput = open(str(savepath / f'nuscenes_infos_{nsweeps}sweeps_train.pkl'), 'wb')
        pickle.dump(self.infos['train'], toutput)

        voutput = open(str(savepath / f'nuscenes_infos_{nsweeps}sweeps_val.pkl'), 'wb')
        pickle.dump(self.infos['val'], voutput)
        print(f"Saved updated train infos: {str(savepath / f'nuscenes_infos_{nsweeps}sweeps_train.pkl')}")
        print(f"Saved updated val infos: {str(savepath / f'nuscenes_infos_{nsweeps}sweeps_val.pkl')}")
        print(f'Complete: {len(saved_files)} processed')        

    def update_gt_database(self, save_gt_dir, nsweeps=0):
        print('Updating database infos for meshed objects...')
        if nsweeps == 0:
            nsweeps = self.nsweeps
            
        objects = glob.glob(f'{str(save_gt_dir)}/*.pcd')
        db_dict = {}
        db_dict['car'] = []

        for obj_path in objects:

            db_obj_dict = {}
            db_obj_dict['name'] = obj_path.split("_oid")[0].split('_')[-1]
            db_obj_dict['path'] = '/'.join(obj_path.split('/')[-3:])
            db_obj_dict['image_idx'] = int(obj_path.split("#")[-1].split("_")[0])
            db_obj_dict['gt_idx'] = int(obj_path.split("_oid")[-1].split('_')[0])
            db_obj_dict['box3d_lidar'] = np.array(obj_path.split('_bbox')[-1].split('.pcd')[0].split('%'), dtype=np.float64)
            db_obj_dict['num_points_in_gt'] = int(obj_path.split("_p")[-1].split('_')[0])
            db_dict['car'].append(db_obj_dict)

        savepath = self.root_dir / f'infos_meshed_{self.extra_tag}'
        savepath.mkdir(parents=True, exist_ok=True)
        db_dict_output = open(str(savepath / f'nuscenes_dbinfos_{nsweeps}sweeps_withvelo.pkl'), 'wb')
        pickle.dump(db_dict, db_dict_output)
        print(f"Saved gt database infos: {str(savepath / f'nuscenes_dbinfos_{nsweeps}sweeps_withvelo.pkl')}")
        print(f'Complete: {len(objects)} objects in database') 

    def get_image(self, idx, channel, brightness=1, return_token=False):
        sample_record = self.sample_records[idx]
        img_token = sample_record['data'][channel]
        img_file = self.nusc.get_sample_data_path(img_token)
        img = Image.open(img_file).convert("RGB")
        
        # change brightness if desired. 1 is to keep as original
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        if return_token:
            return np.array(img), img_token
        else:
            return np.array(img)
    
    def get_pointcloud(self, idx, nsweeps=0, return_token=False, as_nparray=True):
        if nsweeps == 0:
            nsweeps = self.nsweeps
        
        sample_record = self.sample_records[idx]
        pc_sweeps, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample_record, chan='LIDAR_TOP', ref_chan='LIDAR_TOP', nsweeps=nsweeps)
        if as_nparray:
            pc_sweeps = pc_sweeps.points.T[:,:3]
        
        if return_token:
            lidar_token = sample_record['data']['LIDAR_TOP']
            return pc_sweeps, lidar_token
        else:
            return pc_sweeps
    
    def get_lidar_token_from_idx(self, idx):
        sample_record = self.sample_records[idx]
        return sample_record['data']['LIDAR_TOP']
    
    def get_camera_token_from_idx(self, idx, channel):
        sample_record = self.sample_records[idx]
        return sample_record['data'][channel]
    
    def get_sample_token_from_idx(self, idx):
        sample_record = self.sample_records[idx]
        return sample_record['token']
    
    def get_camera_instances(self, idx, channel):
        sample_record = self.sample_records[idx]

        # v1.0-mini mask format
        # imgfile = self.nusc.get('sample_data', sample_record['data'][channel])['filename']
        # image_id = os.path.basename(imgfile).split('.')[0]
        
        # t4025 custom mask format
        image_id = sample_record['data'][channel] 
        
        ann_ids = self.masks[channel].getAnnIds(imgIds=[image_id], catIds=[class2idx[c] for c in self.classes])
        instances = self.masks[channel].loadAnns(ann_ids)
        instances = sorted(instances, key=lambda x: x['area'], reverse=True)
        return instances

    def map_pointcloud_to_image(self, idx, camera_channel, nsweeps=1, min_dist=1.0):
        """
        Code mostly adapted from the original nuscenes-devkit:
        https://github.com/nutonomy/nuscenes-devkit/blob/cbca1b882aa4fbaf8714ea0d0897457e60a5caae/python-sdk/nuscenes/nuscenes.py#L834

        The difference is that the outputs are modified to fit within
        the existing code structure.
        """

        img, camera_token = self.get_image(idx, camera_channel, return_token=True)
        pc, lidar_token = self.get_pointcloud(idx, nsweeps=nsweeps, return_token=True, as_nparray=False)
        pc_lidar = np.array(pc.points, copy=True)

        cam = self.nusc.get('sample_data' , camera_token)
        pointsensor = self.nusc.get('sample_data', lidar_token)

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: project points from camera frame to image using camera intrinsics
        pc_cam = pc.points[:3, :]
        depths = pc.points[2, :]
        pts_2d = view_points(pc_cam, np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        fov_inds = np.ones(depths.shape[0], dtype=bool)
        fov_inds = np.logical_and(fov_inds, depths > min_dist)
        fov_inds = np.logical_and(fov_inds, pts_2d[0, :] > 0)
        fov_inds = np.logical_and(fov_inds, pts_2d[0, :] < img.shape[1])
        fov_inds = np.logical_and(fov_inds, pts_2d[1, :] > 0)
        fov_inds = np.logical_and(fov_inds, pts_2d[1, :] < img.shape[0])

        # Restrict points to the camera's fov        
        imgfov = {"pc_lidar": pc_lidar[:3, fov_inds].T,
                  "pc_cam": pc_cam[:3, fov_inds].T,
                  "pts_img": np.floor(pts_2d[:2, fov_inds]).astype(int).T,
                  "fov_inds": fov_inds }
        return imgfov
    
    def get_mask_instance_clouds(self, idx, camera_channels=None, min_dist=1.0, use_bbox=False):
        """
        Returns the pointclouds of the individual objects, bounded by the 
        segmentation mask
        """
        start = time.time()        
        
        if camera_channels is None:
            camera_channels = self.camera_channels
            print(f'Cameras not specified. Using self.camera_channels = {self.camera_channels}')

        if type(camera_channels) is not list:
            camera_channels = [camera_channels]
        else:
            print(f'Returning instances for all {len(camera_channels)} cameras')

        i_clouds = []
        for camera_channel in camera_channels:
            tic = time.time()
            camera_token = self.get_camera_token_from_idx(idx, channel=camera_channel)
            lidar_token = self.get_lidar_token_from_idx(idx)
            img = self.get_image(idx, channel=camera_channel)

            imgfov = self.map_pointcloud_to_image(idx, camera_channel, self.nsweeps, min_dist=min_dist)

            instances = self.get_camera_instances(idx, channel=camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov,
                                                        use_bbox=use_bbox)

            filtered_icloud = [x for x in instance_pts['lidar_xyzls'] if len(x) != 0]
            i_clouds.extend(filtered_icloud)

        return i_clouds
        
    
    
    def render_pointcloud_in_image(self, idx, camera_channel, mask=False, nsweeps=1, min_dist=1.0, point_size=15, brightness=1):
        camera_token = self.get_camera_token_from_idx(idx, channel=camera_channel)
        lidar_token = self.get_lidar_token_from_idx(idx)
        img = self.get_image(idx, channel=camera_channel, brightness=brightness)
        
        imgfov = self.map_pointcloud_to_image(idx, camera_channel, nsweeps=nsweeps, min_dist=min_dist)

        if mask == True:
            instances = self.get_camera_instances(idx, camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], instances, imgfov['pts_img'], imgfov['pc_lidar'], imgfov['pc_cam'])
            
            try:
                all_instance_uv = np.vstack(instance_pts['img_uv'])
                all_instance_cam = np.vstack(instance_pts['cam_xyz'])
                projected_points = np.hstack((all_instance_uv[:,:2], all_instance_cam[:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, img, instances=instances, clip_distance=min_dist, point_size=point_size)
            except:
                print('No mask; drawing whole pointcloud instead')
                projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, img, instances=None, clip_distance=min_dist, point_size=point_size)
        else:         
            projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
            shared_utils.draw_lidar_on_image(projected_points, img, instances=None, clip_distance=min_dist, point_size=point_size)
            
