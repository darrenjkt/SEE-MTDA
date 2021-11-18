import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import kitti_utils
import datasets.shared_utils as shared_utils
import pickle
import glob
from pathlib import Path
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO

CAMERA_CHANNELS = ['image_2', 'image_3']
class2idx = {'Pedestrian':0, 'Car':2}

class KittiObjects:
    def __init__(self, root_dir, dataset_cfg, extra_tag):
        self.root_dir = Path(root_dir)
        self.dataset_cfg = dataset_cfg        
        self.extra_tag = extra_tag        
        self.mask_dir = self.root_dir / 'training' / 'masks' / dataset_cfg.DET2D_MODEL if dataset_cfg.DET2D_MASK else None
        self.camera_channels = dataset_cfg.CAMERA_CHANNELS if dataset_cfg.DET2D_MASK else []
        self.masks = self.__load_masks() if dataset_cfg.DET2D_MASK else None
        self.infos = self.__load_infos()
        self.classes = dataset_cfg.CLASSES
        self.dataset_name = dataset_cfg.NAME
        self.shrink_mask_percentage = dataset_cfg.SHRINK_MASK_PERCENTAGE if dataset_cfg.DET2D_MASK else 0
        print(f"KittiObjects initialised!")

    def __len__(self):
        """
        Returns number of samples in the dataset
        """
        return len(self.infos['train']) + len(self.infos['val'])

    def __load_masks(self):
        print("Loading masks...")
        masks = {}
        for channel in self.camera_channels:
            mask_json = self.mask_dir / f"{channel}.json"
            masks[channel] = COCO(mask_json)
        return masks
    
    def __load_infos(self):
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / 'kitti_infos_train.pkl'), 'rb') as trainpkl:
            train_infos = pickle.load(trainpkl)
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / 'kitti_infos_val.pkl'), 'rb') as valpkl:
            val_infos = pickle.load(valpkl)
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / 'kitti_infos_trainval.pkl'), 'rb') as tvpkl:
            trainval_infos = pickle.load(tvpkl)
        return {'train': train_infos, 'val': val_infos, 'trainval': trainval_infos}
    
    def find_info_idx(self, infos, frame_id):
        for i, dic in enumerate(infos):        
            if dic['point_cloud']['lidar_idx'] == frame_id:
                return i
        return -1
    
    def update_infos(self, save_dir):
        saved_files = glob.glob(f'{str(save_dir)}/*')
        
        frame_ids = [Path(fname).stem for fname in saved_files]
        rel_pcds = ['/'.join(fname.split('/')[-2:]) for fname in saved_files]
        id_path = dict(zip(frame_ids, rel_pcds))
        
        for frame_id, path in id_path.items():
            
            tinfos_idx = self.find_info_idx(self.infos['train'], frame_id)
            vinfos_idx = self.find_info_idx(self.infos['val'], frame_id)
            tvinfos_idx = self.find_info_idx(self.infos['trainval'], frame_id)
            if tinfos_idx != -1:
                self.infos['train'][tinfos_idx]['meshed_lidar_path'] = path
            if vinfos_idx != -1:
                self.infos['val'][vinfos_idx]['meshed_lidar_path'] = path
            if tvinfos_idx != -1:
                self.infos['trainval'][tvinfos_idx]['meshed_lidar_path'] = path
                
        savepath = self.root_dir / f'infos_meshed_{self.extra_tag}'
        savepath.mkdir(parents=True, exist_ok=True)
        
        toutput = open(str(savepath / 'kitti_infos_train.pkl'), 'wb')
        pickle.dump(self.infos['train'], toutput)
        voutput = open(str(savepath / 'kitti_infos_val.pkl'), 'wb')
        pickle.dump(self.infos['val'], voutput)
        tvoutput = open(str(savepath / 'kitti_infos_trainval.pkl'), 'wb')
        pickle.dump(self.infos['trainval'], tvoutput)
        
        print(f"Saved updated train infos: {str(savepath / 'kitti_infos_train.pkl')}")
        print(f"Saved updated val infos: {str(savepath / 'kitti_infos_val.pkl')}")
        print(f"Saved updated trainval infos: {str(savepath / 'kitti_infos_trainval.pkl')}")
    
    def get_infos(self, idx):
        frame_id = f'{idx:06}'
        tinfos_idx = self.find_info_idx(self.infos['train'], frame_id)
        vinfos_idx = self.find_info_idx(self.infos['val'], frame_id)
        if tinfos_idx != -1:
            return self.infos['train'][tinfos_idx]
        elif vinfos_idx != -1:
            return self.infos['val'][vinfos_idx]
        else:
            print("frame_id not found in infos")
            return None
    
    # Loading methods
    def get_image(self, idx, channel, brightness=1):
        img_file = self.root_dir / 'training' / f'{channel}/{idx:06}.png'
        img = Image.open(img_file).convert("RGB")

        # change brightness if desired. 1 is to keep as original
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        return np.array(img)

    def get_pointcloud(self, idx):        
        lidar_file = self.root_dir / 'training' / f'velodyne/{idx:06}.bin'
        return np.fromfile(lidar_file, dtype=np.float32).reshape((-1,4))[:,:3]

    def get_calibration(self, idx):
        calib_file = self.root_dir / 'training' / f'calib/{idx:06}.txt'
        return kitti_utils.Calibration(calib_file)
        
    def get_camera_instances(self, idx, channel):
        """
        Returns all instances detected by the instance detection for the particular requested sequence
        """
        ann_ids = self.masks[channel].getAnnIds(imgIds=idx, catIds=[class2idx[c] for c in self.classes])
        instances = self.masks[channel].loadAnns(ann_ids)
        instances = sorted(instances, key=lambda x: x['area'], reverse=True) 
        return instances

    def map_pointcloud_to_image(self, pc_velo, calib, image,min_dist=1.0):
        ''' Filter lidar points, keep those in image FOV '''
        pts_2d = calib.project_velo_to_imageuv(pc_velo)
        
        # We keep the indices to associate any processing in image domain with the original lidar points
        # min_dist is to keep any lidar points further than that minimum distance
        IMG_H, IMG_W, _ = image.shape
        fov_inds = (pts_2d[:,0]<IMG_W) & (pts_2d[:,0]>=0) & \
            (pts_2d[:,1]<IMG_H) & (pts_2d[:,1]>=0)
        fov_inds = fov_inds & (pc_velo[:,0]>min_dist)
        imgfov_pc_velo = pc_velo[fov_inds,:]
        imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

        imgfov = {"pc_lidar": imgfov_pc_velo,
                  "pc_cam": imgfov_pc_rect,
                  "pts_img": np.floor(pts_2d[fov_inds,:]).astype(int) ,
                  "fov_inds": fov_inds }
        return imgfov
        

    def get_mask_instance_clouds(self, idx, min_dist=1.0, shrink_percentage=None):
        """
        Returns the individual clouds for each mask instance. 
        
        Return: list of (N,4) np arrays (XYZL), each corresponding to one object instance (XYZ points) with a label (L)
        """
        if type(self.camera_channels) is not list:
            self.camera_channels = [self.camera_channels]

        if shrink_percentage == None:
            shrink_percentage = self.shrink_mask_percentage

        i_clouds = []
        for camera_channel in self.camera_channels:
            
            pc_velo = self.get_pointcloud(idx)

            # We only limit the point cloud range when getting instance points. 
            # Once we got the instances, we concat it with the whole range pcd
            mask = shared_utils.mask_points_by_range(pc_velo, self.dataset_cfg.POINT_CLOUD_RANGE)
            pc_velo = pc_velo[mask]
            img = self.get_image(idx, channel=camera_channel)
            calib = self.get_calibration(idx)

            # Project to image
            imgfov = self.map_pointcloud_to_image(pc_velo[:,:3], calib, img, min_dist=min_dist)        
            instances = self.get_camera_instances(idx, channel=camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov['pts_img'], 
                                                        imgfov['pc_lidar'], 
                                                        imgfov['pc_cam'],
                                                        shrink_percentage=shrink_percentage)

            filtered_icloud = [x for x in instance_pts['lidar_xyzls'] if len(x) != 0]
            i_clouds.extend(filtered_icloud)
        
        return i_clouds   

    def render_pointcloud_in_image(self, idx, camera_channel, mask=False, min_dist=1.0, point_size=2, brightness=1, shrink_percentage=0):
        """
        Project LiDAR points to image and draw
        """
        pc_velo = self.get_pointcloud(idx)
        img = self.get_image(idx, channel=camera_channel, brightness=brightness)
        calib = self.get_calibration(idx)

        # Project to image
        imgfov = self.map_pointcloud_to_image(pc_velo[:,:3], calib, img, min_dist=1.0) 

        if mask == True:
            instances = self.get_camera_instances(idx, channel=camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov['pts_img'], 
                                                        imgfov['pc_lidar'], 
                                                        imgfov['pc_cam'],
                                                        shrink_percentage=shrink_percentage)
            
            try:
                all_instance_uv = np.vstack(instance_pts['img_uv'])
                all_instance_cam = np.vstack(instance_pts['cam_xyz'])
                projected_points = np.hstack((all_instance_uv[:,:2], all_instance_cam[:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, img, instances=instances, clip_distance=min_dist, point_size=point_size, shrink_percentage=shrink_percentage)
            except Exception as e:
                # Some instances don't have a mask
                print(e)
                # print('No points in mask; drawing whole pointcloud instead')
                projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, img, instances=None, clip_distance=min_dist, point_size=point_size)
        else:
            projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
            shared_utils.draw_lidar_on_image(projected_points, img, instances=None, clip_distance=min_dist, point_size=point_size)