import cv2
import numpy as np
import matplotlib.pyplot as plt
import datasets.shared_utils as shared_utils
import pickle
import glob
from pathlib import Path
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
import open3d as o3d
import json

# For the instance segmentation masks
class2idx = {'Pedestrian':0, 'Car':2}

class BarajaObjects:
    def __init__(self, root_dir, dataset_cfg, extra_tag):
        self.root_dir = Path(root_dir)
        self.dataset_cfg = dataset_cfg        
        self.extra_tag = extra_tag        
        self.mask_dir = self.root_dir / 'test' / 'masks' / dataset_cfg.DET2D_MODEL if dataset_cfg.DET2D_MASK else None
        self.camera_channels = dataset_cfg.CAMERA_CHANNELS if dataset_cfg.DET2D_MASK else []
        self.masks = self.__load_masks() if dataset_cfg.DET2D_MASK else None
        self.infos = self.__load_infos()
        self.classes = dataset_cfg.CLASSES
        self.dataset_name = dataset_cfg.NAME
        self.shrink_mask_percentage = dataset_cfg.SHRINK_MASK_PERCENTAGE if dataset_cfg.DET2D_MASK else 0
        self.calib = self.__load_calibration()
        print(f"BarajaObjects initialised!")

    def __len__(self):
        """
        Returns number of samples in the dataset
        """
        return len(self.infos)

    def __load_masks(self):
        print("Loading masks...")
        masks = {}
        for channel in self.camera_channels:
            mask_json = self.mask_dir / f"{channel}.json"
            masks[channel] = COCO(mask_json)
        return masks
    
    def __load_infos(self):
        with open(str(self.root_dir / 'infos' / 'baraja_infos_test.pkl'), 'rb') as pkl:
            infos = pickle.load(pkl)
        return infos
    
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
            
            infos_idx = self.find_info_idx(self.infos, frame_id)
            if infos_idx != -1:
                self.infos[infos_idx]['meshed_lidar_path'] = path            
                
        savepath = self.root_dir / f'infos_meshed_{self.extra_tag}'
        savepath.mkdir(parents=True, exist_ok=True)
        
        output = open(str(savepath / 'baraja_infos_test.pkl'), 'wb')
        pickle.dump(self.infos, output)
        
        print(f"Saved updated infos: {str(savepath / 'baraja_infos_test.pkl')}")
        
    def get_infos(self, idx):
        frame_id = f'{idx:06}'
        infos_idx = self.find_info_idx(self.infos, frame_id)
        if infos_idx != -1:
            return self.infos[infos_idx]        
        else:
            print(f"frame_id: {frame_id}, not found in infos")
            return None

    
    # Loading methods
    def get_image(self, idx, channel, brightness=1):
        img_file = self.root_dir / 'test' / f'image/front/{idx:06}.png'
        img = Image.open(img_file).convert("RGB")

        # change brightness if desired. 1 is to keep as original
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        return np.array(img)

    def get_pointcloud(self, idx):        
        lidar_file = self.root_dir / 'test' / f'pcd/{idx:06}.pcd'
        assert lidar_file.exists(), f'No lidar file found at {lidar_file}'
        pcd = o3d.io.read_point_cloud(str(lidar_file))
        return np.asarray(pcd.points)
    
    def __load_calibration(self):
        calib_file = self.root_dir / 'test' / 'calib'/ 'front.json'
        assert calib_file.exists(), f'No calib file found at {calib_file}'
        with open(calib_file) as f:
            return json.load(f)                           
        
    def get_camera_instances(self, idx, channel):
        """
        Returns all instances detected by the instance detection for the particular requested sequence
        """
        ann_ids = self.masks[channel].getAnnIds(imgIds=idx, catIds=[class2idx[c] for c in self.classes])
        instances = self.masks[channel].loadAnns(ann_ids)
        instances = sorted(instances, key=lambda x: x['area'], reverse=True) 
        return instances

    def map_pointcloud_to_image(self, points, calib, image, min_dist=1.0):
        """
        Filter lidar points, keep those in image FOV
        """
        IMG_H, IMG_W, _ = image.shape
        cameramat = np.array(calib['intrinsic']).reshape((3,3))
        camera2sensorframe = np.array(calib['extrinsic']).reshape((4,4))

        pts_3d_hom = np.hstack((points, np.ones((points.shape[0],1)))).T # (4,N)
        pts_imgframe = np.dot(camera2sensorframe[:3], pts_3d_hom) # (3,4) * (4,N) = (3,N)
        image_pts = np.dot(cameramat, pts_imgframe).T # (3,3) * (3,N)

        image_pts[:,0] /= image_pts[:,2]
        image_pts[:,1] /= image_pts[:,2]
        uv = image_pts.copy()
        fov_inds =  (uv[:,0] > 0) & (uv[:,0] < IMG_W -1) & \
                    (uv[:,1] > 0) & (uv[:,1] < IMG_H -1)        

        imgfov = {"pc_lidar": points[fov_inds,:],
                  "pc_cam": image_pts[fov_inds,:], # same as pts_img, just here to keep it consistent across datasets
                  "pts_img": np.round(uv[fov_inds,:],0).astype(int),
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
            
            points = self.get_pointcloud(idx)

            # We only limit the point cloud range when getting instance points. 
            # Once we got the instances, we concat it with the whole range pcd
            img = self.get_image(idx, channel=camera_channel)

            # Project to image
            imgfov = self.map_pointcloud_to_image(points, self.calib, img, min_dist=min_dist)        
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

    def render_pointcloud_in_image(self, idx, camera_channel='front', 
                                   mask=False, min_dist=1.0, 
                                   point_size=2, alpha=0.9, 
                                   brightness=1, shrink_percentage=0,
                                   color_scheme='jet', map_range=25):
        """
        Project LiDAR points to image and draw
        """
        points = self.get_pointcloud(idx)
        img = self.get_image(idx, channel=camera_channel, brightness=brightness)

        # Project to image
        imgfov = self.map_pointcloud_to_image(points, self.calib, img, min_dist=1.0) 


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
                shared_utils.draw_lidar_on_image(projected_points, 
                                                 img, instances=instances, 
                                                 clip_distance=min_dist, 
                                                 point_size=point_size, alpha=alpha,                                           
                                                 color_scheme=color_scheme, map_range=map_range,
                                                 shrink_percentage=shrink_percentage)
            except Exception as e:
                # Some instances don't have a mask
                print(e)
                # print('No points in mask; drawing whole pointcloud instead')
                projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, 
                                                 img, instances=None, 
                                                 clip_distance=min_dist, 
                                                 point_size=point_size, alpha=alpha,                                           
                                                 color_scheme=color_scheme, map_range=map_range,
                                                 shrink_percentage=shrink_percentage)
        else:
            projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
            shared_utils.draw_lidar_on_image(projected_points, 
                                             img, instances=None, 
                                             clip_distance=min_dist, 
                                             point_size=point_size, alpha=alpha,                                           
                                             color_scheme=color_scheme, map_range=map_range,
                                             shrink_percentage=shrink_percentage)