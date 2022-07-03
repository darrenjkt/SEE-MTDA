import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import kitti_utils
import sc.datasets.shared_utils as shared_utils
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
        self.shrink_mask_percentage = dataset_cfg.get('SHRINK_MASK_PERCENTAGE', 0)
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

    def get_pointcloud(self, idx, append_labels=False, add_ground_lift=False):        
        lidar_file = self.root_dir / 'training' / f'velodyne/{idx:06}.bin'
        xyz_pts = np.fromfile(lidar_file, dtype=np.float32).reshape((-1,4))[:,:3]

        if append_labels:
            # Get labels
            sample_infos = self.get_infos(idx)
            pcd_gtboxes = shared_utils.populate_gtboxes(sample_infos, "kitti", self.classes, add_ground_lift=add_ground_lift)

            ## (X,Y,Z,OBJ_ID,CLASS)
            # Obj ID is [0,N] for objs, and -1 for stuff
            # Class is 1 for car, 0 for others
            o3dpcd = shared_utils.convert_to_o3dpcd(xyz_pts)
            obj_pcds = [o3dpcd.crop(gtbox) for gtbox in pcd_gtboxes['gt_boxes']]
            obj_pts = np.concatenate([np.asarray(obj.points) for obj in obj_pcds])
            dists = [o3dpcd.compute_point_cloud_distance(obj) for obj in obj_pcds]
            cropped_inds = np.concatenate([np.where(np.asarray(d) < 0.01)[0] for d in dists])
            pcd_without_objects = np.asarray(o3dpcd.select_by_index(cropped_inds, invert=True).points)

            
            obj_np = [np.array(obj.points) for obj in obj_pcds]
            obj_np_ids = np.vstack([np.hstack([obj, np.ones((len(obj),1))*(o_id+1)]) for o_id, obj in enumerate(obj_np)])
            obj_np_ids_carlabel = np.hstack([obj_np_ids, np.ones((len(obj_np_ids), 1))])
            pcd_without_objects_id = np.hstack([pcd_without_objects, -1*np.ones((len(pcd_without_objects), 1))])
            pcd_without_objects_label = np.hstack([pcd_without_objects_id, np.zeros((len(pcd_without_objects), 1))])
            labelled_pcd = np.vstack([obj_np_ids_carlabel, pcd_without_objects_label])

            return labelled_pcd
        else:
            return xyz_pts


    def get_calibration(self, idx):
        calib_file = self.root_dir / 'training' / f'calib/{idx:06}.txt'
        return kitti_utils.Calibration(calib_file)
        
    def get_camera_instances(self, idx, channel):
        """
        Returns all instances detected by the instance detection for the particular requested sequence
        """
        ann_ids = self.masks[channel].getAnnIds(imgIds=[f'{idx:06d}'], catIds=[class2idx[c] for c in self.classes])
        instances = self.masks[channel].loadAnns(ann_ids)
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
        

    def get_mask_instance_clouds(self, idx, append_labels=False, min_dist=1.0, shrink_percentage=None, use_bbox=False):
        """
        Returns the individual clouds for each mask instance. 

        :append_labels: Returns a instance pointcloud with track ids and class labels for each car
        :use_bbox: Instead of getting points within the mask, it'll get points within the bbox
        
        Return: list of (N,4) np arrays (XYZL), each corresponding to one object instance (XYZ points) with a label (L)
        """
        if type(self.camera_channels) is not list:
            self.camera_channels = [self.camera_channels]

        if shrink_percentage == None:
            shrink_percentage = self.shrink_mask_percentage

        i_clouds = []
        for camera_channel in self.camera_channels:
            
            pc_velo = self.get_pointcloud(idx, append_labels=append_labels)

            # We only limit the point cloud range when getting instance points. 
            # Once we got the instances, we concat it with the whole range pcd
            mask = shared_utils.mask_points_by_range(pc_velo, self.dataset_cfg.POINT_CLOUD_RANGE)
            pc_velo = pc_velo[mask]
            img = self.get_image(idx, channel=camera_channel)
            calib = self.get_calibration(idx)

            # Project to image
            imgfov = self.map_pointcloud_to_image(pc_velo[:,:3], calib, img, min_dist=min_dist)        
            if append_labels:
                imgfov['pc_labelled'] = pc_velo[imgfov['fov_inds'],:]

            imgfov['img_shape'] = img.shape[:2] # H, W
            instances = self.get_camera_instances(idx, channel=camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov,
                                                        shrink_percentage=shrink_percentage, 
                                                        use_bbox=use_bbox,
                                                        labelled_pcd=append_labels)

            if append_labels:
                filtered_icloud = [x for x in instance_pts['labelled_pcd'] if len(x) != 0]    
            else:
                filtered_icloud = [x for x in instance_pts['pointcloud'] if len(x) != 0]
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