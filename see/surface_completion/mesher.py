import open3d as o3d
import numpy as np
import time
import os
from datasets.shared_utils import convert_to_o3dpcd, get_o3dbox
import pprint
from pathlib import Path
import glob

from datasets.nuscenes.nuscenes_objects import NuscenesObjects
from datasets.waymo.waymo_objects import WaymoObjects
from datasets.kitti.kitti_objects import KittiObjects
from datasets.baraja.baraja_objects import BarajaObjects
from mesher_methods import db_scan, ball_pivoting, poisson_surface_reconstruction, alpha_shapes
from mesher_methods import vres_ring_based_sampling, surface_area_based_sampling, virtual_lidar_sampling

__DATASETS__ = {
    'nuscenes': NuscenesObjects,
    'kitti': KittiObjects,
    'waymo': WaymoObjects,
    'baraja': BarajaObjects
}

__CLUSTERING__ = {
    'db_scan': db_scan
}

__MESH_ALGORITHMS__ = {
    'ball_pivoting': ball_pivoting,
    'alpha_shapes': alpha_shapes,
    'poisson_surface_reconstruction': poisson_surface_reconstruction
}

__SAMPLING__ = {
    'vres_ring_based_sampling': vres_ring_based_sampling,
    'surface_area_based_sampling': surface_area_based_sampling,
    'virtual_lidar_sampling': virtual_lidar_sampling
}

class Mesher:
    def __init__(self, cfg, cfg_path):
        
        self.data_obj = __DATASETS__[cfg.DATASET.NAME]( root_dir=cfg.DATA_PATH, 
                                                        dataset_cfg=cfg.DATASET,
                                                        extra_tag=Path(cfg_path).resolve().stem)
        self.mesher_cfg = cfg.MESHER
        self.min_lidar_pts_to_mesh = cfg.MESHER.MIN_LIDAR_PTS_TO_MESH
        self.classes = self.data_obj.classes
        self.target_phi_theta = self.load_target_sampling_patterns(cfg_path, cfg.MESHER.SAMPLING.REFERENCE_PATTERN_FOLDER) if cfg.MESHER.SAMPLING.NAME == 'virtual_lidar_sampling' else None

        if self.data_obj.dataset_name == 'nuscenes':
            self.save_dir = self.data_obj.root_dir / 'samples' / f'MESHED_LIDAR_TOP_{self.data_obj.nsweeps}SWEEPS_{Path(cfg_path).stem}'
            self.save_gt_db_dir = self.data_obj.root_dir / f'infos_meshed_{self.data_obj.extra_tag}' / f'gt_database_{self.data_obj.nsweeps}sweeps_withvelo'
        elif self.data_obj.dataset_name == 'kitti':
            self.save_dir = self.data_obj.root_dir / 'training' / f'meshed_lidar_{Path(cfg_path).stem}'
        elif self.data_obj.dataset_name == 'waymo':
            self.save_dir = self.data_obj.root_dir / f'waymo_meshed_{Path(cfg_path).stem}'
        elif self.data_obj.dataset_name == 'baraja':
            self.save_dir = self.data_obj.root_dir / 'test' / f'meshed_{Path(cfg_path).stem}'
        else:
            print(f"mesher.py [__init__]: {self.data_obj.dataset_name} not supported")
            exit()

        print(f"\nMesher Initialised!")
        
    def load_target_sampling_patterns(self, cfg_path, folder_name):
        ref_pattern_paths = sorted(glob.glob(f'{str(Path(folder_name))}/*.npy'))
        ref_patterns = [np.load(path) for path in ref_pattern_paths]
        assert len(ref_patterns) > 0, f'No reference patterns found at {folder_name}'
        print(f"Loaded {len(ref_patterns)} reference patterns from {folder_name}")
        return ref_patterns

    def get_pcd_gtboxes(self, idx, add_ground_lift=True):
        """
        Get the pcd and boxes for a single frame. 
        add_ground_lift: adds 20cm to box centroid to lift it off the ground to avoid getting ground points
        """
        sample_infos = self.data_obj.get_infos(idx)
        o3dpcd = convert_to_o3dpcd(self.data_obj.get_pointcloud(idx))            
        
        if self.data_obj.dataset_name == 'nuscenes':
            zip_infos = zip(sample_infos['gt_boxes'], sample_infos['gt_names'], sample_infos['num_lidar_pts'])
        elif self.data_obj.dataset_name in ['kitti', 'waymo','baraja']:
            anno = sample_infos['annos']
            annos_name = [name for name in anno['name']]
            zip_infos = zip(anno['gt_boxes_lidar'], annos_name, anno['num_points_in_gt'])
        else: 
            print(f"{self.data_obj.dataset_name} is an unsupported dataset")
            return None
            
        pcd_gtboxes = {}        
        pcd_gtboxes['gt_boxes'], pcd_gtboxes['num_lidar_pts'], pcd_gtboxes['xyzlwhry_gt_boxes'] = [], [], []
        for gt_anno in zip_infos:
            bbox_corners, num_pts, xyzlwhry_bbox = get_o3dbox(gt_anno, classes=self.classes) 
            if bbox_corners != None:
                if add_ground_lift:
                    bbox_corners.center = bbox_corners.center + [0,0,0.2] # Add 20cm to box centroid z-axis to get rid of the ground plane
                pcd_gtboxes['gt_boxes'].append(bbox_corners)
                pcd_gtboxes['num_lidar_pts'].append(num_pts)
                pcd_gtboxes['xyzlwhry_gt_boxes'].append(xyzlwhry_bbox)

        pcd_gtboxes['pcd'] = o3dpcd            

        return pcd_gtboxes

    def mesh_gt_pts(self, pcd_gtboxes, sample_idx):
        """
        Given the ground truth boxes and their pcds, extract the points in each gt bbox and
        use ball pivoting to mesh it. 
        
        avg_car_len: average car length of the dataset. We use this len/4 as upper radius for ball pivot
        """
        list_meshed_instances = []
        for object_id in range(len(pcd_gtboxes['gt_boxes'])):

            # Crop pcd to get pts in gt box
            gt_box = pcd_gtboxes['gt_boxes'][object_id]
            obj_pcd = pcd_gtboxes['pcd']
            cropped_pcd = obj_pcd.crop(gt_box)
            if len(cropped_pcd.points) < self.min_lidar_pts_to_mesh:
                continue

            try: 
                mesh = __MESH_ALGORITHMS__[self.mesher_cfg.MESH_ALGORITHM.NAME](self.mesher_cfg, cropped_pcd)

                if not mesh.is_empty():
                    if self.data_obj.dataset_name == 'nuscenes':
                        # Because nuscenes uses sweeps, more sweeps doesn't mean more meaningful pts. Upsampling will be way too high if based on num pts of sweep
                        num_pcd_pts = pcd_gtboxes["num_lidar_pts"][object_id]
                    else: 
                        num_pcd_pts = len(cropped_pcd.points)

                    if self.mesher_cfg.SAMPLING.get('REFERENCE_PATTERN_FOLDER', False):
                        target_phi_theta = self.target_phi_theta[sample_idx % len(self.target_phi_theta)]
                    else:
                        target_phi_theta = None

                    # TODO: refactor this to use kwargs instead of empty placeholders in the functions
                    sampled_pts = __SAMPLING__[self.mesher_cfg.SAMPLING.NAME](self.mesher_cfg, mesh, num_pcd_pts=num_pcd_pts, gt_box=gt_box, reference_phi_theta=target_phi_theta)
                    sampled_pts = sampled_pts.crop(gt_box)
                    if sampled_pts.is_empty():
                        continue
                    list_meshed_instances.append(sampled_pts)
            except Exception as e:
                print(f'mesher.py [mesh_gt_pts]: sample idx: {sample_idx} - object idx: {object_id} - Error: {e}')
                continue

            if self.data_obj.dataset_cfg.get('UPDATE_GT_DATABASE', False):
                self.save_to_gt_database(sampled_pts, sample_idx, object_id, pcd_gtboxes['xyzlwhry_gt_boxes'][object_id])                                    

        return list_meshed_instances
    
    def mesh_det_pts(self, i_cloud, sample_idx):
        list_meshed_instances = []
        i_cloud = [pcd for pcd in i_cloud if len(pcd) >= self.min_lidar_pts_to_mesh]

        for object_id, instance in enumerate(i_cloud):
            pcd = convert_to_o3dpcd(instance[:,:3])
            f_pcd = __CLUSTERING__[self.mesher_cfg.CLUSTERING.NAME](self.mesher_cfg, pcd)
            if len(f_pcd.points) < self.min_lidar_pts_to_mesh:
                continue

            try: 
                mesh = __MESH_ALGORITHMS__[self.mesher_cfg.MESH_ALGORITHM.NAME](self.mesher_cfg, f_pcd)
                if not mesh.is_empty():
                    sampled_pts = __SAMPLING__[self.mesher_cfg.SAMPLING.NAME](self.mesher_cfg, mesh, len(f_pcd.points))
                    sampled_pts = sampled_pts.crop(f_pcd.get_axis_aligned_bounding_box())
                    if sampled_pts.is_empty():
                        continue
                    list_meshed_instances.append(sampled_pts)
            except Exception as e:
                print(f'mesher.py [mesh_det_pts]: sample idx: {sample_idx} - object idx: {object_id} - Error: {e}')
                continue

        return list_meshed_instances
    
    def replace_pts_with_mesh_pts(self, original_pcd, list_mesh_instances, label_points=False, point_dist_thresh=0.1):
        """
        Remove the cropped pointcloud from the original pointcloud and replace with the mesh pts. 
        Right now the seglabels are hardcoded for only bg and car class. If I extend this work
        to pedestrians, then the label will be determined by seg mask or gt label. 
        
        original_pcd: open3d.Pointcloud
        list_mesh_instances: list(open3d.Pointcloud)
        """
        if not list_mesh_instances:
            if label_points:
                original_pcd = np.asarray(original_pcd.points)
                one_hot_bg_label = np.hstack( (np.ones((original_pcd.shape[0], 1)), np.zeros((original_pcd.shape[0], 1)) ) )
                original_pcd_labelled = np.hstack((original_pcd, one_hot_bg_label))
                return original_pcd_labelled
            else:    
                return np.asarray(original_pcd.points)

        mesh_obj_points = np.concatenate([np.asarray(inst_pcd.points) for inst_pcd in list_mesh_instances])
        dists = [original_pcd.compute_point_cloud_distance(meshed_instance) for meshed_instance in list_mesh_instances]
        cropped_inds = np.concatenate([np.where(np.asarray(d) < point_dist_thresh)[0] for d in dists])
        pcd_without_object = np.asarray(original_pcd.select_by_index(cropped_inds, invert=True).points)

        if label_points:
            one_hot_bg_label = np.hstack((np.ones((pcd_without_object.shape[0], 1)), np.zeros((pcd_without_object.shape[0], 1))))
            one_hot_car_label = np.hstack((np.zeros((mesh_obj_points.shape[0], 1)), np.ones((mesh_obj_points.shape[0], 1))))
            pcd_without_object_labelled = np.hstack((pcd_without_object, one_hot_bg_label))
            mesh_obj_points_labelled = np.hstack((mesh_obj_points, one_hot_car_label))
            pcd_with_meshed_objects_labelled = np.vstack((pcd_without_object_labelled, mesh_obj_points_labelled))
            return pcd_with_meshed_objects_labelled
        else:
            pcd_with_meshed_objects = np.vstack((mesh_obj_points, pcd_without_object))
            return pcd_with_meshed_objects
    
    def save_pcd(self, points, sample_idx, labelled_pcd=False):
        if self.data_obj.dataset_name == 'nuscenes':
            sample_token = self.data_obj.get_sample_token_from_idx(sample_idx)
            save_fname = str(self.save_dir / f'{sample_token}#{sample_idx:06}')
        elif self.data_obj.dataset_name in ['kitti','baraja']:
            save_fname = str(self.save_dir / f'{sample_idx:06}')
        elif self.data_obj.dataset_name == 'waymo':
            info = self.data_obj.get_infos(sample_idx)
            sequence_name = info['point_cloud']['lidar_sequence']
            sample_idx = info['point_cloud']['sample_idx']
            save_fname = str(self.save_dir / sequence_name /f'{sample_idx:04}')            
        else:
            print(f"mesher.py [save_pcd]: {self.data_obj.dataset_name} not supported")
            return

        if labelled_pcd:
            save_fname = save_fname + '.bin'
            os.makedirs(os.path.dirname(save_fname), exist_ok=True)
            points = np.float32(points)
            points.tofile(save_fname)
        else:
            # .pcd format from o3d only saves (N,3) shape
            save_fname = save_fname + '.pcd'
            os.makedirs(os.path.dirname(save_fname), exist_ok=True)
            save_pcd = o3d.geometry.PointCloud()
            save_pcd.points = o3d.utility.Vector3dVector(points)
            try:
                o3d.io.write_point_cloud(save_fname, save_pcd, write_ascii=False)
            except Exception as e:
                print(f'sample idx: {sample_idx} - pcd pts: {save_pcd} - Error: {e}')

    def save_to_gt_database(self, pcd, sample_idx, object_idx, xyzlwhry_bbox):
        # Save to gt database
        if self.data_obj.dataset_name == 'nuscenes':
            sample_token = self.data_obj.get_sample_token_from_idx(sample_idx)
            bbox_str = '%'.join([str(xyzlwhry_bbox[i]) for i in range(len(xyzlwhry_bbox))])
            file_name = f'{sample_token}#{sample_idx:06}_car_oid{object_idx}_p{len(pcd.points)}_bbox{bbox_str}.pcd'
            save_fname = str(self.save_gt_db_dir / file_name)
        elif self.data_obj.dataset_name == 'kitti':
            # TODO: Currently kitti doesn't support gt database update
            print(f"{self.data_obj.dataset_name} gt update not supported at the moment")
            # bbox_str = '%'.join([str(xyzlwhry_bbox[i]) for i in range(len(xyzlwhry_bbox))])
            # file_name = f'{sample_idx:06}_car_oid{object_idx}_p{len(pcd.points)}_bbox{bbox_str}.pcd'
            # save_fname = str(self.save_gt_db_dir / file_name)
        else:
            print(f"mesher.py [save_to_gt_database]: {self.data_obj.dataset_name} not supported")
            return

        # Save as bin file for compatibility with existing OpenPCDet gt_sampling
        os.makedirs(os.path.dirname(save_fname), exist_ok=True)
        o3d.io.write_point_cloud(save_fname, pcd, write_ascii=False)