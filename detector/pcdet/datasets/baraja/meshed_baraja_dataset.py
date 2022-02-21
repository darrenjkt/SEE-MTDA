import copy
import pickle

import numpy as np
from skimage import io
import open3d as o3d
import json

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti, self_training_utils
from ..dataset import DatasetTemplate


class MeshedBarajaDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / 'test'

        split_dir = self.root_path / 'ImageSets' / 'test.txt'
        assert split_dir.exists(), f"No file found at {split_dir}"
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        print('self.sample_id_list = ', self.sample_id_list)
        self.infos = []
        self.include_data(self.mode)

    def include_data(self, mode):
        print(f'len(self.infos) = {len(self.infos)}')
        if self.logger is not None:
            self.logger.info('Loading MeshedBarajaDataset dataset')
        baraja_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                baraja_infos.extend(infos)
        self.infos.extend(baraja_infos)

        if self.logger is not None:
            self.logger.info('Total samples for MeshedBarajaDataset dataset: %d' % (len(self.infos)))

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'pcd' / ('%s.pcd' % idx)
        assert lidar_file.exists()
        pcd = o3d.io.read_point_cloud(str(lidar_file))
        return np.asarray(pcd.points, dtype=np.float32)

    def get_calib(self):
        calib_file = self.root_split_path / 'calib'/ 'front.json'
        assert calib_file.exists(), f'No calib file found at {calib_file}'
        with open(calib_file) as f:
            calib = json.load(f)
            return calib

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image' / 'front' / ('%s.png' % idx)
        assert img_file.exists(), f'No image file found at {img_file}'
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        # Annotations using SUSTechPOINTS is in JSON
        label_file = self.root_split_path / 'label' / ('%s.json' % idx)
        assert label_file.exists()
        with open(label_file) as f:
            label = json.load(f)
            return label

    def get_infos_from_idx(self, index):

        for i, dic in enumerate(self.infos):        
            if dic['point_cloud']['lidar_idx'] == index:
                return i
        return -1

    def get_meshed_lidar(self, idx):
        """
        Loads meshed lidar for a sample
        Args:
            idx: int, Sample index
        Returns:
            lidar: (N, 3), 2D np.float32 array
        """
        info_idx = self.get_infos_from_idx(f'{int(idx):06d}')
        info = self.infos[info_idx]
        meshed_lidar_file = self.root_split_path / info['meshed_lidar_path']
        assert meshed_lidar_file.exists(), f"No file at: {meshed_lidar_file}"
        
        pcd = o3d.io.read_point_cloud(str(meshed_lidar_file))
        return np.asarray(pcd.points, dtype=np.float32)
        
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        from . import baraja_utils

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 3, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # Get labels
            obj_list = self.get_label(sample_idx)            
            heading, dimensions, locations = [], [], []
            obj_type, obj_id = [], []
            for i in range(len(obj_list)):
                psr = obj_list[i]['psr']
                locs = [psr['position']['x'],psr['position']['y'],psr['position']['z']]
                dims = [psr['scale']['x'],psr['scale']['y'],psr['scale']['z']]
                dimensions.append(dims)
                locations.append(locs)
                heading.append(psr['rotation']['z'])
                obj_id.append(obj_list[i]['obj_id'])
                obj_type.append(obj_list[i]['obj_type'])

            anno = {}
            anno['name'] = np.array(obj_type)
            anno['location'] = np.array(locations)
            anno['dimensions'] = np.array(dimensions)
            anno['heading_angles'] = np.array(heading)
            anno['obj_ids'] = np.array(obj_id)
            anno['gt_boxes_lidar'] = np.concatenate([anno['location'], anno['dimensions'], anno['heading_angles'][...,np.newaxis]], axis=1)

            pcd = baraja_utils.convert_to_o3dpcd(self.get_lidar(sample_idx))
            num_points_in_gt = []
            for i in range(len(anno['gt_boxes_lidar'])):
                gt_box = anno['gt_boxes_lidar'][i]
                try:
                    o3dbox = baraja_utils.get_o3dbox(gt_box)
                except Exception as e:
                    print(f"{sample_idx}: {e}")
                num_points_in_gt.append(len(pcd.crop(o3dbox).points))

            anno['num_points_in_gt'] = np.array(num_points_in_gt)
            info['annos'] = anno

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return None, {}

        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        from ..kitti import kitti_utils

        map_name_to_kitti = {'Car':'Car'}

        eval_det_annos = copy.deepcopy(det_annos)
        if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
            print(f'Removing gt with points less than {self.dataset_cfg.FILTER_MIN_POINTS_IN_GT}')
            eval_gt_annos = []
            for info in self.infos:
                annos = info['annos']
                minpts_mask = (annos['num_points_in_gt'] >= self.dataset_cfg.FILTER_MIN_POINTS_IN_GT)
                for key in annos.keys():                    
                    annos[key] = annos[key][minpts_mask]
                eval_gt_annos.append(copy.deepcopy(annos))
        else:
            eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        print(f"Getting evaluation results for the classes: {class_names}")
        kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
        kitti_utils.transform_annotations_to_kitti_format(
            eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
            info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
        )
        kitti_class_names = [map_name_to_kitti[x] for x in class_names]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)


    def __getitem__(self, index):

        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        if self.dataset_cfg.get('MESHED', False):
            points = self.get_meshed_lidar(sample_idx)
        else:
            points = self.get_lidar(sample_idx)
        
        calib = self.get_calib()

        img_shape = self.get_image_shape(sample_idx)
        if self.dataset_cfg.FOV_POINTS_ONLY:
            from . import baraja_utils
            fov_flag = baraja_utils.get_fov_flag(points, img_shape, calib)
            points = points[fov_flag]

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'image_shape': img_shape
        }

        if 'annos' in info:
            annos = info['annos']                        
            gt_boxes_lidar = annos['gt_boxes_lidar']
            gt_names = annos['name']
            num_points_in_gt = annos['num_points_in_gt']

            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            # Add a filter for minimum points so that we can compare only the objects that are meshed
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (annos['num_points_in_gt'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': gt_names if mask is None else gt_names[mask],
                'gt_boxes': gt_boxes_lidar if mask is None else gt_boxes_lidar[mask],
                'num_points_in_gt': num_points_in_gt if mask is None else num_points_in_gt[mask]
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

def create_baraja_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = MeshedBarajaDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    test_filedir = save_path / 'infos'
    test_filedir.mkdir(parents=True, exist_ok=True)

    print('---------------Start to generate data infos---------------')

    baraja_infos = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(test_filedir / 'baraja_infos_test.pkl', 'wb') as f:
        pickle.dump(baraja_infos, f)
    print('Baraja info train file is saved to %s' % test_filedir)

    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_baraja_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_baraja_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car'],
            data_path=ROOT_DIR / 'data' / 'baraja',
            save_path=ROOT_DIR / 'data' / 'baraja'
        )
