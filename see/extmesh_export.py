import pickle
import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
from sc.datasets.shared_utils import convert_to_o3dpcd
import yaml
import open3d as o3d
from easydict import EasyDict
from sc.mesher.mesher import Mesher
from pathlib import Path
import shutil
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Mesh all instances in the point cloud')
    parser.add_argument('--cfg_file', required=True, help='Specify cfg_file')
    args = parser.parse_args()

    print("\n----- Args -----")
    for arg in vars(args):
        print (f'- {arg}: {getattr(args, arg)}')
    print("\n")

    mesh_cfg = cfg_from_yaml_file(args.cfg_file)

    # Copy over cfg file to keep on record
    cfg_filepath = Path(args.cfg_file).resolve()
    tag = cfg_filepath.stem
    save_cfg_filepath = Path(mesh_cfg.DATA_PATH) / f'infos_meshed_{tag}' / cfg_filepath.name
    save_cfg_filepath.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_filepath, save_cfg_filepath)

    return args, mesh_cfg   

def cfg_from_yaml_file(cfg_file):
    cfg = EasyDict()
    print("\n----- Cfgs -----")
    with open(cfg_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
        print(yaml.dump(yaml_config))

    cfg.update(EasyDict(yaml_config))

    return cfg

if __name__ == "__main__":    

    args, mesh_cfg = parse_args()    

    mesher = Mesher(cfg=mesh_cfg, cfg_path=args.cfg_file)
    save_dir = mesher.data_obj.root_dir / 'exported' / 'vc' / 'test'

    for sample_idx in tqdm(range(mesher.data_obj.__len__()), total=mesher.data_obj.__len__()):
        pcd_gtboxes = mesher.get_pcd_gtboxes(sample_idx)

        for object_id in range(len(pcd_gtboxes['gt_boxes'])):    

            object_pcd = pcd_gtboxes['pcd'].crop(pcd_gtboxes['gt_boxes'][object_id])

            if len(object_pcd.points) < mesher.min_lidar_pts_to_mesh:
                continue

            box_pts = np.asarray(pcd_gtboxes['gt_boxes'][object_id].get_box_points())
            label = pcd_gtboxes['xyzlwhry_gt_boxes'][object_id][:7]

            # Save object
            # Save partial, complete, labels
            partial_dir = save_dir / 'partial'
            partial_dir.mkdir(exist_ok=True, parents=True)

            label_dir = save_dir / 'label' 
            label_dir.mkdir(exist_ok=True, parents=True)

            fname =  f'frame-{sample_idx}_car-{object_id:03d}'    
            o3d.io.write_point_cloud(str(partial_dir / f'{fname}.pcd'), object_pcd)
            with open(str(label_dir / f'{fname}.pkl'), 'wb') as f:
                label = {'bbox_pts': box_pts, 
                         'gtbox': label,
                         'pc_id': fname,
                         'dataset': mesher.data_obj.dataset_name}
                pickle.dump(label, f)

    file_glob = glob.glob(f'{str(label_dir)}/*')
    file_list = [fname.split('/')[-1].split('.')[0] for fname in file_glob]
    with open(f'{save_dir}/file_list.txt','w') as f:
        f.writelines('\n'.join(file_list))