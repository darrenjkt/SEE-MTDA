import numpy as np
import os
import glob
import time
import argparse
from pathlib import Path
from pathos.pools import ProcessPool
from tqdm import tqdm
from sc.datasets.shared_utils import convert_to_o3dpcd
import setproctitle
import yaml
import open3d as o3d
from easydict import EasyDict
from sc.mesher.mesher import Mesher
import shutil
import time
import pickle

def mesh_process_gt(sample_idx):    
    
    setproctitle.setproctitle(f'Mesh-GT[{mesher.data_obj.dataset_name[:3].upper()}]:{sample_idx}/{mesher.data_obj.__len__()}')
    t0 = time.time()

    pcd_gtboxes = mesher.get_pcd_gtboxes(sample_idx)
    list_mesh_instances = mesher.mesh_gt_pts(pcd_gtboxes, sample_idx)
    add_seg_labels = mesher.mesher_cfg.get('ADD_SEG_LABELS', False)
    final_pcd = mesher.replace_pts_with_mesh_pts(pcd_gtboxes['pcd'], list_mesh_instances, label_points=add_seg_labels, point_dist_thresh=mesher.mesher_cfg.REPLACE_OBJECT_WITH_MESH.POINT_DISTANCE_THRESH)
    
    time_taken_frame = time.time() - t0
    if len(list_mesh_instances) != 0:
        time_taken_car = time_taken_frame/len(list_mesh_instances)
    else:
        time_taken_car = None

    mesher.save_pcd(final_pcd, sample_idx, labelled_pcd=add_seg_labels)

def mesh_process_det(sample_idx):    
    
    setproctitle.setproctitle(f'Mesh-DET[{mesher.data_obj.dataset_name[:3].upper()}]:{sample_idx}/{mesher.data_obj.__len__()}')
    t0 = time.time()
    i_cloud = mesher.data_obj.get_mask_instance_clouds(sample_idx)
    list_mesh_instances = mesher.mesh_det_pts(i_cloud, sample_idx)
    
    add_seg_labels = mesher.mesher_cfg.get('ADD_SEG_LABELS', False)
    original_pcd = convert_to_o3dpcd(mesher.data_obj.get_pointcloud(sample_idx))
    final_pcd = mesher.replace_pts_with_mesh_pts(original_pcd, list_mesh_instances, label_points=add_seg_labels, point_dist_thresh=mesher.mesher_cfg.REPLACE_OBJECT_WITH_MESH.POINT_DISTANCE_THRESH)        
    
    time_taken_frame = time.time() - t0
    if len(list_mesh_instances) != 0:
        time_taken_car = time_taken_frame/len(list_mesh_instances)
    else:
        time_taken_car = None

    mesher.save_pcd(final_pcd, sample_idx, labelled_pcd=add_seg_labels)
    return time_taken_frame, time_taken_car

def mesh_process_lidarseg(sample_idx):    
    
    setproctitle.setproctitle(f'Mesh-LSEG[{mesher.data_obj.dataset_name[:3].upper()}]:{sample_idx}/{mesher.data_obj.__len__()}')
    t0 = time.time()
    icloud_paths = glob.glob(f'{mesher.data_obj.root_dir}/exported/lidar_seg_data_groundlift/points_in_bbox/icloud/*.pcd')
    i_cloud = [np.asarray(o3d.io.read_point_cloud(path).points) for path in icloud_paths if  f'{sample_idx:06}_' in path]
    list_mesh_instances = mesher.mesh_det_pts(i_cloud, sample_idx)
    
    add_seg_labels = mesher.mesher_cfg.get('ADD_SEG_LABELS', False)
    original_pcd = convert_to_o3dpcd(mesher.data_obj.get_pointcloud(sample_idx))
    final_pcd = mesher.replace_pts_with_mesh_pts(original_pcd, list_mesh_instances, label_points=add_seg_labels, point_dist_thresh=mesher.mesher_cfg.REPLACE_OBJECT_WITH_MESH.POINT_DISTANCE_THRESH)        
    
    time_taken_frame = time.time() - t0
    if len(list_mesh_instances) != 0:
        time_taken_car = time_taken_frame/len(list_mesh_instances)
    else:
        time_taken_car = None

    mesher.save_pcd(final_pcd, sample_idx, labelled_pcd=add_seg_labels)
    return time_taken_frame, time_taken_car

def mesh_process_ext(sample_idx):
    """ 
    Loads in meshes of individual objects that were processed externally and joins them with the original cloud
    """
    setproctitle.setproctitle(f'Mesh-EXT[{mesher.data_obj.dataset_name[:3].upper()}]:{sample_idx}/{mesher.data_obj.__len__()}')
    
    t0 = time.time()
    original_pcd = convert_to_o3dpcd(mesher.data_obj.get_pointcloud(sample_idx))
    completed_pcd_paths = set(glob.glob(f'{mesher.data_obj.root_dir}/exported/vc/test/completed-{mesher.mesher_cfg.EXPORT_NAME}/*.pcd'))
    completed_meta_paths = [pcd_paths.replace('completed-', 'metadata-').replace('.pcd', '.pkl') for pcd_paths in list(completed_pcd_paths)]

    frame_objs = set([pcd_path for pcd_path in completed_pcd_paths if f'frame-{sample_idx}_' in pcd_path])    
    frame_meta = set([meta_path for meta_path in completed_meta_paths if f'frame-{sample_idx}_' in meta_path])
    num_pts = [pickle.load(open(pkl_file, 'rb'))['num_pts'] for pkl_file in frame_meta]
    list_mesh_instances = [o3d.io.read_point_cloud(obj_path) for idx, obj_path in enumerate(frame_objs) if num_pts[idx] > mesher.mesher_cfg.MIN_LIDAR_PTS_TO_MESH]    

    time_taken_frame = time.time() - t0
    if len(list_mesh_instances) != 0:
        time_taken_car = time_taken_frame/len(list_mesh_instances)
    else:
        time_taken_car = None

    final_pcd = mesher.replace_pts_with_mesh_pts(original_pcd, list_mesh_instances, label_points=False)
    mesher.save_pcd(final_pcd, sample_idx, labelled_pcd=False)

    return time_taken_frame, time_taken_car
    

def run(cfg):
    """
    Runs the processing of the dataset in parallel. Oddly enough this exact function doesn't work when I run it from
    within the mesher class. It only seems to work when the mesh_process function is in the global scope of the main.
    """    
    if cfg.MESHER.NAME == 'gt_mesh':
        print('surface_completion.py [run]: Meshing with GT box boundaries')
        mesh_process = mesh_process_gt
    elif cfg.MESHER.NAME == 'det_mesh':
        print('surface_completion.py [run]: Meshing with instance seg boundaries')
        mesh_process = mesh_process_det
    elif cfg.MESHER.NAME in ['ext_mesh', 'det_ext_mesh']:
        print('surface_completion.py [run]: Loading in externally meshed objects')
        mesh_process = mesh_process_ext
    elif cfg.MESHER.NAME == 'lseg_mesh':
        print('surface_completion.py [run]: Loading in isolated clouds')
        mesh_process = mesh_process_lidarseg
    else:        
        print('surface_completion.py [run]: Mesher method not supported')
        return None
    
    t1 = time.time()
    sample_indices = range(0, mesher.data_obj.__len__())
    # sample_indices = range(0, 1000)

    # Baraja only ()
    # indices_file = mesher.data_obj.root_dir / 'ImageSets' / 'test.txt'
    # assert indices_file.exists(), f"No file found at {indices_file}"
    # sample_indices = [int(x.strip()) for x in open(indices_file).readlines()] if indices_file.exists() else range(0, mesher.data_obj.__len__())    

    avg_time_p = []
    print(f'\nParallel processing: Generating {os.cpu_count()} processes')
    with ProcessPool(os.cpu_count()) as p:
        time_taken = list(tqdm(p.imap(mesh_process, sample_indices), total=mesher.data_obj.__len__()))
        avg_time_p.append(time_taken)
    
    avg_time = avg_time_p[0]
    avg_frame = [i[0] for i in avg_time]
    frame_mean = sum(avg_frame)/len(avg_frame)
    avg_car = [i[1] for i in avg_time if i[1] is not None]
    car_mean = sum(avg_car)/len(avg_car)
    print(f'Average time per frame = {frame_mean}s')
    print(f'Average time per car = {car_mean}s')

    # Update infos with the meshed paths
    mesher.data_obj.update_infos(mesher.save_dir)

    if mesher.data_obj.dataset_cfg.get('UPDATE_GT_DATABASE', False):
        print('Updating gt_database')
        mesher.data_obj.update_gt_database(mesher.save_gt_db_dir)

    print(f'Time taken for {mesher.data_obj.__len__()} files: {time.time()-t1}')

    
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
    run(mesh_cfg)

    # print(mesh_cfg.get('MESHER', 'none found'))