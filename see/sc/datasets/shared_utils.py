import random
import colorsys
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry

def populate_gtboxes(sample_infos, dataset_name, classes, add_ground_lift=False):
    if dataset_name == 'nuscenes':
        zip_infos = zip(sample_infos['gt_boxes'], sample_infos['gt_names'], sample_infos['num_lidar_pts'])
    elif dataset_name in ['kitti', 'waymo','baraja']:
        anno = sample_infos['annos']
        annos_name = [name for name in anno['name']]
        zip_infos = zip(anno['gt_boxes_lidar'], annos_name, anno['num_points_in_gt'])
    else: 
        print(f"{dataset_name} is an unsupported dataset")
        return None

    pcd_gtboxes = {}        
    pcd_gtboxes['gt_boxes'], pcd_gtboxes['num_lidar_pts'], pcd_gtboxes['xyzlwhry_gt_boxes'] = [], [], []
    for idx, gt_anno in enumerate(zip_infos):
        bbox_corners, num_pts, xyzlwhry_bbox = get_o3dbox(gt_anno, classes=classes) 
        if bbox_corners != None:
            if add_ground_lift:
                bbox_corners.center = bbox_corners.center + [0,0,0.2] # Add 20cm to box centroid z-axis to get rid of the ground plane
            pcd_gtboxes['gt_boxes'].append(bbox_corners)
            pcd_gtboxes['num_lidar_pts'].append(num_pts)
            pcd_gtboxes['xyzlwhry_gt_boxes'].append(xyzlwhry_bbox)

    return pcd_gtboxes   

def get_pts_in_mask(dataset, instances, imgfov, shrink_percentage=0, use_bbox=False, labelled_pcd=False):
    """
    Return the points that are in each instance mask with category id.
    
    :dataset: COCO object that loaded in the json annotations from seg network
    :instances: List of instances loaded from dataset
    :imgfov_pts_2d: List of (u,v) pixel coordinates that are in the image fov
    :imgfov_pc_lidar: List of (x,y,z) coordinates in lidar frame within image fov
    :imgfov_pc_cam: List of (x,y,z) coordinates in camera frame within image fov. For KITTI this is the rectified frame.
    :use_bbox: Get points in bbox instead of in mask

    TODO: Not a fan of requiring the COCO object to be passed in. I only need it for converting from seg polygon to mask for now
    """
    imgfov_pts_2d, imgfov_pc_lidar, imgfov_pc_cam = imgfov['pts_img'], imgfov['pc_lidar'], imgfov['pc_cam']
    list_instance_pts_uv, list_instance_pc_cam_xyz, list_instance_pc_lidar_xyzls, list_labelled_pcd = [], [], [], []

    if labelled_pcd:
        imgfov_labelled_pc = imgfov['pc_labelled']
        
    for instance_orig in instances:
        try:
            instance = instance_orig.copy()
            if shrink_percentage != 0:
                instance['segmentation'] = shrink_instance_masks(instance['segmentation'], shrink_percentage=shrink_percentage)
            
            seg_mask = dataset.annToMask(instance)
            if use_bbox:
                bbox = np.array(instance['bbox'])
                bbox[2:4] = bbox[0:2] + bbox[2:4]
                boxmask = np.zeros(seg_mask.shape, dtype=np.uint8)
                boxmask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = True
                mask = boxmask
            else:
                mask = seg_mask                

        except Exception as e:
            # Some instances don't have a mask
            print(f'Frame: {instance["image_id"]} - Instance: {instance["id"]} Error')
            print(f'Exception: {e}')
            continue

        

        # imgfov_pts_2d is an array of pixels u,v shape (N,2) 
        # This line queries the u,v value in the mask array of shape (imgH,imgW) and return the value 0 or 1 at the queried u,v index
        # This will return an array of (N,2) with 0,1 depending on the value in the mask
        mask_idx = np.array(mask[imgfov_pts_2d[:,1],imgfov_pts_2d[:,0]], dtype=np.bool)

        instance_pts_uv = imgfov_pts_2d[mask_idx,:]
        instance_pc_lidar = imgfov_pc_lidar[mask_idx,:]

        if labelled_pcd:
            instance_labelled_pcd = imgfov_labelled_pc[mask_idx,:]
            list_labelled_pcd.append(instance_labelled_pcd)
        
        if imgfov_pc_cam is not None:
            instance_pc_cam = imgfov_pc_cam[mask_idx,:] 
            list_instance_pc_cam_xyz.append(instance_pc_cam)
                
        # We attach class label and segmentation mask len
        # If the object is split into more than 2 parts then
        # we can increase the clustering value
        instance_pc_lidar_xyzls = np.hstack((instance_pc_lidar, \
                                            instance['category_id']*np.ones((instance_pc_lidar.shape[0],1)), \
                                            instance['segmentation'].__len__()*np.ones((instance_pc_lidar.shape[0],1)) ))
        
        list_instance_pts_uv.append(instance_pts_uv)        
        list_instance_pc_lidar_xyzls.append(instance_pc_lidar_xyzls)
    
    instance_pts = {"img_uv": list_instance_pts_uv,
                    "cam_xyz":list_instance_pc_cam_xyz,
                    "lidar_xyzls": list_instance_pc_lidar_xyzls,
                    "labelled_pcd": list_labelled_pcd }
    return instance_pts

def draw_lidar_on_image(projected_points, img, instances=None, 
                        clip_distance=1.0, point_size=5, alpha=0.8, 
                        color_scheme='jet', map_range=25.0,
                        shrink_percentage=0, display=True):
    ''' 
    Function that takes in the transformed points and draws it on the image.
    This function is called by all the "show_" functions.    
    
    :img: Img in RGB format
    '''
    # Draw mask polygons
    if instances is not None:
        for instance_orig in instances:
            instance = instance_orig.copy()
            if shrink_percentage != 0:
                instance['segmentation'] = shrink_instance_masks(instance['segmentation'], shrink_percentage=shrink_percentage)
            segs = instance['segmentation']
            segs = [np.array(seg, np.int32).reshape((1, -1, 2))
                    for seg in segs]
            for seg in segs: 
                cv2.drawContours(img, seg, -1, (0,255,0), 2)
            bbox = np.array(instance['bbox'], dtype=np.uint64)
            bbox[2:4] = bbox[0:2] + bbox[2:4]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
            
    xs, ys, colors = [], [], []
    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba(point[2], alpha=alpha, color_scheme=color_scheme, map_range=map_range))
    
    if display:
        plt.figure(figsize=(15,10))
        plt.imshow(img)
        plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none", alpha=alpha)   
        plt.show() 
        
    return img
    

def visualise_instance_clouds(instance_clouds, original_cloud=None):
    """
    Visualise the instance point clouds either by themselves, or in 
    context with the original point cloud.
    
    :instance_clouds: List of (N,3) numpy arrays 
    :original_cloud: (N,3) numpy array 
    """
    # Colour each instance differently, the additional color generation is to ensure always bright, saturated colors
    color_list = []
    for instance_cloud in instance_clouds:
        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        r,g,b = [i for i in colorsys.hls_to_rgb(h,l,s)]
        colors = [[r,g,b] for i in range(instance_cloud.shape[0])]
        color_list.append(colors)

    i_fcloud = [x for x in instance_clouds if len(x) != 0]
    color_list_filt = [x for x in color_list if len(x) != 0]

    # Visualise all instances to double check. Here we join all the instance clouds to visualise each instance.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(i_fcloud)[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(color_list_filt))
    
    if original_cloud is not None:
        # We can also visualise it together with the entire cloud
        orig_cloud = o3d.geometry.PointCloud()
        orig_cloud.points = o3d.utility.Vector3dVector(original_cloud[:,:3])
        orig_cloud.colors = o3d.utility.Vector3dVector([[0.85,0.85,0.85] for i in range(original_cloud[:,:3].shape[0])])
        o3d.visualization.draw_geometries([orig_cloud, pcd])
        
    else:
        o3d.visualization.draw_geometries([pcd])
                
def visualise_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    o3d.visualization.draw_geometries([pcd])
    
    
def rgba(r, alpha=0.8, color_scheme='jet', map_range=25.0):
    """Generates a color based on range.

    Args:
    r: the range value of a given point.
    Returns:
    The color for a given range
    """
    c = plt.get_cmap(color_scheme)((r % map_range) / map_range)
    c = list(c)
    c[-1] = 0.8  # alpha
    return c

def gtbox_to_corners(box):
    """
    Takes an array containing [x,y,z,l,w,h,r], and returns an [8, 3] matrix that 
    represents the [x, y, z] for each 8 corners of the box.
    
    Note: Openpcdet __getitem__ gt_boxes are in the format [x,y,z,l,w,h,r,alpha]
    where alpha is "observation angle of object, ranging [-pi..pi]"
    """
    # To return
    corner_boxes = np.zeros((8, 3))

    translation = box[0:3]
    l, w, h = box[3], box[4], box[5] # waymo, nusc, kitti is all l,w,h after OpenPCDet processing
    rotation = box[6]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    return bounding_box.transpose(), rotation_matrix

def convert_to_o3dpcd(points):
    if type(points) == list:
        pcds = []
        for pointcloud in points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
            pcds.append(pcd)
        return pcds
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        return pcd

def compress_to_bev(obj_points):
    if type(obj_points) == o3d.cpu.pybind.geometry.PointCloud:
        obj_points = np.asarray(obj_points.points)
    
    bev = np.hstack([obj_points[:,:2], np.zeros((obj_points.shape[0], 1))])
    return convert_to_o3dpcd(bev)

def get_o3dbox(anno_info, classes):
    """
    Convert from gt_boxes in the format [x,y,z,l,w,h,ry] to open3d oriented bbox
    which can then be used to crop a pointcloud
    """

    gt_box, class_name, num_lidar_pts = anno_info
    if class_name in classes:  
        box_corners, r_mat = gtbox_to_corners(gt_box)
        boxpts = o3d.utility.Vector3dVector(box_corners)
        o3dbox = o3d.geometry.OrientedBoundingBox().create_from_points(boxpts)
        o3dbox.color = np.array([1,0,0])
        o3dbox.center = gt_box[0:3]
        o3dbox.R = r_mat
        return o3dbox, num_lidar_pts, gt_box
    else:
        return None, None, None
    
    
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def shrink_shapely_polygon(my_polygon, shrink_percentage):
    ''' returns a shapely polygon that is shrunk by a certain factor '''    

    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*(shrink_percentage/100)
    my_polygon_resized = my_polygon.buffer(-shrink_distance) #shrink
        
    return my_polygon_resized

def shrink_instance_masks(seg_masks, shrink_percentage):
    seg_list = []
    for seg in seg_masks:
        u = seg[::2]
        v = seg[1::2]
        poly = geometry.Polygon([[x,y] for x,y in zip(u,v)])
        resized_poly = shrink_shapely_polygon(poly, shrink_percentage=shrink_percentage)
        if isinstance(resized_poly, geometry.MultiPolygon):
            resized_poly = list(resized_poly)
            for r_poly in resized_poly:
                if r_poly.is_empty:
                    continue
                resized_x, resized_y = r_poly.exterior.coords.xy
                seg_list.append([int(val) for pair in zip(resized_x, resized_y) for val in pair])
        else:
            if resized_poly.is_empty:
                return seg_masks
            resized_x, resized_y = resized_poly.exterior.coords.xy
            seg_list.append([int(val) for pair in zip(resized_x, resized_y) for val in pair])
            
    return seg_list

def cart2sph(points):
    # Phi is up-down, theta is left-right
    x,y,z = points[:,0], points[:,1], points[:,2]
    r = np.linalg.norm(points, axis=1)
    phi = np.arctan2(np.linalg.norm(points[:,:2],axis=1),z) # also = np.arccos(z/r)
    theta = np.arctan2(y,x) # theta in radians, lidar is 360 degrees hence the 3.14 to -3.14 values
    
    return np.concatenate((r[:,np.newaxis],phi[:,np.newaxis],theta[:,np.newaxis]), axis=1)

def sph2cart(points):
    r, theta, phi = points[:,0], points[:,1], points[:,2]
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.concatenate((x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]), axis=1)

    