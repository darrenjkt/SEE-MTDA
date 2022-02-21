import open3d as o3d
import numpy as np
from sc.datasets.shared_utils import cart2sph, sph2cart, convert_to_o3dpcd, compress_to_bev
from sklearn.neighbors import NearestNeighbors


## ----- CLUSTERING AND SAMPLING -----

def db_scan(mesh_cfg, pcd, eps=None):
    centroid_dist = np.linalg.norm(np.asarray(pcd.get_center()))
    initial_ring_height = centroid_dist * np.tan(mesh_cfg.VRES * np.pi/180)
    o_pcd = pcd
    if eps is None:
        eps = mesh_cfg.CLUSTERING.EPS_SCALING * initial_ring_height

    if mesh_cfg.CLUSTERING.get('USE_BEV', False):
        dbpcd = compress_to_bev(o_pcd)
    else:
        dbpcd = o_pcd

    labels = np.array(dbpcd.cluster_dbscan(eps=eps, min_points=3, print_progress=False))
    y = np.bincount(labels[labels >= 0])
    if len(y) != 0:
        value = np.argmax(y)
        most_points = np.argwhere(labels == value)
        f_pcd = o_pcd.select_by_index(most_points)
        return f_pcd
    else:
        return o3d.geometry.PointCloud()

def vres_ring_based_sampling(mesh_cfg, mesh, num_pcd_pts, gt_box=None, reference_phi_theta=None):
    """
    Uses vertical resolution of lidar and trigonometry to determine what the
    ring separation is going to be at certain distances. Based on supplied
    optimal_ring_height, we will try to sample each mesh such that the number
    of points are approximately equal to a specific ring separation.

    Num inital points in a sense indicate how good the mesh quality is. If we upsample the 
    existing amount of points, we can in a sense mitigate the bad quality meshes.
    """
    if mesh_cfg.NAME == 'det_mesh':
        centroid_distance = np.linalg.norm(np.asarray(mesh.get_center()))        
    elif mesh_cfg.NAME == 'gt_mesh' and gt_box is not None:
        centroid_distance = np.linalg.norm(gt_box.get_center())
    else:
        print('mesher_methods.py [vres_ring_based_sampling]: Unsupported mesher')
        return None

    ring_height = centroid_distance * np.tan(mesh_cfg.VRES * np.pi/180)    
    upsampling_rate = ring_height/mesh_cfg.SAMPLING.OPTIMAL_RING_HEIGHT
    if mesh_cfg.SAMPLING.TYPE == 'uniform':
        return mesh.sample_points_uniformly(int(upsampling_rate*num_pcd_pts))
    elif mesh_cfg.SAMPLING.TYPE == 'poisson':
        return mesh.sample_points_poisson_disk(int(upsampling_rate*num_pcd_pts))
    else:
        print('mesher_methods.py [vres_ring_based_sampling]: Unsupported sampling')
        return None

def surface_area_based_sampling(mesh_cfg, mesh, pts_per_m2=None, num_pcd_pts=None, gt_box=None, reference_phi_theta=None):
    """
    Sample based on how big the surface area is (irregardless of distance). We just set
    a fixed number e.g. 100 pts per m2. In 3D cars, are the same size regardless of distance.
    So this would hopefully mean that at 50m KITTI, the car has same num points as car at
    20m nuScenes. The only diff is then in the actual quality of mesh at 20m vs 50m. 
    
    Try: 100, 250, 500
    """
    if pts_per_m2 == None:
        pts_per_m2 = mesh_cfg.SAMPLING.PTS_PER_M2
    
    surface_area = mesh.get_surface_area()
    num_pts_to_sample = max(int(surface_area*pts_per_m2), 1) # sample min 1 point so it doesn't throw error
    
    if mesh_cfg.SAMPLING.TYPE == 'uniform':
        return mesh.sample_points_uniformly(num_pts_to_sample)
    elif mesh_cfg.SAMPLING.TYPE == 'poisson':
        return mesh.sample_points_poisson_disk(num_pts_to_sample)
    else:
        print('mesher_methods.py [surface_area_based_sampling]: Unsupported sampling')
        return None

def virtual_lidar_sampling(mesh_cfg, mesh, pts_per_m2=None, knn_dist_thresh=None, reference_phi_theta=None, num_pcd_pts=None, gt_box=None):

    if pts_per_m2 == None:
        pts_per_m2 = mesh_cfg.SAMPLING.PTS_PER_M2
    if knn_dist_thresh == None:
        knn_dist_thresh = mesh_cfg.SAMPLING.KNN_DIST_THRESH        
    
    densely_sampled_pts = mesh.sample_points_uniformly(int(mesh.get_surface_area()*pts_per_m2))
    densely_sampled_pts_sph = cart2sph(np.asarray(densely_sampled_pts.points))
    try: 
        X = densely_sampled_pts_sph[:,1:3]
        Y = filter_to_mesh_range(reference_phi_theta, densely_sampled_pts_sph)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(Y)
        indices = indices[:,0]
        distances = distances[:,0]
            
        filt_indices = indices[np.where(distances < knn_dist_thresh)]
        dense_filt = densely_sampled_pts_sph[filt_indices]
        sampled_obj = sph2cart(dense_filt)    
        return convert_to_o3dpcd(sampled_obj)
    except:
        # No phi/theta at the object location
        return o3d.geometry.PointCloud()

## ----- MESH METHODS -----

def ball_pivoting(mesh_cfg, pcd, gt_box=None, lower_radius=None, upper_radius=None):
    """
    BP is a non-smooth mesh since the points of the pointcloud are also the vertices of 
    the resulting triangle mesh without any modifications.
    """
    if upper_radius is None:
        upper_radius = mesh_cfg.MESH_ALGORITHM.UPPER_RADIUS
    if lower_radius is None:
        lower_radius = mesh_cfg.MESH_ALGORITHM.LOWER_RADIUS
    
    ball_radius = np.linspace(lower_radius, upper_radius, 20)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0,0,0]))
    bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(ball_radius))
    
    return bp_mesh
    
def alpha_shapes(mesh_cfg, pcd, alpha=None):
    """
    Alpha is a non-smooth mesh since the points of the pointcloud are also the vertices of 
    the resulting triangle mesh without any modifications.
    """
    if alpha is None:
        centroid = np.linalg.norm(pcd.get_center())
        ring_height = centroid * np.tan(mesh_cfg.VRES * np.pi/180)
        alpha = mesh_cfg.MESH_ALGORITHM.ALPHA_SCALING * ring_height + mesh_cfg.MESH_ALGORITHM.ALPHA_OFFSET
    
    alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    alpha_mesh.compute_vertex_normals()
    
    return alpha_mesh    

def poisson_surface_reconstruction(mesh_cfg, pcd, depth=None, density_thresh=None, return_raw=False):
    """
    Poisson SR is a smooth mesh. Depth defines the depth of the octree for the surface reconstruction
    and hence implies the resolution of the resulting triangle mesh. High depth means a mesh with more
    details.

    The output of PSR is a set of densities where high densities are on the points and the interpolated
    points are lower densities. Density thresh sets which parts of the mesh to include.
    """
    if depth is None:
        depth = mesh_cfg.MESH_ALGORITHM.DEPTH
    if density_thresh is None:
        density_thresh = mesh_cfg.MESH_ALGORITHM.DENSITY_THRESH
        
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0,0,0]))    
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)    
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    
    if return_raw:
        import matplotlib.pyplot as plt
        
        density_colors = plt.get_cmap('plasma')(
            (densities - densities.min()) / (densities.max() - densities.min()))
        density_colors = density_colors[:, :3]
        density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        return density_mesh
    else:        
        vertices_to_remove = densities < np.quantile(densities, density_thresh)
        density_mesh.remove_vertices_by_mask(vertices_to_remove)
        return density_mesh    


# ------ HELPER METHODS ------

def filter_to_mesh_range(target_phi_theta, mesh_polar):
    target_phi, target_theta = target_phi_theta[:,0], target_phi_theta[:,1]
    mesh_phi, mesh_theta = mesh_polar[:,1], mesh_polar[:,2]
    
    inds = np.where(target_phi > min(mesh_phi)) and \
        np.where(target_phi < max(mesh_phi)) and \
        np.where(target_theta > min(mesh_theta)) and \
        np.where(target_theta < max(mesh_theta))
    
    return target_phi_theta[inds[0]]