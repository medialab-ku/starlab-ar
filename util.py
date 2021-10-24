import numpy as np
from numpy import linalg as LA
import open3d as o3d
import cv2


"""
Settings
"""

SAMPLING_UNIFORM = 2048 * 5
SAMPLING_POISSON = 2048

CONSTRUCT_ALPHA = 0.2

"""
Functions
"""

def show_depth_map(name: str, map: np.ndarray) -> None:
    image = 255 * (map / 4)
    image = image.astype(np.uint8)
    cv2.imshow(name, image)
    cv2.waitKey(1)

def show_normal_map(name: str, map: np.ndarray) -> None:
    image = 255 * (map * 0.5 + 0.5)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, image)
    cv2.waitKey(1)

def write_depth_map(path: str, map: np.ndarray) -> None:
    image = 255 * (map / 4)
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)
    cv2.waitKey(1)

def write_normal_map(path: str, map: np.ndarray) -> None:
    image = 255 * (map * 0.5 + 0.5)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
    cv2.waitKey(1)

def read_tsdf_volume(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        return np.load(f)

def write_tsdf_volume(path: str, tsdf: np.ndarray) -> None:
    with open(path, 'wb') as f:
        np.save(f, tsdf)

def write_tsdf_mesh(path: str, tsdf: np.ndarray) -> None:
    vertices, triangles = mcubes.marching_cubes(tsdf, 0)
    vertices[:, 0:2] = -vertices[:, 0:2] # NOTE: yz-negation
    triangles[:, 1:] = triangles[:, 2:0:-1] # NOTE: reordering
    mcubes.export_obj(vertices, triangles, path)

def read_mesh(path: str) -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(path)

def write_mesh(path: str, mesh: o3d.geometry.TriangleMesh) -> None:
    return o3d.io.write_triangle_mesh(path, mesh)

def show_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

def convert_mesh2pcd(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_uniformly   (number_of_points=SAMPLING_UNIFORM)
    pcd = mesh.sample_points_poisson_disk(number_of_points=SAMPLING_POISSON, pcl=pcd)
    return pcd

def convert_pcd2mesh(pcd, alpha=1):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha * CONSTRUCT_ALPHA)
    mesh.compute_vertex_normals()
    return mesh

def convert_pcd2pts(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points, dtype=np.float32)

def convert_pts2pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def normalize_pts(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    mean = pts.mean(axis=0)
    pts_ = pts - mean
    norm = LA.norm(pts_, axis=1).max() * 2
    pts_ = pts_ / norm
    return pts_, mean, norm
