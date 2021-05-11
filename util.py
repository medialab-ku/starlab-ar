import copy

import numpy as np
import open3d as o3d


"""
Utility Settings
"""

CROPPING_DENOMINATOR = 2

SAMPLING_UNIFORM = 2048 * 5
SAMPLING_POISSON = 2048

CONSTRUCT_ALPHA = 0.2


"""
Utility Functions
"""

def load_mesh(path):
    return o3d.io.read_triangle_mesh(path)

def save_mesh(path, mesh):
    return o3d.io.write_triangle_mesh(path, mesh)

def load_pcd(path):
    return o3d.io.read_point_cloud(path)

def save_pcd(path, pcd):
    return o3d.io.write_point_cloud(path, pcd)

def visualize_mesh(mesh):
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])

def crop_mesh(mesh):
    mesh = copy.deepcopy(mesh)
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles)[:len(mesh.triangles) // CROPPING_DENOMINATOR, :])
    mesh.triangle_normals = o3d.utility.Vector3dVector(
        np.asarray(mesh.triangle_normals)[:len(mesh.triangle_normals) // CROPPING_DENOMINATOR, :])
    return mesh

def convert_mesh2pcd(mesh):
    pcd = mesh.sample_points_uniformly(number_of_points=SAMPLING_UNIFORM)
    pcd = mesh.sample_points_poisson_disk(number_of_points=SAMPLING_POISSON, pcl=pcd)
    return pcd

def convert_pcd2mesh(pcd):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, CONSTRUCT_ALPHA)
    mesh.compute_vertex_normals()
    return mesh
