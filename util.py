import copy
import numpy as np
import open3d as o3d
import trimesh
import h5py
import random
from scipy.stats import special_ortho_group

import config

"""
Utility Functions
"""

def read_mesh(path):
    # NOTE: expect auto-triangulation from trimesh
    return trimesh.load(path, force='mesh').as_open3d
    # return o3d.io.read_triangle_mesh(path)

def write_mesh(path, mesh):
    return o3d.io.write_triangle_mesh(path, mesh)

def read_pcd(path):
    return o3d.io.read_point_cloud(path)

def write_pcd(path, pcd):
    return o3d.io.write_point_cloud(path, pcd)

def read_pcd_vscan(path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.loadtxt(path))
    return pcd

def visualize_mesh(mesh):
    o3d.visualization.draw_geometries([mesh])

def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])

def convert_mesh2pcd(mesh):
    pcd = mesh.sample_points_uniformly   (number_of_points=config.SAMPLING_UNIFORM)
    pcd = mesh.sample_points_poisson_disk(number_of_points=config.SAMPLING_POISSON, pcl=pcd)
    return pcd

def write_data(path, complete_pcds, incomplete_pcds, labels):
    with h5py.File(path, 'w') as f:
        f.create_dataset(config.HDF5_DATASET_NAME_COMPLETE_PCDS  , data=complete_pcds)
        f.create_dataset(config.HDF5_DATASET_NAME_INCOMPLETE_PCDS, data=incomplete_pcds)
        f.create_dataset(config.HDF5_DATASET_NAME_LABELS         , data=labels)

def augment_data(complete_pcd, incomplete_pcd):

    # copy data
    complete_pcd = copy.deepcopy(complete_pcd)
    incomplete_pcd = copy.deepcopy(incomplete_pcd)

    # set random augmentation value
    translation = np.asarray([random.uniform(*config.AUGMENT_TRANSLATION),
                              random.uniform(*config.AUGMENT_TRANSLATION),
                              random.uniform(*config.AUGMENT_TRANSLATION)], dtype=np.float32)
    rotation    = np.asarray(special_ortho_group.rvs(3), dtype=np.float32)
    scaling     = np.asarray(random.uniform(*config.AUGMENT_SCALING), dtype=np.float32)

    # augment data
    complete_pcd   = complete_pcd   * scaling
    complete_pcd   = complete_pcd   @ rotation.T
    complete_pcd   = complete_pcd   + translation
    incomplete_pcd = incomplete_pcd * scaling
    incomplete_pcd = incomplete_pcd @ rotation.T
    incomplete_pcd = incomplete_pcd + translation

    return complete_pcd, incomplete_pcd
