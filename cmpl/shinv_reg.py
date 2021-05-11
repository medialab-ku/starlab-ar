import open3d as o3d
import numpy as np
from numpy import linalg as LA
import torch

import cmpl.shinv_config as cfg


def convert_torch2numpy(torch):
    return torch.detach().cpu().numpy().squeeze()

def convert_numpy2pcd(numpy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(numpy)
    return pcd


def compute_fpfh(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=cfg.REG_FPFH_NN[0]))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=cfg.REG_FPFH_NN[1]))
    return pcd_down, pcd_fpfh

def compute_stats(numpy):
    mean = numpy.mean(axis=0)
    cov = np.cov(numpy.T)
    eig_w, eig_v = LA.eig(cov)
    eig_i = eig_w.argsort()[::-1]
    eig_w = eig_w[eig_i]
    eig_v = eig_v[:, eig_i]
    return mean, cov, eig_w, eig_v


def apply_global_registration(source_pcd, target_pcd, voxel_size, distance):
    source_down, source_fpfh = compute_fpfh(source_pcd, voxel_size)
    target_down, target_fpfh = compute_fpfh(target_pcd, voxel_size)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance))
    return result.transformation

def apply_local_registration(source, target, distance, init):
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result.transformation


def run(source_points, target_points):

    # convert data
    source_numpy = convert_torch2numpy(source_points)
    target_numpy = convert_torch2numpy(target_points)
    source_pcd = convert_numpy2pcd(source_numpy)
    target_pcd = convert_numpy2pcd(target_numpy)

    # apply registration process
    result_global = apply_global_registration(source_pcd, target_pcd, cfg.REG_VOXEL_SIZE, cfg.REG_DIST_GLOBAL)
    result_local = apply_local_registration(source_pcd, target_pcd, cfg.REG_DIST_LOCAL, result_global)

    # get roation and translation
    transformation = result_local.astype(np.float32)
    rotation = torch.from_numpy(transformation[:3, :3].T).cuda()
    translation = torch.from_numpy(transformation[3, :3]).cuda()

    return torch.add(torch.matmul(source_points, rotation), translation)
