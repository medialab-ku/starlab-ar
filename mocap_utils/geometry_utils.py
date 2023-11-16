# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
# sys.path.append("/")
import numpy as np
import torch
from torch.nn import functional as F
import cv2
import numpy.matlib as npm
import mocap_utils.geometry_utils_torch as gut


def flip_hand_pose(pose):
    pose = pose.copy()
    if len(pose.shape) == 1:
        pose = pose.reshape(-1, 3)
        pose[:, 1] *= -1
        pose[:, 2] *= -1
        return pose.reshape(-1,)
    else:
        assert len(pose.shape) == 2
        pose[:, 1] *= -1
        pose[:, 2] *= -1
        return pose


def flip_hand_joints_3d(joints_3d):
    assert joints_3d.shape[1] == 3
    assert len(joints_3d.shape) == 2
    rot_mat = np.diag([-1, 1, 1])
    return np.matmul(rot_mat, joints_3d.T).T


def __quaternion_to_angle_axis_torch(quat):
    quat = quat.clone()
    if quat.dim() == 1:
        assert quat.size(0) == 4
        quat = quat.view(1, 4)
        angle_axis = gut.quaternion_to_angle_axis(quat)[0]
    elif quat.dim() == 2:
        assert quat.size(1) == 4
        angle_axis = gut.quaternion_to_angle_axis(quat)
    else:
        assert quat.dim() == 3
        dim0 = quat.size(0)
        dim1 = quat.size(1)
        assert quat.size(2) == 4
        quat = quat.view(dim0*dim1, 4)
        angle_axis = gut.quaternion_to_angle_axis(quat)
        angle_axis = angle_axis.view(dim0, dim1, 3)
    return angle_axis


def quaternion_to_angle_axis(quaternion):
    quat = quaternion
    if isinstance(quat, torch.Tensor):
        return __quaternion_to_angle_axis_torch(quaternion)
    else:
        assert isinstance(quat, np.ndarray)
        quat_torch = torch.from_numpy(quat)
        angle_axis_torch = __quaternion_to_angle_axis_torch(quat_torch)
        return angle_axis_torch.numpy()


def __angle_axis_to_quaternion_torch(aa):
    aa = aa.clone()
    if aa.dim() == 1:
        assert aa.size(0) == 3 
        aa = aa.view(1, 3)
        quat = gut.angle_axis_to_quaternion(aa)[0]
    elif aa.dim() == 2:
        assert aa.size(1) == 3
        quat = gut.angle_axis_to_quaternion(aa)
    else:
        assert aa.dim() == 3
        dim0 = aa.size(0)
        dim1 = aa.size(1)
        assert aa.size(2) == 3
        aa = aa.view(dim0*dim1, 3)
        quat = gut.angle_axis_to_quaternion(aa)
        quat = quat.view(dim0, dim1, 4)
    return quat


def angle_axis_to_quaternion(angle_axis):
    aa = angle_axis
    if isinstance(aa, torch.Tensor):
        return __angle_axis_to_quaternion_torch(aa)
    else:
        assert isinstance(aa, np.ndarray)
        aa_torch = torch.from_numpy(aa)
        quat_torch = __angle_axis_to_quaternion_torch(aa_torch)
        return quat_torch.numpy()


def __angle_axis_to_rotation_matrix_torch(aa):
    aa = aa.clone()
    if aa.dim() == 1:
        assert aa.size(0) ==3 
        aa = aa.view(1, 3)
        rotmat = gut.angle_axis_to_rotation_matrix(aa)[0][:3, :3]
    elif aa.dim() == 2:
        assert aa.size(1) == 3
        rotmat = gut.angle_axis_to_rotation_matrix(aa)[:, :3, :3]
    else:
        assert aa.dim() == 3
        dim0 = aa.size(0)
        dim1 = aa.size(1)
        assert aa.size(2) == 3
        aa = aa.view(dim0*dim1, 3)
        rotmat = gut.angle_axis_to_rotation_matrix(aa)
        rotmat = rotmat.view(dim0, dim1, 4, 4)
        rotmat = rotmat[:, :, :3, :3]
    return rotmat


def angle_axis_to_rotation_matrix(angle_axis):
    aa = angle_axis
    if isinstance(aa, torch.Tensor):
        return __angle_axis_to_rotation_matrix_torch(aa)
    else:
        assert isinstance(aa, np.ndarray)
        aa_torch = torch.from_numpy(aa)
        rotmat_torch = __angle_axis_to_rotation_matrix_torch(aa_torch)
        return rotmat_torch.numpy()


def __rotation_matrix_to_angle_axis_torch(rotmat):
    rotmat = rotmat.clone()
    if rotmat.dim() == 2:
        assert rotmat.size(0) == 3
        assert rotmat.size(1) == 3
        rotmat0 = torch.zeros((1, 3, 4))
        rotmat0[0, :, :3] = rotmat
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)[0]
    elif rotmat.dim() == 3:
        dim0 = rotmat.size(0)
        assert rotmat.size(1) == 3
        assert rotmat.size(2) == 3
        rotmat0 = torch.zeros((dim0, 3, 4))
        rotmat0[:, :, :3] = rotmat
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)
    else:
        assert rotmat.dim() == 4
        dim0 = rotmat.size(0)
        dim1 = rotmat.size(1)
        assert rotmat.size(2) == 3
        assert rotmat.size(3) == 3
        rotmat0 = torch.zeros((dim0*dim1, 3, 4))
        rotmat0[:, :, :3] = rotmat.view(dim0*dim1, 3, 3)
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)
        aa = aa.view(dim0, dim1, 3)
    return aa


def rotation_matrix_to_angle_axis(rotmat):
    if isinstance(rotmat, torch.Tensor):
        return __rotation_matrix_to_angle_axis_torch(rotmat)
    else:
        assert isinstance(rotmat, np.ndarray)
        rotmat_torch = torch.from_numpy(rotmat)
        aa_torch = __rotation_matrix_to_angle_axis_torch(rotmat_torch)
        return aa_torch.numpy()
    

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    assert isinstance(x, torch.Tensor), "Current version only supports torch.tensor"

    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def angle_axis_to_rot6d(aa):
    assert aa.dim() == 2
    assert aa.size(1) == 3
    bs = aa.size(0)

    rotmat = angle_axis_to_rotation_matrix(aa)
    rot6d = rotmat[:, :3, :2]
    return rot6d

#-----------------------------------------------------------------------------------------#
#                         tracking & temporal optimization utils                                    
#-----------------------------------------------------------------------------------------#

def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def smooth_global_rot_matrix(pred_rots, OE_filter):
    rot_mat = batch_rodrigues(pred_rots[None]).squeeze(0)
    smoothed_rot_mat = OE_filter.process(rot_mat)
    smoothed_rot = rotation_matrix_to_angle_axis(smoothed_rot_mat.reshape(1,3,3)).reshape(-1)
    return smoothed_rot

    device = pred_rots.device
    #print('before',pred_rots)
    rot_euler = transform_rot_representation(pred_rots.cpu().numpy(), input_type='vec',out_type='mat')
    smoothed_rot = OE_filter.process(rot_euler)
    smoothed_rot = transform_rot_representation(smoothed_rot, input_type='mat',out_type='vec')
    smoothed_rot = torch.from_numpy(smoothed_rot).float().to(device)
    #print('after',smoothed_rot)
    return smoothed_rot

class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
        s = value
    else:
        s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s

class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x, print_inter=False):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    
    if isinstance(edx, float):
        cutoff = self.mincutoff + self.beta * np.abs(edx)
    elif isinstance(edx, np.ndarray):
        cutoff = self.mincutoff + self.beta * np.abs(edx)
    elif isinstance(edx, torch.Tensor):
        cutoff = self.mincutoff + self.beta * torch.abs(edx)
    if print_inter:
        print(self.compute_alpha(cutoff))
    return self.x_filter.process(x, self.compute_alpha(cutoff))

def check_filter_state(OE_filters, signal_ID, show_largest=False, smooth_coeff=3.):
    if len(OE_filters)>100:
        del OE_filters
    if signal_ID not in OE_filters:
        if show_largest:
            OE_filters[signal_ID] = create_OneEuroFilter(smooth_coeff)
        else:
            OE_filters[signal_ID] = {}
    if len(OE_filters[signal_ID])>1000:
        del OE_filters[signal_ID]

def create_OneEuroFilter(smooth_coeff):
    return {'smpl_thetas': OneEuroFilter(smooth_coeff, 0.7), 'cam': OneEuroFilter(1.6, 0.7), 'smpl_betas': OneEuroFilter(0.6, 0.7), 'global_rot': OneEuroFilter(smooth_coeff, 0.7)}


def smooth_results(filters, body_pose=None, body_shape=None, cam=None):
    if body_pose is not None:
        # print(body_pose.shape)
        # global_rot = smooth_global_rot_matrix(body_pose[:3], filters['global_rot'])
        # body_pose = torch.cat([global_rot, filters['smpl_thetas'].process(body_pose[3:])], 0)
        body_pose = filters['smpl_thetas'].process(body_pose)
    if body_shape is not None:
        body_shape = filters['smpl_betas'].process(body_shape)
    if cam is not None:
        cam = filters['cam'].process(cam)
    return body_pose, body_shape, cam


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def get_tracked_ids(detections, tracked_objects):
    tracked_ids_out = np.array([obj.id for obj in tracked_objects])
    tracked_points = np.array([obj.last_detection.points[0] for obj in tracked_objects])
    org_points = np.array([obj.points for obj in detections])
    tracked_ids = [tracked_ids_out[np.argmin(np.linalg.norm(tracked_points-point[None], axis=1))] for point in org_points]
    return tracked_ids

def get_tracked_ids3D(detections, tracked_objects):
    tracked_ids_out = np.array([obj.id for obj in tracked_objects])
    tracked_points = np.array([obj.last_detection.points for obj in tracked_objects])
    org_points = np.array([obj.points for obj in detections])
    tracked_ids = [tracked_ids_out[np.argmin(np.linalg.norm(tracked_points.reshape(-1,4)-point.reshape(1,4), axis=1))] for point in org_points]
    return tracked_ids