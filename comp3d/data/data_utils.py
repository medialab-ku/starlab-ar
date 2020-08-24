import h5py
import numpy as np
import pandas as pd
import transforms3d
import random
import math
import torch

def pad_cloudN(P, Nin):
    """ Pad or subsample 3D Point cloud to Nin number of points """
    N = P.shape[0]
    P = P[:].astype(np.float32)

    rs = np.random.random.__self__
    choice = np.arange(N)
    if N > Nin: # need to subsample
        ii = rs.choice(N, Nin)
        choice = ii
    elif N < Nin: # need to pad by duplication
        ii = rs.choice(N, Nin - N)
        choice = np.concatenate([range(N),ii])
    P = P[choice, :]

    return P

def augment_cloud(Ps, args):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if args.pc_augm_scale > 1:
        s = random.uniform(1/args.pc_augm_scale, args.pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if args.pc_augm_rot:
        angle = random.uniform(0, 2*math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) # y=upright assumption
    if args.pc_augm_mirror_prob > 0: # mirroring x&z, not y
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)
    result = []
    for P in Ps:
        P[:,:3] = np.dot(P[:,:3], M.T)

        if args.pc_augm_jitter:
            sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
            P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
        result.append(P)
    return result

def load_h5(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    f = h5py.File(path, 'r')
    cloud_data = np.array(f['data'])
    f.close()

    return cloud_data.astype(np.float64)

def load_csv(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    return pd.read_csv(path, delim_whitespace=True, header=None).values


def normal2unit(vertices):
    """
    Return: (vertice_num, 3) => normalized into unit sphere
    """
    center = np.mean(vertices, axis=0)
    vertices -= center
    distance = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distance)
    return vertices

def rotate(points, degree: float, axis: int):
    """Rotate along upward direction"""
    rotate_matrix = np.identity(3, dtype=float)
    theta = (degree/360)*2*np.pi
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    axises = [0, 1, 2]
    assert  axis in axises
    axises.remove(axis)

    rotate_matrix[axises[0], axises[0]] = cos
    rotate_matrix[axises[0], axises[1]] = -sin
    rotate_matrix[axises[1], axises[0]] = sin
    rotate_matrix[axises[1], axises[1]] = cos
    points = points @ rotate_matrix
    return points, rotate_matrix

class Transform():
    def __init__(self, 
                normal: bool, 
                shift: float = None, 
                scale: float = None, 
                rotate: list = None,
                random:bool= False):

        self.normal = normal
        self.shift = shift
        self.scale = scale
        self.rotate = rotate
        self.random = random

    def __call__(self, points, gt, transform_gt=True):
        if self.normal:
            points = normal2unit(points)
            if transform_gt:
                gt = normal2unit(gt)

        if self.shift:
            shift = self.shift
            if self.random:
                shift = (np.random.rand(3)*2 - 1) * self.shift
            points += shift
            if transform_gt:
                gt += shift
        
        if self.scale:
            scale = self.scale
            if self.random:
                scale = np.random.rand(1).item() * self.scale
            points *= scale
            if transform_gt:
                gt *= scale
        
        Rt = np.identity(3, dtype=float)
        if self.rotate:
            for i, rot in enumerate(self.rotate):
                if rot > 0:
                    degree = rot
                    if self.random: 
                        degree = (np.random.rand(1).item()*2 - 1) * rot
                    points, rotmat = rotate(points, degree, i)
                    if transform_gt:
                        gt, rotmat = rotate(gt, degree, i)
                    
                    Rt = rotmat @ Rt

        return points, gt, Rt


def make_mask_gt(sauce, gtpts, K):
    """
    Args:
        sauce   ( S x 3 tensor )
        gtpts   ( B x N x 3 tensor )
        K       (constant) num of neighbors
    Outputs:
        mask    ( B x S tensor )
    """
    def knn(points, queries, K):
        """
        Args:
            points   ( B x N x 3 tensor )
            queries  ( B x M x 3 tensor )  M < N
            K        (constant) num of neighbors
        Outputs:
            knn     (B x M x K x 3 tensor) sorted K nearest neighbor
            indices (B x M x K tensor) knn indices
        """
        value = None
        indices = None
        num_batch = points.shape[0]
        for i in range(num_batch):
            point = points[i]
            query = queries[i]
            dist = torch.cdist(point, query)
            idxs = dist.topk(K, dim=0, largest=False, sorted=True).indices
            idxs = idxs.transpose(0, 1)
            nn = point[idxs].unsqueeze(0)
            value = nn if value is None else torch.cat((value, nn))

            idxs = idxs.unsqueeze(0)
            indices = idxs if indices is None else torch.cat((indices, idxs))

        return value.float(), indices.long()


    # (nsauce, 3) -> (bs, nsauce, 3)
    nsauce = sauce.size(0)
    sauce = sauce.repeat(gtpts.size(0), 1)
    sauce = sauce.reshape(gtpts.size(0), nsauce, -1)

    knn_pts, knn_indices = knn(sauce, gtpts, K)
    knn_indices = knn_indices.reshape(knn_indices.size(0), -1)
    mask = torch.zeros(sauce.size(0), sauce.size(1), 1).cuda()

    for i in range(sauce.size(0)):
        m = mask[i]
        idx = knn_indices[i]
        m[idx] = 1

    return mask


def test_mask():
    import _init_paths
    from parse_args import parse_args
    from shapenet import ShapeNet
    import open3d as o3d
    import numpy as np

    def sample_uniform_ball3d_2(num):
        rand = torch.randn(num, 5)
        norm = torch.norm(rand, dim=1)
        norm = norm.reshape(num, 1)
        pts = rand[:, [2, 3, 4]] / norm
        return pts

    num_sauce = 16384
    K = 1
    args = parse_args()
    sauce = sample_uniform_ball3d_2(num_sauce) * 0.5
    dataset = ShapeNet(args, 'test', None)

    gt, partial, meta = dataset[100]
    gtpts = torch.from_numpy(gt)
    mask = make_mask_gt(sauce.float(), gtpts.unsqueeze(0).float(), K).squeeze(2).squeeze(0)
    mask = mask.bool()

    color_gt = np.zeros((gt.shape[0], 3))                 # black
    color_sauce = np.ones((sauce.shape[0], 3)) * 0.5      # gray
    color_sauce[mask == True] = [1., 0., 0.]

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt)
    pcd_gt.colors = o3d.utility.Vector3dVector(color_gt)
    pcd_sauce = o3d.geometry.PointCloud()
    pcd_sauce.points = o3d.utility.Vector3dVector(sauce)
    pcd_sauce.colors = o3d.utility.Vector3dVector(color_sauce)
    o3d.visualization.draw_geometries([pcd_gt, pcd_sauce])


if __name__ == '__main__':
    test_mask()