import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Function
try:
    import expansion_penalty
except:
    pass
import math
import sys
from numbers import Number
from collections import Set, Mapping, deque


def square_distance(src, dst):
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2     
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # xm*xm + ym*ym + zm*zm
    return dist

def farthest_point_sample(xyz, npoint):
    
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
    	# Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, xyz, new_xyz, nsample=500, density_only=True):
    """
    code borrowed from: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists: [B, S, N] Record the Euclidean distance between the center point and all points
    sqrdists = square_distance(new_xyz, xyz) # shape (B, S, N)
    # Find all distances greater than radius^2, its group_idx is directly set to N; the rest retain the original value
    
    if not density_only:
        group_idx[sqrdists > radius ** 2] = N
        # Do ascending order, the front is greater than radius^2 are N, will be the maximum, so will take the first nsample points directly in the remaining points
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        # Considering that there may be points in the previous nsample points that are assigned N (ie, less than nsample points in the spherical area), this point needs to be discarded, and the first point can be used instead.
        # group_first: [B, S, k], actually copy the value of the first point in group_idx to the dimension of [B, S, K], which is convenient for subsequent replacement.
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        # Find the point where group_idx is equal to N
        mask = group_idx == N
        # Replace the value of these points with the value of the first point
        group_idx[mask] = group_first[mask]
        return group_idx
    else:
        raw_mat = torch.zeros(B,S,N)
        density_mat = torch.zeros(B,S)
        raw_mat[sqrdists <= radius ** 2] = 1
        density_mat = torch.sum(raw_mat,2)
        # print(torch.max(sqrdists))
        return density_mat


class kNNRepulsionLoss(nn.Module):
    """
    adapted PU-Net's uniform loss
    """
    def __init__(self, k=10, n_seeds=20, h=0.01):
        super(kNNRepulsionLoss,self).__init__()
        self.k = k
        self.n_seeds = n_seeds
        self.h = h
    def forward(self, pcs):
        tic = time.time()
        n_seeds = self.n_seeds
        k = self.k
        seeds = farthest_point_sample(pcs,n_seeds) # which gives index
        temp = time.time()
        seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) # grad

        pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
        seeds_new = seeds_value.unsqueeze(1).repeat(1,2048,1,1)
        dist = pcs_new.add(-seeds_new)
        dist_value = torch.norm(dist,dim=3)
        toc = time.time()
        dist_new = dist_value.transpose(1,2)
        tac = time.time()
        top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
        top_dist_net = top_dist[:,:,1:]
        weights = torch.exp(-torch.pow(top_dist_net,2)*(1/(self.h**2)))
        repulsion = torch.mul(-top_dist_net,weights)
        return repulsion.sum(2).sum(1).mean()

