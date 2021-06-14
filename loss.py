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

