import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset


from arguments import Arguments

from utils.pc_transform import voxelize
from utils.plot import draw_any_set
from utils.common_utils import *
from utils.inversion_dist import *
from loss import *

from shape_inversion import ShapeInversion

from model.treegan_network import Generator, Discriminator
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw

import random

if __name__ == "__main__":
    args = Arguments(stage='inversion').parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    
    if not os.path.isdir('./logs/'):
        os.mkdir('./logs/')
    if not os.path.isdir('./saved_results'):
        os.mkdir('./saved_results')
    
    if args.dist:
        rank, world_size = dist_init(args.port)

    trainer = Trainer(args)
    trainer.run()
    
    