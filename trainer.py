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

class Trainer(object):

    def __init__(self, args):
        self.args = args
        
        if self.args.dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        self.inversion_mode = args.inversion_mode
        
        save_inversion_dirname = args.save_inversion_path.split('/')
        log_pathname = './logs/'+save_inversion_dirname[-1]+'.txt'
        args.log_pathname = log_pathname

        self.model = ShapeInversion(self.args)
        if self.inversion_mode == 'morphing':
            self.model2 = ShapeInversion(self.args)
            self.model_interp = ShapeInversion(self.args)
        
        if self.args.dataset in ['MatterPort','ScanNet','KITTI','PartNet']:
            dataset = PlyDataset(self.args)
        else: 
            dataset = CRNShapeNet(self.args)
        
        sampler = DistributedSampler(dataset) if self.args.dist else None

        if self.inversion_mode == 'morphing':
            self.dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                sampler=sampler,
                num_workers=1,
                pin_memory=False)
        else:
            self.dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                sampler=sampler,
                num_workers=1,
                pin_memory=False)

        # set generator parameter file path
        if self.args.GAN_save_every_n_data > 0:
            if not os.path.exists(self.args.GAN_ckpt_path):
                os.makedirs(self.args.GAN_ckpt_path)

    def train(self):
        for i, data in enumerate(self.dataloader):
            tic = time.time()
            if self.args.dataset in ['MatterPort','ScanNet','KITTI']:
                # without gt
                partial, index = data
                gt = None
            else:
                # with gt
                gt, partial, index = data
                gt = gt.squeeze(0).cuda()
                
                ### simulate pfnet ball-holed data
                if self.args.inversion_mode == 'simulate_pfnet':
                    n_removal = 512
                    choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                    chosen = random.sample(choice,1)
                    dist = gt.add(-chosen[0].cuda())
                    dist_val = torch.norm(dist,dim=1)
                    top_dist, idx = torch.topk(dist_val, k=2048-n_removal)
                    partial = gt[idx]
            
            partial = partial.squeeze(0).cuda()
            # reset G for each new input
            self.model.reset_G(pcd_id=index.item())

            # set target and complete shape 
            # for ['reconstruction', 'jittering', 'morphing'], GT is used for reconstruction
            # else, GT is not involved for training
            if partial is None or self.args.inversion_mode in ['reconstruction', 'jittering', 'morphing','ball_hole','knn_hole']:
                self.model.set_target(gt=gt)
            else:
                self.model.set_target(gt=gt, partial=partial)
            
            # initialization
            self.model.select_z(select_y=False)
            # inversion
            self.model.run()
            toc = time.time()
            if self.rank == 0:
                print(i ,'out of',len(self.dataloader),'done in ',int(toc-tic),'s')
            
            if self.args.visualize:
                pcd_list = self.model.checkpoint_pcd
                flag_list = self.model.checkpoint_flags
                output_dir = self.args.save_inversion_path + '_visual'
                if self.args.inversion_mode == 'jittering':
                    output_stem = str(index.item())
                    draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(4,10))
                else:
                    output_stem = str(index.item())
                    draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(3,4))

            # save generator params as file
            if self.args.GAN_save_every_n_data > 0 and i % self.args.GAN_save_every_n_data == 0:
                GAN_ckpt_file_path = os.path.join(self.args.GAN_ckpt_path, self.args.GAN_ckpt_save) + str(i) + '_' +'.pt'
                torch.save({
                        'G_state_dict': self.model.G.state_dict(),
                        'D_state_dict': self.model.D.state_dict(),
                }, GAN_ckpt_file_path)
                print('Generator parameter saved at : ' + GAN_ckpt_file_path)
                

        
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<,rank',self.rank,'completed>>>>>>>>>>>>>>>>>>>>>>')


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
    
    