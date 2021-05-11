import os
from copy import deepcopy

import numpy as np
import torch
from torch.autograd import Variable

import cmpl.shinv_config as cfg
from cmpl.shinv_gan import Generator, Discriminator
from cmpl.shinv_util import LRScheduler, DiscriminatorLoss, distChamfer, distChamfer_raw
import cmpl.shinv_reg as registration


class ShapeInversion(object):

    def __init__(self):

        self.rank, self.world_size = 0, 1

        # init seed for static masks: ball_hole, knn_hole, voxel_mask
        self.iterations = cfg.ITERATION
        self.G_lrs = cfg.LEARNING_RATES_G
        self.z_lrs = cfg.LEARNING_RATES_Z
        self.select_num = cfg.SELECT_NUM_Z

        self.loss_log = []

        # create model
        self.G = Generator(features=cfg.TREE_GAN_FEATS_G, degrees=cfg.TREE_GAN_DEGREES).cuda()
        self.D = Discriminator(features=cfg.TREE_GAN_FEATS_D).cuda()

        self.G.optim = torch.optim.Adam(
            [{'params': self.G.get_params(i)}
             for i in range(7)],
            lr=self.G_lrs[0],
            betas=(0, 0.99),
            weight_decay=0,
            eps=1e-8)
        self.z = torch.zeros((1, 1, 96)).normal_().cuda()
        self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam([self.z], lr=self.z_lrs[0], betas=(0, 0.99))

        # load weights
        checkpoint = torch.load(os.path.join(os.path.dirname(__file__), cfg.CHECK_POINT_PATH), map_location=cfg.DEVICE)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])

        self.G.eval()
        if self.D is not None:
            self.D.eval()
        self.G_weight = deepcopy(self.G.state_dict())

        # prepare latent variable and optimizer
        self.G_scheduler = LRScheduler(self.G.optim)
        self.z_scheduler = LRScheduler(self.z_optim)

        # loss functions
        self.ftr_net = self.D
        self.criterion = DiscriminatorLoss()

        # for visualization
        self.checkpoint_pcd = []  # to save the staged checkpoints
        self.checkpoint_flags = []  # plot subtitle

        if len(cfg.LOSS_WEIGHT_D) == 1:
            self.w_D_loss = cfg.LOSS_WEIGHT_D * len(self.G_lrs)
        else:
            self.w_D_loss = cfg.LOSS_WEIGHT_D

    def reset_G(self):
        """
        to call in every new fine_tuning
        before the 1st one also okay
        """
        self.G.load_state_dict(self.G_weight, strict=False)
        self.G.eval()
        self.checkpoint_pcd = []  # to save the staged checkpoints
        self.checkpoint_flags = []

    def set_target(self, gt=None, partial=None):
        '''
        set target
        '''
        if gt is not None:
            self.gt = gt.unsqueeze(0)
            # for visualization
            self.checkpoint_flags.append('GT')
            self.checkpoint_pcd.append(self.gt)
        else:
            self.gt = None

        if partial is not None:
            self.target = partial.unsqueeze(0)
        else:
            self.target = self.pre_process(self.gt, stage=-1)
        # for visualization
        self.checkpoint_flags.append('target')
        self.checkpoint_pcd.append(self.target)

    def run(self, ith=-1):
        loss_dict = {}
        curr_step = 0
        count = 0
        for stage, iteration in enumerate(self.iterations):

            for i in range(iteration):
                curr_step += 1
                # setup learning rate
                self.G_scheduler.update(curr_step, self.G_lrs[stage])
                self.z_scheduler.update(curr_step, self.z_lrs[stage])

                # forward
                self.z_optim.zero_grad()

                tree = [self.z]
                x = self.G(tree)
                x = registration.run(x, self.target)

                # masking
                x_map = self.pre_process(x, stage=stage)

                ### compute losses
                ftr_loss = self.criterion(self.ftr_net, x_map, self.target)

                dist1, dist2, _, _ = distChamfer(x_map, self.target)
                cd_loss = dist1.mean() + dist2.mean()

                # nll corresponds to a negative log-likelihood loss
                nll = self.z ** 2 / 2
                nll = nll.mean()

                ### loss
                loss = ftr_loss * self.w_D_loss[stage] + nll * cfg.LOSS_WEIGHT_NLL + cd_loss

                # backward
                loss.backward()
                self.z_optim.step()

            # save checkpoint for each stage
            self.checkpoint_flags.append('s_' + str(stage) + ' x')
            self.checkpoint_pcd.append(x)
            self.checkpoint_flags.append('s_' + str(stage) + ' x_map')
            self.checkpoint_pcd.append(x_map)

            # test only for each stage
            if self.gt is not None:
                dist1, dist2, _, _ = distChamfer(x, self.gt)
                test_cd = dist1.mean() + dist2.mean()

        if self.gt is not None:
            loss_dict = {
                'ftr_loss': np.asscalar(ftr_loss.detach().cpu().numpy()),
                'nll': np.asscalar(nll.detach().cpu().numpy()),
                'cd': np.asscalar(test_cd.detach().cpu().numpy()),
            }
            self.loss_log.append(loss_dict)

        ### save point clouds
        x = x
        x_np = x[0].detach().cpu().numpy()
        x_map_np = x_map[0].detach().cpu().numpy()
        target_np = self.target[0].detach().cpu().numpy()
        if self.gt is not None:
            gt_np = self.gt[0].detach().cpu().numpy()
        return x_np

    def select_z(self, select_y=False):
        with torch.no_grad():
            if self.select_num == 0:
                self.z.zero_()
                return
            elif self.select_num == 1:
                self.z.normal_()
                return
            z_all, y_all, loss_all = [], [], []
            for i in range(self.select_num):
                z = torch.randn(1, 1, 96).cuda()
                tree = [z]
                with torch.no_grad():
                    x = self.G(tree)
                    x = registration.run(x, self.target)
                ftr_loss = self.criterion(self.ftr_net, x, self.target)
                z_all.append(z)
                loss_all.append(ftr_loss.detach().cpu().numpy())

            loss_all = np.array(loss_all)
            idx = np.argmin(loss_all)

            self.z.copy_(z_all[idx])
            if select_y:
                self.y.copy_(y_all[idx])

            x = self.G([self.z])
            x = registration.run(x, self.target)

            # visualization
            if self.gt is not None:
                x_map = self.pre_process(x, stage=-1)
                dist1, dist2, _, _ = distChamfer(x, self.gt)
                cd_loss = dist1.mean() + dist2.mean()

                self.checkpoint_flags.append('init x')
                self.checkpoint_pcd.append(x)
                self.checkpoint_flags.append('init x_map')
                self.checkpoint_pcd.append(x_map)
            return z_all[idx]

    def pre_process(self, pcd, stage=-1):
        """
        transfer a pcd in the observation space:
        with the following mask_type:
            none: for ['reconstruction', 'jittering', 'morphing']
            ball_hole, knn_hole: randomly create the holes from complete pcd, similar to PF-Net
            voxel_mask: baseline in ShapeInversion
            tau_mask: baseline in ShapeInversion
            k_mask: proposed component by ShapeInversion
        """
        pcd_map = self.k_mask(self.target, pcd, stage)
        return pcd_map

    def k_mask(self, target, x, stage=-1):
        """
        masking based on CD.
        target: (1, N, 3), partial, can be < 2048, 2048, > 2048
        x: (1, 2048, 3)
        x_map: (1, N', 3), N' < 2048
        x_map: v1: 2048, 0 masked points
        """
        stage = max(0, stage)
        knn = cfg.MASKS_K[stage]
        if knn == 1:
            cd1, cd2, argmin1, argmin2 = distChamfer(target, x)
            idx = torch.unique(argmin1).type(torch.long)
        else:
            # dist_mat shape (B, 2048, 2048), where B = 1
            dist_mat = distChamfer_raw(target, x)
            # indices (B, 2048, k)
            val, indices = torch.topk(dist_mat, k=knn, dim=2, largest=False)
            # union of all the indices
            idx = torch.unique(indices).type(torch.long)

        mask_tensor = torch.zeros(2048, 1)
        mask_tensor[idx] = 1
        mask_tensor = mask_tensor.cuda().unsqueeze(0)
        x_map = torch.mul(x, mask_tensor)

        return x_map
