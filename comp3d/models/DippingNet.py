import _init_paths
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from ShellNet import *
from common import weights_init
from dist_chamfer import chamferDist as chamfer
from data_utils import make_mask_gt

import open3d as o3d


def sample_uniform_ball3d_1(num):
    pts = []
    for i in range(num):
        s = np.random.normal(0, 1)
        t = np.random.normal(0, 1)
        u = np.random.normal(0, 1)
        v = np.random.normal(0, 1)
        w = np.random.normal(0, 1)
        norm = (s * s + t * t + u * u + v * v + w * w) ** 0.5
        x, y, z = u / norm, v / norm, w / norm
        pts.append([x, y, z])
    return torch.cuda.FloatTensor(pts)


def sample_uniform_ball3d_2(num):
    rand = torch.randn(num, 5)
    norm = torch.norm(rand, dim=1)
    norm = norm.reshape(num, 1)
    pts = rand[:, [2, 3, 4]] / norm
    return pts


def DippingNet_setup(args):
    args.odir = 'results/%s/DippingNet_%s' % (args.dataset, args.dist_fun)
    args.odir += '_nsauce%d' % (args.nsauce)
    args.odir += '_lr%.4f' % (args.lr)
    args.odir += '_' + args.optim
    args.odir += '_B%d' % (args.batch_size)
    args.odir += '_rotaug' if args.rotaug else ''
    args.classmap = ''

    # generate sphere cloud
    args.sauce = sample_uniform_ball3d_2(args.nsauce) * 0.5
    print('input', args.inpts, 'sauce', args.nsauce)


def DippingNet_create_model(args):
    """ Creates model """
    model = nn.DataParallel(
        DippingNet(args, in_points=args.inpts, bottleneck_size=args.code_nfts))
    args.enc_params = sum([p.numel() for p in model.module.encoder.parameters()])
    args.dec_params = sum([p.numel() for p in model.module.decoder.parameters()])
    args.nparams = sum([p.numel() for p in model.module.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    print(model)
    model.cuda()
    model.apply(weights_init)
    return model


def DippingNet_step(args, gts, inputs):
    """
    probs
        (B, S, 1)
    merges
        (B, N + S, 3)
    """
    probs, merges = args.model.forward(inputs, args.sauce)
    print (probs)
    gts = gts.cuda()

    masks_gt = make_mask_gt(args.sauce.cuda(), gts, 1)
    bce = torch.nn.BCELoss()(probs, masks_gt)

    B, N_in, _ = inputs.size()
    _, N_gt, _ = gts.size()
    S = args.nsauce

    mask = probs.round().bool()
    #print (torch.sum(mask.long()))
    pad_mask = torch.ones(B, N_in, 1).cuda().bool()
    pad_gts = torch.zeros(B, N_in + S - N_gt, 3).cuda()
    # (bs, N + S, 1)
    mask_padded = torch.cat((pad_mask, mask), dim=1)
    # (bs, N + S)
    mask_padded = mask_padded.squeeze(2)
    # (bs, N + S, 3)
    gts_padded = torch.cat((pad_gts, gts), dim=1)

    # retaining input points, and masking new points
    # set false-masked point with one of gt's points so that its chamfer distance becomes 0
    # More thankfully, completion3d's implementation of chamfer does not divide the distance sum with n
    merges_masked1 = merges.clone()
    merges_masked1[mask_padded == False] = gts_padded[mask_padded == False]

    # send false-masked point to far away so that it won't be chosen while calculating distances
    # merges_masked2 = merges.clone()
    # merges_masked2[mask_padded == False] = torch.ones(1, 3).cuda() * 100

    # make output points
    # set false-masked points (0, 0, 0)
    outputs = merges.clone()
    outputs[mask_padded == False] = torch.zeros(1, 3).cuda()

    dist1, dist2 = eval(args.dist_fun)()(merges_masked1, gts)

    loss = torch.mean(dist1) + bce
    dist1 = dist1.data.cpu().numpy()
    dist2 = dist2.data.cpu().numpy()

    emd_cost = np.array([0] * args.batch_size)

    if args.model.training:
        return loss, dist1, dist2, emd_cost, outputs.data.cpu().numpy()
    else:
        return loss.item(), dist1, dist2, emd_cost, outputs.data.cpu().numpy()


class PointClsCon(nn.Module):
    def __init__(self, bottleneck_size=1027):
        self.bottleneck_size = bottleneck_size
        super(PointClsCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size / 2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size / 2), int(self.bottleneck_size / 4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size / 4), 1, 1)

        self.sg = nn.Sigmoid()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.sg(self.conv4(x))
        return x


class DippingNet(nn.Module):
    def __init__(self, args, in_points=1024, bottleneck_size=1024):
        super(DippingNet, self).__init__()
        self.in_points = in_points
        self.bottleneck_size = bottleneck_size
        self.encoder = nn.Sequential(
            ShellNet_RI_Feature(in_points, out_dim=bottleneck_size, conv_scale=1, dense_scale=1, has_bn=True,
                                global_feat=True),
            nn.Linear(bottleneck_size, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = PointClsCon(bottleneck_size=3 + self.bottleneck_size)

    def forward(self, x, sauce):
        batch_size = x.size(0)
        nsauce = sauce.size(0)
        inputs = x
        x = self.encoder(x)

        # (bs, feat) -> (bs, nsauce, feat)
        x = x.repeat_interleave(nsauce, dim=0)
        x = x.reshape(batch_size, nsauce, -1)

        # (nsauce, 3) -> (bs, nsauce, 3)
        sauce = Variable(sauce, requires_grad=True)
        sauce = sauce.repeat(batch_size, 1)
        sauce = sauce.reshape(batch_size, nsauce, -1)

        # (bs, nsauce, feat) -> (bs, nsauce, 3 + feat)
        x = torch.cat((sauce, x), dim=2)

        # (bs, nsauce, 3 + feat) -> (bs, 3 + feat, nsauce)
        x = x.transpose(2, 1).contiguous()
        # (bs, 3 + feat, nsauce) -> (bs, 1, nsauce)
        y = self.decoder(x)
        # (bs, 1, nsauce) -> (bs, nsauce, 1)
        y = y.transpose(1, 2).contiguous()

        # (bs, inpts + nsauce, 3)
        merged_pts = torch.cat((inputs, sauce), dim=1)

        return y, merged_pts


def test_sauce():
    sauce = sample_uniform_ball3d_2(1024)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sauce.numpy())
    o3d.visualization.draw_geometries([pcd])


def test_net():
    from parse_args import parse_args
    args = parse_args()
    DippingNet_setup(args)
    args.model = DippingNet_create_model(args)
    gts = torch.randn(args.batch_size, args.ngtpts, 3)
    inputs = torch.randn(args.batch_size, args.inpts, 3)
    loss, dist1, dist2, emd_cost, outputs = DippingNet_step(args, gts, inputs)
    print ('loss', loss, 'dist1', dist1, 'dist2', dist2, 'emd_cost', emd_cost)


if __name__ == '__main__':
    test_net()
