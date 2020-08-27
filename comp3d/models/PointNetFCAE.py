import _init_paths
import torch
from torch.autograd import Variable
import torch.nn as nn
from common import PointNetfeat
from dist_chamfer import chamferDist as chamfer
import numpy as np

def PointNetFCAE_setup(args):
    args.odir = 'results/%s/PointNetFCAE_%s' % (args.dataset, args.dist_fun)
    args.odir += '_npts%d' % (args.npts)
    args.odir += '_code%d' % (args.code_nfts)
    args.odir += '_lr%.4f' % (args.lr)
    args.odir += '_' + args.optim
    args.odir += '_B%d' % (args.batch_size)

def PointNetFCAE_create_model(args):
    """ Creates model """
    model = nn.DataParallel(PointNetFCAE(args, args.npts))
    args.enc_params = sum([p.numel() for p in model.module.encoder.parameters()])
    args.dec_params = sum([p.numel() for p in model.module.decoder.parameters()])
    args.nparams = sum([p.numel() for p in model.module.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    print(model)
    model.cuda()
    return model

def PointNetFCAE_step(args, targets_in, clouds_data):
    #targets = Variable(torch.from_numpy(targets_in), requires_grad=False).float().cuda()
    targets = Variable(torch.cuda.FloatTensor(targets_in), requires_grad=False).float().cuda()
    targets = targets.contiguous()
    inp = Variable(torch.cuda.FloatTensor(clouds_data), requires_grad=True).float().cuda()
    inp = inp.transpose(2, 1).contiguous()
    outputs = args.model(inp)[0]
    targets = targets.transpose(2, 1).contiguous()
    N = targets.size()[1]
    dist1, dist2 = eval(args.dist_fun)()(outputs, targets)
    # EMD not working in pytorch (see pytorch-setup.md)
    #emd_cost = args.emd_mod(outputs[:, 0:N,:], targets)/N
    #emd_cost = emd_cost.data.cpu().numpy()
    emd_cost = 0#args.emd_mod(outputs[:, 0:N, :], targets)/N
    emd_cost = np.array([0]*args.batch_size)#emd_cost.data.cpu().numpy()

    loss = torch.mean(dist1) + torch.mean(dist2)
    dist1 = dist1.data.cpu().numpy()
    dist2 = dist2.data.cpu().numpy()

    if args.model.training:
        return {'loss': loss, 'dist1': dist1, 'dist2': dist2, 'emd_cost': emd_cost,
                'outputs': outputs.data.cpu().numpy()}
    else:
        return {'loss': loss.item(), 'dist1': dist1, 'dist2': dist2, 'emd_cost': emd_cost,
                'outputs': outputs.data.cpu().numpy()}

class PointNetFCAE(nn.Module):
    """ PointNet Encoder, MLP Decoder"""
    def __init__(self, args, num_points=2048, output_channels=3):
        super(PointNetFCAE, self).__init__()
        self.args = args
        self.num_points = num_points
        self.output_channels = output_channels
        self.encoder = nn.Sequential(
            PointNetfeat(args, num_points, global_feat=True, trans=False),
            nn.Linear(args.code_nfts, args.code_nfts),
            nn.BatchNorm1d(args.code_nfts),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.code_nfts, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_channels*num_points),
            nn.Tanh()
        )

    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        x = x.view(-1, self.output_channels, self.num_points)
        x = x.transpose(2, 1).contiguous()

        return x, code


def test_net():
    from parse_args import parse_args
    args = parse_args()
    PointNetFCAE_setup(args)
    args.model = PointNetFCAE_create_model(args)
    gts = torch.randn(args.batch_size, args.ngtpts, 3).cuda()
    inputs = torch.randn(args.batch_size, args.inpts, 3).cuda()
    loss, dist1, dist2, emd_cost, outputs = PointNetFCAE_step(args, gts, inputs)
    print ('loss', loss, 'dist1', dist1, 'dist2', dist2, 'emd_cost', emd_cost)


if __name__ == '__main__':
    test_net()