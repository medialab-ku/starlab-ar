import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0,os.path.join(os.path.dirname(__file__), ".."))
from data_utils import load_h5, load_csv, pad_cloudN
from vis import plot_pcds
import torch.utils.data as data

class ShapeNet(data.Dataset):
    def __init__(self, args, split='train', transform=None):
        self.args=args
        self.split = split
        self.transform = transform
        args.DATA_PATH = 'data/%s' % (args.dataset)
        classmap = load_csv(args.DATA_PATH + '/synsetoffset2category.txt')
        args.classmap = {}
        for i in range(classmap.shape[0]):
            args.classmap[str(classmap[i][1]).zfill(8)] = classmap[i][0]

        self.data_paths = sorted([os.path.join(args.DATA_PATH, split, 'partial', k.rstrip()+ '.h5') for k in open(args.DATA_PATH + '/%s.list' % (split)).readlines()])
        N = int(len(self.data_paths)/args.batch_size)*args.batch_size
        self.data_paths = self.data_paths[0:N]

    def __getitem__(self, index):
        fname = self.data_paths[index]
        partial = load_h5(fname)
        if self.split == 'test':
            gt = partial
        else:
            gt = load_h5(fname.replace('partial', 'gt'))
        if self.transform:
            partial, gt, Rt = self.transform(partial, gt)
        partial  = pad_cloudN(partial, self.args.inpts)
        meta = ['{}.{:d}'.format('/'.join(fname.split('/')[-2:]),0),]
        return gt, partial, meta

    def __len__(self):
        return len(self.data_paths)