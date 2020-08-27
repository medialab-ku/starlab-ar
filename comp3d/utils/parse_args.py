import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='3D Object Point Cloud Completion')

    # Optimization arguments
    parser.add_argument('--optim', default='adagrad', help='Optimizer: sgd|adam|adagrad|adadelta|rmsprop')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')

    #Training/Testing arguments
    parser.add_argument('--mode', type=str, default='train', help='train / eval')
    parser.add_argument('--resume', type=int, default=0, help='If 1, resume training')
    parser.add_argument('--dist_fun', default='chamfer', help='Point Cloud Distance used in training')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--nworkers', default=4, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--emd_eps', default=0.005, type=float, help='One of parameters for MSN emd module')
    parser.add_argument('--emd_iters', default=50, type=int, help='One of parameters for MSN emd module')


    # Model
    parser.add_argument('--NET', default='AtlasNet', help='Network used')
    parser.add_argument('--code_nfts', default=1024, type=int, help='Encoder output feature size')

    # Model
    parser.add_argument('--nb_primitives', default=4, type=int, help='Number of primitives for AtlasNet')


    # Point Cloud Distance function

    # Dataset
    parser.add_argument('--dataset', default='shapenet', help='Dataset name: shapenet')
    parser.add_argument('--npts', default=2048, type=int, help='Number of output points generated')
    parser.add_argument('--nsauce', default=2048, type=int, help='Number of sauce points generated')
    parser.add_argument('--inpts', default=1024, type=int, help='Number of input points')
    parser.add_argument('--ngtpts', default=2048, type=int, help='Number of ground-truth points')
    parser.add_argument('--rotaug', default=0, type=int, help='If 1, rotation augmentation enabled')

    args = parser.parse_args()
    args.start_epoch = 0
    if args.dataset == 'shapenet16384':
        args.ngtpts = 16384

    return args