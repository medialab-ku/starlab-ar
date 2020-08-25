import torch
import time
import numpy as np
import json
import logging
import random
from tqdm import tqdm
import torch.optim as optim
import torchnet as tnt
import os
from collections import defaultdict
import h5py
from shapenet import ShapeNet
from data_utils import Transform


def check_overwrite(fname):
    if os.path.isfile(fname):
        valid = ['y', 'yes', 'no', 'n']
        inp = None
        while inp not in valid:
            inp = input(
                '%s already exists. Do you want to overwrite it? (y/n)'
                % fname)
            if inp.lower() in ['n', 'no']:
                raise Exception('Please create new experiment.')


def parse_experiment(odir):
    stats = json.loads(open(odir + '/trainlog.txt').read())
    valloss = [k['loss_val'] for k in stats if 'loss_val' in k.keys()]
    epochs = [k['epoch'] for k in stats if 'loss_val' in k.keys()]
    last_epoch = max(epochs)
    idx = np.argmin(valloss)
    best_val_loss = float('%.6f' % (valloss[idx]))
    best_epoch = epochs[idx]
    val_results = odir + '/results_val_%d' % (best_epoch)
    val_results = open(val_results).readlines()
    first_line = val_results[0]
    num_params = int(first_line.rstrip().split(' ')[-1])
    enc_params = int(val_results[1].rstrip().split(' ')[-1])
    dec_params = int(val_results[2].rstrip().split(' ')[-1])

    return last_epoch, best_epoch, best_val_loss, num_params, enc_params, dec_params


def model_at(args, i):
    return os.path.join(args.odir, 'models/model_%d.pth.tar' % (i))


def resume(args, i):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    if 'args' not in list(checkpoint.keys()):  # Pre-trained model?
        r_args = args
        model = eval(args.NET + '_create_model')(r_args)  # use original arguments, architecture can't change
        optimizer = create_optimizer(args, model)
        model.load_state_dict(checkpoint)
        checkpoint['epoch'] = 0
        args.start_epoch = None
    else:
        r_args = checkpoint['args']
        model = eval(args.NET + '_create_model')(r_args)  # use original arguments, architecture can't change
        args.nparams = r_args.nparams
        args.enc_params = r_args.enc_params
        args.dec_params = r_args.dec_params
        optimizer = create_optimizer(args, model)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']

    if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups: group['initial_lr'] = args.lr
    stats = json.loads(open(os.path.join(args.odir, 'trainlog.txt')).read())
    return model, optimizer, stats


def create_optimizer(args, model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr, rho=0.9, epsilon=1e-6, weight_decay=args.wd)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr, alpha=0.99, epsilon=1e-8, weight_decay=args.wd)
    return optimizer


def set_seed(seed, cuda=True):
    """ Sets seeds in all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def data_setup(args, split, num_workers):
    transform = Transform(
        normal=False,
        rotate=[180, 180, 180],
        random=True
    )
    if (args.rotaug):
        dataset = ShapeNet(args, split, transform)
    else:
        dataset = ShapeNet(args, split, None)
    return torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True, num_workers=num_workers)


def train(args, epoch, data_loader, writer):
    """ Trains for one epoch """
    print("Training....")
    args.model.train()

    N = len(data_loader.dataset)
    Nb = int(N / args.batch_size)
    if Nb * args.batch_size < N:
        Nb += 1

    meters = []
    # Names of losses used
    lnm = ['train_loss']
    Nl = len(lnm)
    for i in range(Nl):
        meters.append(tnt.meter.AverageValueMeter())
    t0 = time.time()

    # iterate over dataset in batches
    dataloader_iter = iter(data_loader)
    for bidx in tqdm(range(Nb)):
        gts, partials, metas = next(dataloader_iter)
        t_loader = 1000 * (time.time() - t0)
        t0 = time.time()

        args.optimizer.zero_grad()

        results = args.step(args, gts.float().cuda(), partials.float().cuda())
        loss = results['loss']
        dist1 = results['dist1']
        dist2 = results['dist2']
        emd_cost = results['emd_cost']
        outputs = results['outputs']
        loss.backward()

        args.optimizer.step()

        t_trainer = 1000 * (time.time() - t0)
        # List of losses used - corresponding to loss names above
        losses = [loss, ]
        for ix, l in enumerate(losses):
            meters[ix].add(l.item())

        if (bidx % 50) == 0:
            prt = 'Train '
            plot_dict = {}
            for ix in range(Nl):
                prt += '%s %f, ' % (lnm[ix], losses[ix].item())
                plot_dict[lnm[ix]] = losses[ix].item()
            writer.add_scalar('train', plot_dict, epoch * Nb + bidx)
            prt += 'Loader %f ms, Train %f ms.\n' % (t_loader, t_trainer)
            print(prt)
        logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss.item(), t_loader, t_trainer)
        t0 = time.time()

    return [meters[ix].value()[0] for ix in range(Nl)]


def test(split, args, epoch, writer):
    """ Evaluated model on test set """
    print("Testing....")
    args.model.eval()

    test_dataloader = data_setup(args, 'test', num_workers=1)

    meters = []
    lnm = ['val_loss', ]
    Nl = len(lnm)
    for i in range(Nl):
        meters.append(tnt.meter.AverageValueMeter())

    t0 = time.time()

    N = len(test_dataloader.dataset)
    Nb = int(N / args.batch_size)
    if Nb * args.batch_size < N:
        Nb += 1
    # iterate over dataset in batches
    dataloader_iter = iter(test_dataloader)
    for bidx in tqdm(range(Nb)):
        gts, partials, metas = next(dataloader_iter)
        t_loader = 1000 * (time.time() - t0)
        t0 = time.time()

        results = args.step(args, gts.float().cuda(), partials.float().cuda())
        loss = results['loss']
        dist1 = results['dist1']
        dist2 = results['dist2']
        emd_cost = results['emd_cost']
        outputs = results['outputs']

        t_trainer = 1000 * (time.time() - t0)
        losses = [loss, ]
        for ix, l in enumerate(losses):
            meters[ix].add(l)

        logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss, t_loader, t_trainer)
        t0 = time.time()

    plot_dict = {}
    for ix, m in enumerate(meters):
        plot_dict[lnm[ix]] = m.value()[0]
    writer.add_scalar('val', plot_dict, epoch)

    return [meters[ix].value()[0] for ix in range(Nl)]


def samples(split, args, N):
    print("Sampling ...")
    args.model.eval()

    collected = defaultdict(list)
    predictions = {}
    class_samples = defaultdict(int)
    if hasattr(args, 'classmap'):
        for val in args.classmap:
            class_samples[val[0]] = 0
    else:
        count = 0

    val_dataloader = data_setup(args, split, num_workers=1)
    L = len(val_dataloader.dataset)
    Nb = int(L / args.batch_size)
    if Nb * args.batch_size < L:
        Nb += 1

    # iterate over dataset in batches
    dataloader_iter = iter(val_dataloader)
    for bidx in tqdm(range(Nb)):
        gts, partials, metas = next(dataloader_iter)
        run_net = False
        for idx in range(gts.shape[0]):
            if hasattr(args, 'classmap'):
                fname = metas[0][idx][:metas[0][idx].rfind('.')]
                synset = fname.split('/')[-2]
                if class_samples[synset] <= N:
                    run_net = True
                    break
            elif count <= N:
                run_net = True
                break
        if run_net:
            results = args.step(args, gts.float().cuda(), partials.float().cuda())
            loss = results['loss']
            dist1 = results['dist1']
            dist2 = results['dist2']
            emd_cost = results['emd_cost']
            outputs = results['outputs']

            for idx in range(gts.shape[0]):
                if hasattr(args, 'classmap'):
                    fname = metas[0][idx][:metas[0][idx].rfind('.')]
                    synset = fname.split('/')[-2]
                    if class_samples[synset] > N:
                        continue
                    class_samples[synset] += 1
                else:
                    fname = str(bidx)
                    if count > N:
                        break
                    count += 1
                collected[fname].append((outputs[idx:idx + 1, ...], gts[idx:idx + 1, ...],
                                         partials[idx:idx + 1, ...]))

    for fname, lst in collected.items():
        o_cpu, t_cpu, inp = list(zip(*lst))
        o_cpu = o_cpu[0]
        t_cpu, inp = t_cpu[0], inp[0]
        predictions[fname] = (inp, o_cpu, t_cpu)  # input, output, gt (tensor)
    return predictions


def batch_instance_metrics(args, dist1, dist2):
    dgen = np.mean(dist1, 1) + np.mean(dist2, 1)
    return dgen


def metrics(split, args, epoch=0):
    print("Metrics ....")
    db_name = split
    args.model.eval()
    Gerror = defaultdict(list)
    Gerror_emd = defaultdict(list)
    val_dataloader = data_setup(args, split, num_workers=1)
    N = len(val_dataloader.dataset)
    Nb = int(N / args.batch_size)
    if Nb * args.batch_size < N:
        Nb += 1
    # iterate over dataset in batches
    dataloader_iter = iter(val_dataloader)
    for bidx in tqdm(range(Nb)):
        gts, partials, metas = next(dataloader_iter)
        results = args.step(args, gts.float().cuda(), partials.float().cuda())
        loss = results['loss']
        dist1 = results['dist1']
        dist2 = results['dist2']
        emd_cost = results['emd_cost']
        outputs = results['outputs']
        dgens = batch_instance_metrics(args, dist1, dist2)
        for idx in range(gts.shape[0]):
            if hasattr(args, 'classmap'):
                classname = args.classmap[metas[0][idx].split('/')[0]]
            Gerror[classname].append(dgens[idx])
            Gerror_emd[classname].append(emd_cost[idx])

    Gm_errors = []
    Gm_errors_emd = []
    outfile = args.odir + '/results_%s_%d' % (db_name, epoch + 1)
    if args.mode == 'eval':
        outfile = args.odir + '/eval_%s_%d' % (db_name, epoch)
    print("Saving results to %s ..." % (outfile))
    with open(outfile, 'w') as f:
        f.write('#ParametersTotal %d\n' % (args.nparams))
        f.write('#ParametersEncoder %d\n' % (args.enc_params))
        f.write('#ParametersDecoder %d\n' % (args.dec_params))
        for classname in list(Gerror.keys()):
            Gmean_error_emd = np.mean(Gerror_emd[classname])
            Gm_errors_emd.append(Gmean_error_emd)
            Gmean_error = np.mean(Gerror[classname])
            Gm_errors.append(Gmean_error)
            f.write('%s Generator_emd %.6f\n' % (classname, Gmean_error_emd))
            f.write('%s Generator_dist %.6f\n' % (classname, Gmean_error))
        f.write('Generator Class Mean EMD %.6f\n' % (np.mean(Gm_errors_emd)))
        f.write('Generator Class Mean DIST %.6f\n' % (np.mean(Gm_errors)))


def cache_pred(predictions, db_name, args):
    with h5py.File(os.path.join(args.odir, 'inp_' + db_name + '.h5'), 'w') as hf:
        with h5py.File(os.path.join(args.odir, 'predictions_' + db_name + '.h5'), 'w') as hf1:
            with h5py.File(os.path.join(args.odir, 'gt_' + db_name + '.h5'), 'w') as hf2:
                for fname, o_cpu in predictions.items():
                    hf.create_dataset(name=fname, data=o_cpu[0])
                    hf1.create_dataset(name=fname, data=o_cpu[1])
                    hf2.create_dataset(name=fname, data=o_cpu[2])
