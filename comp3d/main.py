import os
import sys
import random
sys.path.append(os.getcwd())
import _init_paths
import json
import torch
from parse_args import parse_args
from AtlasNet import *
from AtlasNet_RI import *
from PointNetFCAE import *
from DippingNet import *
from train_utils import train, test, metrics, samples, set_seed, \
    resume, cache_pred, create_optimizer, model_at, parse_experiment, \
    check_overwrite, data_setup
import emd_module as emd
import math
from torch.utils.tensorboard import SummaryWriter

def save_model(args, epoch):
    torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': args.model.state_dict(),
                'optimizer' : args.optimizer.state_dict()},
            os.path.join(args.odir, 'models/model_%d.pth.tar' % (epoch + 1)))

def main():
    args = parse_args()
    eval(args.NET + '_setup')(args)
    set_seed(args.seed)

    summary_dir = os.path.join(args.odir, 'summary')
    if os.path.isdir(summary_dir):
        for f in os.listdir(summary_dir):
            os.remove(os.path.join(summary_dir, f))
    writer = SummaryWriter(summary_dir)

    # Create model and optimizer
    if args.resume or args.mode == 'eval':
        last_epoch, best_epoch, best_val_loss, num_params, \
            enc_params, dec_params = parse_experiment(args.odir)
        i = last_epoch
        if args.mode == 'eval':
            i = best_epoch
        args.resume = model_at(args, i)
        model, args.optimizer, stats = resume(args, i)
    else:
        check_overwrite(os.path.join(args.odir, 'trainlog.txt'))
        model = eval(args.NET + '_create_model')(args)
        args.optimizer = create_optimizer(args, model)
        stats = []
    print("Encoder params : %d" % (args.enc_params))
    print("Decoder params : %d" % (args.dec_params))

    print('Will save to ' + args.odir)
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    if not os.path.exists(args.odir + '/models'):
        os.makedirs(args.odir + '/models')
    with open(os.path.join(args.odir, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(["'"+a+"'" if (len(a)==0 or a[0]!='-') else a for a in sys.argv]))

    args.emd = emd.emdModule()
    args.model = model
    args.step = eval(args.NET + '_step')

    # Training loop
    epoch = args.start_epoch
    train_dataloader = data_setup(args, 'train', args.nworkers)
    if args.mode == 'train':
        for epoch in range(args.start_epoch, args.epochs):
            print('Epoch {}/{} ({}):'.format(epoch + 1, args.epochs, args.odir))

            loss = train(args, epoch, train_dataloader, writer)[0]

            if (epoch+1) % args.test_nth_epoch == 0 or epoch+1==args.epochs:
                loss_val = test('val', args, epoch, writer)[0]
                print('-> Train Loss: {}, \tVal loss: {}'.format(loss, loss_val))
                stats.append({'epoch': epoch + 1, 'loss': loss, 'loss_val': loss_val})
            else:
                loss_val = 0
                print('-> Train loss: {}'.format(loss))
                stats.append({'epoch': epoch + 1, 'loss': loss})

            if (epoch+1) % args.save_nth_epoch == 0 or epoch+1==args.epochs:
                with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
                    json.dump(stats, outfile)

                save_model(args, epoch)
            if (epoch+1) % args.test_nth_epoch == 0 and epoch+1 < args.epochs:
                split = 'val'
                predictions = samples(split, args, 20)
                cache_pred(predictions, split, args)
                metrics(split, args, epoch)

                # visualize points
                rand_class = random.choice(list(predictions.keys()))
                inputs, outputs, gts = predictions[rand_class]
                outputs = torch.from_numpy(outputs)
                rand_index = random.randrange(inputs.shape[0])
                writer.add_mesh('gt', vertices=gts[rand_index].unsqueeze(0), global_step=epoch)
                writer.add_mesh('input', vertices=inputs[rand_index].unsqueeze(0), global_step=epoch)
                writer.add_mesh('output', vertices=outputs[rand_index].unsqueeze(0), global_step=epoch)

            if math.isnan(loss): break

        if len(stats)>0:
            with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
                json.dump(stats, outfile)

    # Evaluation & Write merics
    split = 'val'
    predictions = samples(split, args, 20)
    cache_pred(predictions, split, args)
    metrics(split, args, epoch)

if __name__ == "__main__":
    main()