import argparse
from torch.backends import cudnn
import torch

#last revised: 1-April-2019

parser = argparse.ArgumentParser(description='PyTorch decoupled training')
parser.add_argument('--model', default='ResNet20',
                    help='models, ResNet18, ResNet50, ResNet101, ResNet20, ResNet56, ResNet110, WRN28_10 (default: ResNet18)')
parser.add_argument('--backprop', default=False,
                    help='disable local loss training')
parser.add_argument('--dg', default=True,
                    help='delayed gradients (default: True)')
parser.add_argument('--num-split', type=int, default=2,
                    help='the number of splits of the model (default: 2)')
parser.add_argument('--dataset', default='CIFAR10',
                    help='dataset, MNIST, KuzushijiMNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, STL10 or ImageNet (default: CIFAR10)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 350)')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='initial learning rate (default: 1e-1)')
parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[150, 225, 275],
                    help='decay learning rate at these milestone epochs (default: [200,300,350,375])')
parser.add_argument('--lr-decay-fact', type=float, default=0.1,
                    help='learning rate decay factor to use at milestone epochs (default: 0.25)')
parser.add_argument('--optim', default='SGD',
                    help='optimizer, adam, amsgrad or sgd (default: SGD)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='weight decay (default: 0.0)')
parser.add_argument('--lr-shrink', type=float, default=1,
                    help='shrinking the learning rate in earlier modules (default: 1)')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout after each nonlinearity (default: 0.1)')
parser.add_argument('--warm-up', default=False,
                    help='enable warming up for 3 epochs (default: False)')
parser.add_argument('--progress-bar', action='store_true', default=False,
                    help='show progress bar during training')
parser.add_argument('--save-model', default=False,
                    help='save the model (default: False)')
parser.add_argument('--comment', type=str, default=None,
                    help='comments')
parser.add_argument('--writer', default=True,
                    help='enable writer')
args = parser.parse_args()


if args.backprop:
    string_print = args.model + '_' + args.dataset + '_backprop_' + str(args.backprop) + '_batch_size_' + str(args.batch_size)+'num_epochs_' + str(args.epochs) + '_lr_' + str(args.lr) + '_mile_stones_' +\
               str(args.lr_decay_milestones) + '_rate_' + str(args.lr_decay_fact) + '_optim_' + args.optim + '_momentum_' + str(args.momentum) +\
               '_weight_decay_' + str(args.weight_decay) + '_dropout_' + str(args.dropout) + '_warm_up_' + \
               str(args.warm_up) + '_comment_' + str(args.comment) + '_writer_' + str(args.writer)
else:
    string_print = args.model + '_' + args.dataset + '_DG_' + str(args.dg) + '_num_split_' +\
                   str(args.num_split) + '_batch_size_' + str(args.batch_size)+'num_epochs_' + str(args.epochs) + '_lr_' + str(args.lr) + '_mile_stones_' +\
                   str(args.lr_decay_milestones) + '_rate_' + str(args.lr_decay_fact) + '_optim_' + args.optim + '_momentum_' + str(args.momentum) +\
                   '_weight_decay_' + str(args.weight_decay) + '_lr_shrink_' + str(args.lr_shrink) + '_dropout_' + str(args.dropout) + '_warm_up_' + \
                   str(args.warm_up) + '_comment_' + str(args.comment) + '_writer_' + str(args.writer)
 
print(string_print)