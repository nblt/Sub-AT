import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pickle
import random
import resnet
from utils import get_datasets, get_model, epoch_adversarial, print_args, grad_align_loss, trades_loss, epoch_adversarial_PGD50, AutoAttack

from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd

def set_seed(seed=233): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print ('P-SGD')

parser = argparse.ArgumentParser(description='P(+)-SGD in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    help='model architecture (default: resnet32)')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--n_components', default=40, type=int, metavar='N',
                    help='n_components for PCA') 
parser.add_argument('--params_start', default=0, type=int, metavar='N',
                    help='which epoch start for PCA') 
parser.add_argument('--params_end', default=51, type=int, metavar='N',
                    help='which epoch end for PCA') 
parser.add_argument('--lr', default=1, type=float, metavar='N',
                    help='lr for PSGD') 
parser.add_argument('--gamma', default=0.9, type=float, metavar='N',
                    help='gamma for momentum')
parser.add_argument('--randomseed', 
                    help='Randomseed for training and initialization',
                    type=int, default=0)

########################## attack setting ##########################
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--train_eps', default=8., type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2., type=float, help='step size of attack during training')
parser.add_argument('--train_randinit', action='store_false', help='randinit usage flag (default: on)')
parser.add_argument('--test_eps', default=8., type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2., type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')

parser.add_argument('--gradalign', '--ga', action='store_true', help='gradalign term (default: False)')
parser.add_argument('--glambda', default=0.2, type=float, help='lambda for gradalign')
parser.add_argument('--tradeloss', '--trade', action='store_true', help='use tradeloss (default: False)')
parser.add_argument('--lr_max', '--learning-rate-max', default=0.3, type=float,
                    metavar='cLR', help='maximum learning rate for cyclic learning rates')

args = parser.parse_args()
set_seed(args.randomseed)

    
# Define model
model = torch.nn.DataParallel(get_model(args))
model.cuda()

model.load_state_dict(torch.load('save_PreActResNet18_CIFAR100_eps8/best.pt'))

# AutoAttack(model, norm='Linf', dataset='CIFAR100')

AutoAttack(model, norm='L2', dataset='CIFAR100', epsilon=128)

