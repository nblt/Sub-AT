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
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.autograd import Variable
import torch.autograd as autograd


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pickle
import random
import resnet
import wideresnet

from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd

def get_model_param_vec(model):
    """Return model parameters as a vector"""
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)

def epoch_adversarial(loader, model, args, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(), eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(), eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    model.eval()
    for X,y in loader:
        X,y = X.cuda(), y.cuda()
        # delta = attack(model, X, y, **kwargs)
        #adv samples
        input_adv = adversary.perturb(X, y)
        yp = model(input_adv)
        loss = nn.CrossEntropyLoss()(yp,y)
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    return 1 - total_err / len(loader.dataset), total_loss / len(loader.dataset)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon=8./255, alpha=2./255, attack_iters=50, restarts=10,
               norm='l_inf', early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    upper_limit, lower_limit = 1,0
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.uniform_(-0.5,0.5).renorm(p=2, dim=1, maxnorm=epsilon)
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def epoch_adversarial_PGD50(test_loader, model, epsilon=8):
    model.eval()

    total_err, total_loss = 0, 0
    for X,y in test_loader:
        X,y = X.cuda(), y.cuda()
        #adv samples
        delta = attack_pgd(model, X, y, epsilon=epsilon/255)
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    print ('PGD50: ', 1 - total_err / len(test_loader.dataset))

    return 1 - total_err / len(test_loader.dataset)

def print_args(args):
    print ('Attack Norm:', args.norm)
    print ('train eps: {} train step: {} train gamma: {} train randinit: {}'.format(args.train_eps, args.train_step, args.train_gamma, args.train_randinit))
    print ('test eps: {} test step: {} test gamma: {} test randinit: {}'.format(args.test_eps, args.test_step, args.test_gamma, args.test_randinit))
    print ('Model:', args.arch)
    print ('Dataset:', args.datasets)

def cifar10_dataloaders(batch_size=128, num_workers=2, data_dir='datasets/cifar10'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=128, num_workers=2, data_dir='datasets/cifar100'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def tiny_imagenet_dataloaders(batch_size=128, num_workers=2, data_dir = 'datasets/tiny-imagenet-200', permutation_seed=10):

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    np.random.seed(permutation_seed)
    split_permutation = list(np.random.permutation(100000))

    train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = ImageFolder(val_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def get_datasets(args):
    if args.datasets == 'CIFAR10':
        return cifar10_dataloaders(batch_size=args.batch_size, num_workers=args.workers)

    elif args.datasets == 'CIFAR100':
        return cifar100_dataloaders(batch_size=args.batch_size, num_workers=args.workers)

    elif args.datasets == 'TinyImagenet':
        return tiny_imagenet_dataloaders(batch_size=args.batch_size, num_workers=args.workers, permutation_seed=args.randomseed)

def get_model(args):
    if args.datasets == 'CIFAR10':
        num_class = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    elif args.datasets == 'CIFAR100':
        num_class = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    
    elif args.datasets == 'TinyImagenet':
        num_class = 200
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    
    if args.arch == 'PreActResNet18':
        net = resnet.__dict__[args.arch](num_classes=num_class)

    elif args.arch == 'WideResNet':
        net = wideresnet.__dict__[args.arch](28, num_classes=num_class, widen_factor=10, dropRate=0.0)

    elif args.arch == 'vgg16':
        from vgg import vgg16_bn
        net = vgg16_bn(num_class)

    net.normalize = dataset_normalization
    
    return net

# GradAlign loss
def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta

def get_input_grad(model, X, y, eps, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)
    loss = F.cross_entropy(output, y)

    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]

    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad

def grad_align_loss(model, X, y, args):

    grad1 = get_input_grad(model, X, y,  args.train_eps, delta_init='none', backprop=False)
    grad2 = get_input_grad(model, X, y,  args.train_eps, delta_init='random_uniform', backprop=True)
    grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
    cos = torch.nn.functional.cosine_similarity(grad1, grad2, 1)
    reg = args.glambda * (1.0 - cos.mean())

    return reg

################################ TRADES loss #######################################
def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return model(x_adv), loss

###########################################################################
# penalty on gradient norm
def penalty_on_grad_norm(model, inputs, target_var, criterion):
    params = list(model.parameters())
    output = model(inputs)
    loss = criterion(output, target_var)

    grad = autograd.grad(loss, inputs=params, create_graph=True)
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad])
    grad_penalty = torch.norm(grad_vec) ** 2

    total_loss = loss + grad_penalty * 0.1

    return output, total_loss


def AutoAttack(model, dataset='CIFAR10', norm='Linf', epsilon=8):
    model.eval()
    print ('evaluate AA:', dataset, norm, epsilon)
    epsilon /= 255.
    if dataset == 'CIFAR10':
        __, __, test_loader = cifar10_dataloaders(batch_size=1000)
    elif dataset == 'CIFAR100':
        __, __, test_loader = cifar100_dataloaders(batch_size=1000)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=norm, eps=epsilon, log_path='./log_file.txt',
        version='standard')

    # run attack and save images
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=1000)