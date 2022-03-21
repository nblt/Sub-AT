import argparse
import os
import time
import numpy as np
import random
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from utils import Logger, set_seed, get_datasets, get_model, print_args, epoch_adversarial, epoch_adversarial_PGD50
from utils import AutoAttack, Guided_Attack, grad_align_loss, trades_loss
import wandb

########################## parse arguments ##########################
parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--EXP', metavar='EXP', default='EXP', help='experiment name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='PreActResNet18',
                    help='model architecture (default: PreActResNet18)')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='training datasets')
parser.add_argument('--optimizer',  metavar='OPTIMIZER', default='sgd', type=str,
                    help='optimizer for training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50 iterations)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--log-dir', dest='log_dir',
                    help='The directory used to save the log',
                    default='save_temp', type=str)
parser.add_argument('--log-name', dest='log_name',
                    help='The log file name',
                    default='log', type=str)
parser.add_argument('--randomseed', 
                    help='Randomseed for training and initialization',
                    type=int, default=0)
parser.add_argument('--wandb', action='store_true', help='use wandb for online visualization')
parser.add_argument('--cyclic', action='store_true', help='use cyclic lr schedule (default: False)')
parser.add_argument('--lr_max', '--learning-rate-max', default=0.3, type=float,
                    metavar='cLR', help='maximum learning rate for cyclic learning rates')

########################## attack setting ##########################
adversary_names = ['Fast-AT', 'PGD', 'gradalign', 'GAT', 'trades']
parser.add_argument('--attack', metavar='attack', default='Fast-AT',
                    choices=adversary_names,
                    help='adversary for genernating adversarial examples: ' + ' | '.join(adversary_names) +
                    ' (default: Fast-AT)')

# Fast-AT / PGD
parser.add_argument('--norm', default='linf', type=str, help='linf or l2')
parser.add_argument('--train_eps', default=8., type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2., type=float, help='step size of attack during training')
parser.add_argument('--train_randinit', action='store_false', help='randinit usage flag (default: on)')
parser.add_argument('--test_eps', default=8., type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2., type=float, help='step size of attack during testing')
parser.add_argument('--test_randinit', action='store_false', help='randinit usage flag (default: on)')

# gradalign
parser.add_argument('--gradalign_lambda', default=0.2, type=float, help='lambda for gradalign')
# guideattack
parser.add_argument('--GAT_lambda', default=10.0, type=float, help='lambda for GAT')
# evaluate
parser.add_argument('--pgd50',  action='store_true', help='evaluate the model with pgd50 (default: False)')
parser.add_argument('--autoattack', '--aa', action='store_true', help='evaluate the model with AA (default: False)')



# Record training statistics
train_robust_acc = []
val_robust_acc = []
train_robust_loss = []
val_robust_loss = []
test_natural_acc = []
test_natural_loss = []
arr_time = []
model_idx = 0

def main():

    global args, best_robust, model_idx
    global param_avg, train_loss, train_err, test_loss, test_err, arr_time, adv_acc
    
    args = parser.parse_args()

    # Check the save_dir exists or not
    print ('save dir:', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the log_dir exists or not
    print ('log dir:', args.log_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))
    if args.wandb:
        print ('tracking with wandb!')
        wandb.init(project="Sub-AT", entity="nblt")
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        wandb.run.name = args.EXP + date

    print_args(args)
    print ('random seed:', args.randomseed)
    set_seed(args.randomseed)
    
    args.train_eps /= 255.
    args.train_gamma /= 255.
    args.test_eps /= 255.
    args.test_gamma /= 255.



    # Define model
    model = torch.nn.DataParallel(get_model(args))
    model.cuda()

    cudnn.benchmark = True
    best_robust = 0

    # Prepare Dataloader
    train_loader, val_loader, test_loader = get_datasets(args)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
    if args.cyclic:
        lr_scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr_max, steps_per_epoch=len(train_loader), epochs=30)
    else:
        if args.datasets == 'TinyImagenet':
            print ('TinyImagenet schedule: [50, 80]')
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=[50, 80], last_epoch=args.start_epoch - 1)
        else:
            print ('default schedule: [100, 150]')
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=[100, 150], last_epoch=args.start_epoch - 1)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            print ('from ', args.start_epoch)
            best_robust = checkpoint['best_robust']
            optimizer = checkpoint['optimizer']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(test_loader, model, criterion)    
        if args.pgd50:
            epoch_adversarial_PGD50(test_loader, model)
        if args.autoattack:
            AutoAttack(model, args, dataset=args.datasets)
        return

    is_best = 0
    print ('Start training: ', args.start_epoch, '->', args.epochs)
    
    print ('adversary:', args.attack)
    if args.attack == 'gradalign':
        arr_lambda = [0, 0.03, 0.04, 0.05, 0.06, 0.08, 0.11, 0.15, 0.20, 0.27, 0.36, 0.47, 0.63, 0.84, 1.12, 1.50, 2.00]
        args.gradalign_lambda = arr_lambda[int(args.train_eps)]
        print ('gradalign lambda:', args.gradalign_lambda)
    if args.attack == 'GAT':
        print ('GAT_lambda:', args.GAT_lambda)

    # DLDR sampling
    torch.save(model.state_dict(), os.path.join(args.save_dir,  str(model_idx) +  '.pt'))
    model_idx += 1

    nat_last5 = []
    rob_last5 = []

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch)

        # step learning rates
        if not args.cyclic:
            lr_scheduler.step()

        # evaluate on validation set
        natural_acc = validate(test_loader, model, criterion)

        # evaluate the adversarial robustness on validation set
        robust_acc, adv_loss = epoch_adversarial(val_loader, model, args)
        val_robust_acc.append(robust_acc)
        val_robust_loss.append(adv_loss)
        print ('adv acc on validation set', robust_acc)

        # remember best prec@1 and save checkpoint
        is_best = robust_acc > best_robust
        best_robust = max(robust_acc, best_robust)

        if args.wandb:
            wandb.log({"test natural acc": natural_acc})
            wandb.log({"test robust acc": robust_acc})

        if epoch + 5 >= args.epochs:
            nat_last5.append(natural_acc)        
            robust_acc, adv_loss = epoch_adversarial(test_loader, model, args)
            print ('adv acc on test set', robust_acc)
            rob_last5.append(robust_acc)

        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_robust': best_robust,
            'epochs': epoch,
            'optimizer': optimizer
        }, filename=os.path.join(args.save_dir, 'model.th'))

        # DLDR sampling
        torch.save(model.state_dict(), os.path.join(args.save_dir,  str(model_idx) +  '.pt'))
        model_idx += 1

    print ('train_robust_acc: ', train_robust_acc)
    print ('train_robust_loss: ', train_robust_loss)
    print ('val_robust_acc: ', val_robust_acc)
    print ('val_robust_loss: ', val_robust_loss)
    print ('test_natural_acc: ', test_natural_acc)
    print ('test_natural_loss: ', test_natural_loss)
    print ('total training time: ', np.sum(arr_time))
    print ('last 5 adv acc on test dataset:', np.mean(rob_last5))
    print ('last 5 nat acc on test dataset:', np.mean(nat_last5))
    
    print ('final:')
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final.pt'))
    if args.pgd50:
        epoch_adversarial_PGD50(test_loader, model)
    if args.autoattack:
        AutoAttack(model, args, dataset=args.datasets)


    print ('best:')
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best.pt')))
    robust_acc, adv_loss = epoch_adversarial(test_loader, model, args)
    print ('best adv acc on test dataset:', robust_acc)
    
    if args.pgd50:
        epoch_adversarial_PGD50(test_loader, model)
    if args.autoattack:
        AutoAttack(model, args, dataset=args.datasets)
    validate(test_loader, model, criterion)

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch):
    """
    Run one train epoch
    """
    global train_robust_acc, train_robust_loss, arr_time, args, model_idx
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )  

    # switch to train mode
    model.train()    

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        if args.attack == 'trades':
            # calculate robust loss
            output, loss = trades_loss(model=model,
                            x_natural=input_var,
                            y=target_var,
                            optimizer=optimizer,
                            step_size=args.train_gamma,
                            epsilon=args.train_eps,
                            perturb_steps=args.train_step,
                            beta=6.0)

        elif args.attack == 'GAT':
            out  = model(input_var)
            P_out = nn.Softmax(dim=1)(out)
        
            input_adv = input_var + ((4./255.0)*torch.sign(torch.tensor([0.5]).cuda() - torch.rand_like(input_var).cuda()).cuda())
            input_adv = torch.clamp(input_adv,0.0,1.0)       

            model.eval()
            input_adv = Guided_Attack(model,nn.CrossEntropyLoss(),input_adv,target,eps=8./255.0,steps=1,P_out=P_out,l2_reg=10,alt=(i%2))

            delta = input_adv - input_var
            delta = torch.clamp(delta,-8.0/255.0,8.0/255)
            input_adv = input_var+delta
            input_adv = torch.clamp(input_adv,0.0,1.0)

            model.train()
            adv_out = model(input_adv)
            out  = model(input_var)
            
            output = adv_out
            
            Q_out = nn.Softmax(dim=1)(adv_out)
            P_out = nn.Softmax(dim=1)(out)
        
            '''LOSS COMPUTATION'''
            
            closs = criterion(out, target_var)
            
            reg_loss =  ((P_out - Q_out)**2.0).sum(1).mean(0)
            
            loss = 1.0*closs + args.GAT_lambda*reg_loss
            
        else:
            # adv samples
            with ctx_noparamgrad(model):
                input_adv = adversary.perturb(input_var, target_var)

            # compute output
            output = model(input_adv)
            loss = criterion(output, target_var)

            if args.attack == 'gradalign':
                loss += grad_align_loss(model, input_var, target_var, args)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        output = output.float()
        loss = loss.float()

        # cyclic learning rates
        if args.cyclic:
            lr_scheduler.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 200 or i == 400 or i == 550:
            torch.save(model.state_dict(), os.path.join(args.save_dir,  str(model_idx) +  '.pt'))
            model_idx += 1

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    print ('Total time for epoch [{0}] : {1:.3f}'.format(epoch, batch_time.sum))

    train_robust_loss.append(losses.avg)
    train_robust_acc.append(top1.avg)
    if args.wandb:
        wandb.log({"train robust acc": top1.avg})
        wandb.log({"train robust loss": losses.avg})
    arr_time.append(batch_time.sum) 

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    global test_natural_acc, test_natural_loss  

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()


            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    test_natural_loss.append(losses.avg)
    test_natural_acc.append(top1.avg)
    return top1.avg

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
