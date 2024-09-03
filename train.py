# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import os
import argparse
from glob import glob
from collections import OrderedDict
import random
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import archs
from metrics import iou_score, Recall_suspect_, Precision_certain_
import losses
from utils import str2bool, count_params

from Dropoutblock import DropBlock_search

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='BayesUNet_spatial',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False,
                        type=str2bool)
    parser.add_argument('--dataset', default='neu_seg',
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--size', default=256, type=int,
                        help='image size')
    parser.add_argument('--image-ext', default='bmp',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='bmp',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss_weight',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=500
                        , type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,

                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

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


def train(args, train_loader, model, criterion, criterion_val, optimizer, epoch, device):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()
    torch.autograd.set_detect_anomaly(True)

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.to(device)
        target = target.to(device)
        if args.deepsupervision:
            outputs = model(input)
            n = 4
            output_sum = []
            outputs_sum = []
            for i in range(len(outputs)):
                output_sum.append(outputs[i].unsqueeze(0))
            for j in range(n-1):
                outputs_new = model(input)
                ji = 0
                for output_new in outputs_new:
                    outputs_sum.append(torch.cat((output_sum[ji],output_new.unsqueeze(0)), 0))
                    ji += 1
                output_sum.clear()
                output_sum.extend(outputs_sum)
                outputs_sum.clear()
            output = output_sum[0]
            output_var = torch.var(output, dim=0)
            var = output_var[0]
            var_normals = ((var - torch.min(var)) / (torch.max(var) - torch.min(var))).unsqueeze(0)
            for ii in range(1, n):
                var = output_var[ii]
                var_normal = ((var - torch.min(var)) / (torch.max(var) - torch.min(var))).unsqueeze(0)
                var_normals = torch.cat(((var_normals, var_normal)), 0)
            loss = criterion(output[0], target, var_normals)
            for jj in range(len(output_sum)-1):
                aaaa = torch.mean(output_sum[jj+1], dim=0)
                loss += criterion_val(aaaa, target)
            loss /= len(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iou = iou_score(torch.mean(outputs[0], dim=0), target)

        else:
            n = 4
            output = model(input)
            outputs = output.unsqueeze(0)
            for i in range(n-1):
                output = model(input)
                outputs = torch.cat((outputs, output.unsqueeze(0)),0)
            output_var = torch.var(outputs, dim=0)
            output_mean = torch.mean(outputs, dim=0)
            var = output_var[0]
            var_normals = ((var - torch.min(var)) / (torch.max(var) - torch.min(var))).unsqueeze(0)
            for ii in range(1, n):
                var = output_var[ii]
                var_normal = ((var - torch.min(var)) / (torch.max(var) - torch.min(var))).unsqueeze(0)
                var_normals = torch.cat(((var_normals, var_normal)), 0)

            index = random.randint(0,3)
            loss = criterion(outputs[index], target, var_normals)
            iou = iou_score(output_mean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    new_losses = AverageMeter()
    new_ious = AverageMeter()
    Ps = AverageMeter()
    Rs = AverageMeter()

    def apply_dropout(m):
        if type(m) == DropBlock_search:
            m.train()

    # switch to evaluate mode
    model.eval()
    model.apply(apply_dropout)

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            inputs = input
            for k in range(3):
                inputs = torch.cat((inputs, input))
            outputs, features = model(inputs)
            for kk in range(3):
                output_, features = model(inputs)
                outputs = torch.cat((outputs, output_))
            results = outputs
            result = torch.mean(results, dim=0)
            result = result.unsqueeze(0)
            iou_new = iou_score(result, target)
            recall = Recall_suspect_(results, target)
            precision = Precision_certain_(results, target)
            loss_new = criterion(result, target)
            new_losses.update(loss_new.item(), input.size(0))
            new_ious.update(iou_new, input.size(0))
            Ps.update(precision, input.size(0))
            Rs.update(recall, input.size(0))

    log = OrderedDict([
        ('P', Ps.avg),
        ('R', Rs.avg),
        ('new_losses', new_losses.avg),
        ('new_ious', new_ious.avg),
    ])

    return log


def main():
    device = torch.device('cuda:0')
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[args.loss]().to(device)
        criterion_val = losses.BCEDiceLoss().to(device)

    cudnn.benchmark = True

    img_paths = glob('input/' + args.dataset + '/images/*')


    mask_paths = glob('input/' + args.dataset + '/masks/*')
    #
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)


    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)

    model = model.to(device)

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr':args.lr}
                                ])

    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=args.batch_size,
        batch_size= 1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_iou = 0
    # best_P = 0
    # best_R = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model,  criterion, criterion_val, optimizer,  epoch, device)
        # evaluate on validation set
        val_log = validate(args, val_loader, model,  criterion_val)


        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['new_losses'],
            val_log['new_ious'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1
        torch.save({'model': model.state_dict()}, 'models/%s/model_latest.pth' %args.name)

        if val_log['new_ious'] > best_iou:
            torch.save({'model': model.state_dict()}, 'models/%s/model.pth' %args.name)
            best_iou = val_log['new_ious']
            # best_P = val_log['P']
            # best_R = val_log['R']
            print("=> saved best model")
            trigger = 0
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - best_val_iou %.4f'
            %(train_log['loss'], train_log['iou'], val_log['new_losses'], val_log['new_ious'], best_iou))
        with open('models/%s/Iou_train.txt' % args.name, 'w') as f:
            f.write('%d %.4f\n' % (epoch, train_log['iou']))
        with open('models/%s/Iou_test.txt' % args.name, 'w') as f:
            f.write('%d %.4f\t%.4f\t%.4f\n' % (epoch, val_log['new_ious'], val_log['R'], val_log['P']))
        with open('models/%s/Loss_train.txt' % args.name, 'w') as f:
            f.write('%d %.4f\n' % (epoch, train_log['loss']))
        with open('models/%s/Loss_test.txt' % args.name, 'w') as f:
            f.write('%d %.4f\n' % (epoch, val_log['new_losses']))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
