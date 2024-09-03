# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib

import numpy as np
from tqdm import tqdm


import cv2

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
from torch.utils.data import DataLoader

from dataset import Dataset

import archs
from metrics import iou_score, Recall_suspect, Precision_certain, F1_score_special
from Dropoutblock import DropBlock_search
from sklearn.metrics import roc_curve, roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--name', default='neu_seg_BayesUNet_spatial_woDS',
                        help='model name')
    args = parser.parse_args()

    return args

def apply_dropout(m):
    if type(m) == DropBlock_search:
        m.train()

def roc_auc(y_pred_probs, y_true, threshold=0.5):
    y_true_binary = (y_true > threshold).astype(int)
    y_true_binary = y_true_binary.flatten()
    y_pred_probs = y_pred_probs.flatten()
    auc_value = roc_auc_score(y_true_binary, y_pred_probs)
    return auc_value

def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    model = archs.__dict__[args.arch](args)
    model = model.cuda()


    img_paths = glob('input/' + args.dataset + '/images/*')
    mask_paths = glob('input/' + args.dataset + '/masks/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)


    model.load_state_dict(torch.load('models/%s/model.pth' %args.name)['model'])
    model.eval()
    model.apply(apply_dropout)

    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    starttime = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                input = input.cuda()
                target = target.cuda()
                n = 16

                output_bayes = np.zeros(target.size())
                output_bayes_int = np.zeros(target.size())
                inputs = input
                for k in range(3):
                    inputs = torch.cat((inputs,input))
                outputs, features = model(inputs)
                for kk in range(3):
                    output_, features = model(inputs)
                    outputs = torch.cat((outputs, output_))
                results = outputs[0,:,:,:]# * ((scores[0,:,:]).unsqueeze(0))
                for ii in range(n-1):
                    results = torch.cat((results, outputs[ii+1, :, :, :]),0)# * (scores[ii+1, :, :].unsqueeze(0))),0)
                # results = torch.sigmoid(results)
                result = torch.sum(results, dim=0)
                result = result.unsqueeze(0).unsqueeze(0)

                for l in range(n):
                    output = outputs[l,:,:,:].unsqueeze(0)

                    output_mat = torch.sigmoid(output).data.cpu().numpy()
                    output_int = output_mat > 0.5
                    output_bayes += output_mat #* score_mat  # * (1 / n)
                    output_bayes_int += output_int

                target_mat = target.data.cpu().numpy()
                input_mat = input.data.cpu().numpy()

                output_bayes = output_bayes *(1/n)
                certain = output_bayes_int >= 16
                suspect = output_bayes_int >= 1
                uncertain = certain ^ suspect


                img_paths = val_img_paths[1 * i:1 * (i + 1)]


                for i in range(output_bayes.shape[0]):
                    imsave('output/%s/' % args.name + os.path.basename(img_paths[i]),
                               (output_bayes[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name + os.path.basename(img_paths[i].split('.')[0]+'_certain'+'.'+img_paths[i].split('.')[1]),
                           (certain[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name + os.path.basename(img_paths[i].split('.')[0] + '_suspect' + '.' + img_paths[i].split('.')[1]),
                           (suspect[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name  + os.path.basename(
                        img_paths[i].split('.')[0] + '_uncertain' + '.' + img_paths[i].split('.')[1]),
                           (uncertain[i, 0, :, :] * 255).astype('uint8'))
                    imsave('output/%s/' % args.name  + os.path.basename(
                        img_paths[i].split('.')[0] + '_target' + '.' + img_paths[i].split('.')[1]),
                           (target_mat[i, 0, :,:]*255).astype('uint8'))
                    imsave('output/%s/' % args.name + os.path.basename(
                        img_paths[i].split('.')[0] + '_input' + '.' + img_paths[i].split('.')[1]),
                           (input_mat[i, 0, :, :] * 255).astype('uint8'))



    torch.cuda.empty_cache()
    ious = []
    Accs = []
    Recs = []
    for i in tqdm(range(len(val_img_paths))):
        val_mask_paths = 'input/%s/masks/' % args.dataset + os.path.basename(val_img_paths[i]).split('.')[
                0] + '.png'

        mask = imread(val_mask_paths)
        mask = cv2.resize(mask, (256, 256))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        if args.dataset == 'NEU_Seg-main':
            pb = imread('output/%s/test' % args.name + os.path.basename(val_mask_paths).split('.')[0] + '.jpg')
            certain = imread(
                'output/%s/test' % args.name + os.path.basename(val_mask_paths).split('.')[0] + '_certain' + '.jpg')
            suspect = imread(
                'output/%s/test' % args.name + os.path.basename(val_mask_paths).split('.')[0] + '_suspect' + '.jpg')
        else:
            pb = imread('output/%s/'%args.name+os.path.basename(val_mask_paths).split('.')[0] + '.' + os.path.basename(val_mask_paths).split('.')[-1])
            certain = imread('output/%s/'%args.name+os.path.basename(val_mask_paths).split('.')[0] + '_certain' + '.' + os.path.basename(val_mask_paths).split('.')[-1])
            suspect = imread('output/%s/'%args.name+os.path.basename(val_mask_paths).split('.')[0] + '_suspect' + '.' + os.path.basename(val_mask_paths).split('.')[-1])



        mask = mask.astype('float32') / 255
        pb = pb.astype('float32') / 255
        certain = certain.astype('float32') / 255
        suspect = suspect.astype('float32') / 255

        iou = iou_score(pb, mask)
        ious.append(iou)

        Acc = Precision_certain(certain, mask)
        Accs.append(Acc)

        Rec = Recall_suspect(suspect, mask)
        Recs.append(Rec)

    print('IoU: %.4f, ACC: %.4f, Recs: %.4f' % (np.mean(ious), np.mean(Accs), np.mean(Recs)))

if __name__ == '__main__':
    main()

