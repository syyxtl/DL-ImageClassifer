# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import os
import time
import scipy.io
from model import PCB, PCB_test
from utils import load_data_qg

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids', default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='./Market/pytorch',type=str, help='./test_data')
parser.add_argument('--model_dir', default='./Models/', type=str, help='save model dir')
parser.add_argument('--result_dir', default='./Results/', type=str, help='save result dir')
parser.add_argument('--stage', default='PCB', type=str, help='save model path')
parser.add_argument('--RPP',  default=True, action='store_true', help='use RPP')
parser.add_argument('--batchsize', default=8, type=int, help='batch_size')
parser.add_argument('--feature_H', default=True, action='store_true', help='extract feature_H')
args = parser.parse_args()

test_dir = args.test_dir
model_dir = args.model_dir
stage = args.stage
result_dir = args.result_dir
feature_H = args.feature_H

######################################################################
#************************* GPU concern  *****************************#
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
use_gpu = torch.cuda.is_available()
#************************* GPU concern  *****************************#
######################################################################

datas, images = load_data_qg()

######################################################################
#************************* load model *******************************#
def load_network(network):
    save_path = os.path.join(args.model_dir, stage, 'net_%s.pth'%args.which_epoch)
    network.load_state_dict(torch.load(save_path), False)
    return network
#************************* loaded model *****************************#
######################################################################

######################################################################
#************ extract feature from a trained model ******************#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size() #batch_size, chaanel, height, weight
        count += n
        print(count)
        if args. feature_H:
            ff = torch.FloatTensor(n, 256, 6).zero_() # we have six parts
        else:
            ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts

        for i in range(2):
            if(i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f

        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
        features = torch.cat((features, ff), 0)
    return features
#************ extract feature from a trained model ******************#
######################################################################

######################################################################
#******************** get camera id & label *************************#
def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels
#******************* geted camera id & label ************************#
######################################################################

######################################################################
#************************* main test ********************************#
def test_main():
    print('-------test-----------')
    gallery_path = images['gallery'].imgs
    query_path = images['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    model_structure = PCB(751)
    if args.RPP:
        model_structure = model_structure.convert_to_rpp()

    model = load_network(model_structure)
    model = PCB_test(model, feature_H)

    model = model.eval()
    if use_gpu:
        model = model.cuda()

    gallery_feature = extract_feature(model, datas['gallery'])
    query_feature = extract_feature(model, datas['query'])

    result = {
        'gallery_feature':gallery_feature.numpy(),
        'gallery_label':gallery_label,
        'gallery_cam':gallery_cam,
        'query_feature':query_feature.numpy(),
        'query_label':query_label,
        'query_cam':query_cam
    }

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    if args.RPP:
        save_pre = '/RPP_'
    else:
        save_pre = '/PCB_'

    if args.feature_H:
        save_pre += 'H_'
    else:
        save_pre += 'G_'
    scipy.io.savemat(args.result_dir + save_pre + 'result.mat', result)
#************************** main test  ******************************#
######################################################################

test_main()