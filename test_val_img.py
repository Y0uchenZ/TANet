from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import copy
from dataloader import myloader15 as ls
from dataloader import myloader as DA
from matplotlib import pyplot as plt
from err_calculation import *
from models import *

parser = argparse.ArgumentParser(description='TANet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='max disp')
parser.add_argument('--datapath', default='dataset/kitti2015_train/',
                    help='datapath')
parser.add_argument('--loadmodel', default='ckpt/finetune_486.tar',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_disp_pre_train, all_left_disp, test_left_img, test_right_img, test_disp_pre, test_disp = ls.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_disp_pre_train, all_left_disp, True),
    batch_size=2, shuffle=True, num_workers=2, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_disp_pre, test_disp, False),
    batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model = TANet(args.maxdisp)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL, imgR, disp_pre, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_pre = Variable(torch.FloatTensor(disp_pre))

    if args.cuda:
        imgL, imgR, disp_pre = imgL.cuda(), imgR.cuda(), disp_pre.cuda()

    start_time = time.time()
    with torch.no_grad():
        output = model(imgL, imgR, disp_pre)
    cost_time = time.time()-start_time

    pred_disp = output.data.cpu()  # torch.Size([1, 1, 368, 1232])
    pred_disp = pred_disp.squeeze(1)

    mask = (disp_true > 0)
    mask.detach_()
    epe = EPE_metric(pred_disp, disp_true, mask)
    D1 = D1_metric(pred_disp, disp_true, mask)
    Thres1 = Thres_metric(pred_disp, disp_true, mask, 1.0)
    Thres2 = Thres_metric(pred_disp, disp_true, mask, 2.0)
    Thres3 = Thres_metric(pred_disp, disp_true, mask, 3.0)
    err_pac = [epe, D1, Thres1, Thres2, Thres3]

    return cost_time, err_pac

def main():
    total_times = 0
    total_epe = 0
    total_D1_t = 0
    total_T1 = 0
    total_T2 = 0
    total_T3 = 0
    for batch_idx, (imgL, imgR, disp_pre, disp_L) in enumerate(TestImgLoader):
        cost_time, err = test(imgL, imgR, disp_pre, disp_L)
        print('Iter %d, cost time = %.3f, epe = %.3f, D1_t = %.3f, T1 = %.3f, T2 = %.3f, T3 = %.3f'
              % (batch_idx, cost_time, err[0], err[1]*100, err[2]*100, err[3]*100, err[4]*100))
        total_times += cost_time
        total_epe += err[0]
        total_D1_t += err[1]
        total_T1 += err[2]
        total_T2 += err[3]
        total_T3 += err[4]
    val_time = total_times / len(TestImgLoader)
    val_epe = total_epe / len(TestImgLoader)
    val_D1_t = total_D1_t / len(TestImgLoader) * 100
    val_T1 = total_T1 / len(TestImgLoader) * 100
    val_T2 = total_T2 / len(TestImgLoader) * 100
    val_T3 = total_T3 / len(TestImgLoader) * 100

    print('average time = %.3f, average epe = %.3f, average D1_t = %.3f, average T1 = %.3f, average T2 = %.3f, average T3 = %.3f'
          % (val_time, val_epe, val_D1_t, val_T1, val_T2, val_T3))

if __name__ == '__main__':
    main()