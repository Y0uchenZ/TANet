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
import cv2
from err_calculation import *
from models import *
from visualization import *

parser = argparse.ArgumentParser(description='TANet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='max disp')
parser.add_argument('--datapath', default='',
                    help='datapath')
parser.add_argument('--loadmodel', default='',
                    help='load model')
parser.add_argument('--error_vis', default='',
                    help='save error visualization')
parser.add_argument('--pred_disp', default='',
                    help='save pred_disp')
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

def main():
    for batch_idx, (img_L, img_R, disp_pre, disp_L) in enumerate(TestImgLoader):
        if batch_idx == 10:
            model.eval()
            imgL = Variable(torch.FloatTensor(img_L))
            imgR = Variable(torch.FloatTensor(img_R))
            disp_pre = Variable(torch.FloatTensor(disp_pre))

            if args.cuda:
                imgL, imgR, disp_pre = imgL.cuda(), imgR.cuda(), disp_pre.cuda()

            start_time = time.time()
            with torch.no_grad():
                output = model(imgL, imgR, disp_pre)
            cost_time = time.time() - start_time

            pred_disp = output.data.cpu()  # torch.Size([1, 1, 368, 1232])
            pred_disp = pred_disp.squeeze(1)

            mask = (disp_L > 0)
            mask.detach_()
            epe = EPE_metric(pred_disp, disp_L, mask)
            D1 = D1_metric(pred_disp, disp_L, mask)
            Thres1 = Thres_metric(pred_disp, disp_L, mask, 1.0)
            Thres2 = Thres_metric(pred_disp, disp_L, mask, 2.0)
            Thres3 = Thres_metric(pred_disp, disp_L, mask, 3.0)
            print('time = %.3f, epe = %.3f, D1 = %.3f, T1 = %.3f, T2 = %.3f, T3 = %.3f' % (cost_time, epe, D1*100, Thres1*100, Thres2*100, Thres3*100))

            img_left = img_L.squeeze().numpy().transpose([1, 2, 0])
            error_vis = disp_error_image_func.apply(pred_disp, disp_L).squeeze()
            error_vis = error_vis.numpy().transpose([1, 2, 0])
            error_vis = cv2.cvtColor(error_vis, cv2.COLOR_RGB2BGR)
            # cv2.imshow('error', error_vis)
            # cv2.imshow('left', img_left)
            # cv2.waitKey()
            cv2.imwrite(args.error_vis, error_vis*255)
            img = pred_disp.numpy().transpose([1, 2, 0]).astype(np.uint8)
            # cv2.imshow('disp', img)
            # cv2.waitKey()
            cv2.imwrite(args.pred_disp, img)


if __name__ == '__main__':
    main()
