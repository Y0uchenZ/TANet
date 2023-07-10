import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    filepath_train = filepath + 'training/'
    # filepath_val = filepath + 'val/'
    left_fold  = 'colored_0/'
    right_fold = 'colored_1/'
    disp_pre   = 'disp_pre/'
    disp_noc   = 'disp_occ/'

    image = [img for img in os.listdir(filepath_train+left_fold) if img.find('_10') > -1]

    train = image[:180]
    val   = image[180:]

    left_train  = [filepath_train+left_fold+img for img in train]
    right_train = [filepath_train+right_fold+img for img in train]
    disp_train  = [filepath_train+disp_noc+img for img in train]
    disp_pre_t  = [filepath_train+disp_pre+img for img in train]


    left_val   = [filepath_train+left_fold+img for img in val]
    right_val  = [filepath_train+right_fold+img for img in val]
    disp_val   = [filepath_train+disp_noc+img for img in val]
    disp_pre_v = [filepath_train+disp_pre+img for img in val]

    return left_train, right_train, disp_pre_t, disp_train, left_val, right_val, disp_pre_v, disp_val
