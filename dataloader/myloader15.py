import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# 判断文件是不是用.jpg, .png等来结尾的
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    filepath_train = filepath + 'train/'
    filepath_val = filepath + 'val/'
    left_fold  = 'image_2/'  #左图原图文件夹
    right_fold = 'image_3/'  #右图原图文件夹
    disp_pre = 'disp_pre/'   #前一帧视差图文件夹
    disp_L = 'disp_occ_0/'   #做图视差图文件夹

    #获取图片名称
    image_train = [img for img in os.listdir(filepath_train+left_fold) if img.find('_10') > -1]
    image_val = [img for img in os.listdir(filepath_val+left_fold) if img.find('_10') > -1]
    image_val.sort(key=lambda x: int(x[:-4]))
    # image_pre = [img for img in os.listdir(filepath+disp_pre)]

    #划分训练集和验证集
    # train = image[:180]
    # disp_pre_t = image_pre[:180]
    # val   = image[180:]
    # disp_pre_v = image_pre[180:]

    #获取训练集对应图片的完整路径
    left_train   = [filepath_train+left_fold+img for img in image_train]
    right_train  = [filepath_train+right_fold+img for img in image_train]
    disp_pre_t   = [filepath_train+disp_pre+img for img in image_train]
    disp_train_L = [filepath_train+disp_L+img for img in image_train]

    #获取验证集对应图片的完整路径
    left_val   = [filepath_val+left_fold+img for img in image_val]
    right_val  = [filepath_val+right_fold+img for img in image_val]
    disp_pre_v = [filepath_val+disp_pre+img for img in image_val]
    disp_val_L = [filepath_val+disp_L+img for img in image_val]

    return left_train, right_train, disp_pre_t, disp_train_L, left_val, right_val, disp_pre_v, disp_val_L
