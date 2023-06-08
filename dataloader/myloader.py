import torch.utils.data as data
import random
from PIL import Image
import numpy as np
import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

#文件类型判断
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#读取左右视图的图像，转为RGB（原本为RGBA，A是透明通道）
def default_loader(path):
    return Image.open(path).convert('RGB')

#打开图像，视差图直接打开就行
def disparity_loader(path):
    return Image.open(path)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, disp_pre, left_disparity, training, loader=default_loader, dploader=disparity_loader):
        self.left     = left
        self.right    = right
        self.disp_pre = disp_pre
        self.disp_L   = left_disparity
        self.loader   = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left     = self.left[index]
        right    = self.right[index]
        disp_pre = self.disp_pre[index]
        disp_L   = self.disp_L[index]

        left_img     = self.loader(left)
        right_img    = self.loader(right)
        disp_pre_img = self.dploader(disp_pre)
        dataL        = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            # 指定范围内的随机整数
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            # 使用PIL裁切图片，输入是左上角坐标，右下角坐标
            left_img = left_img.crop((x1, y1, x1+tw, y1+th))
            right_img = right_img.crop((x1, y1, x1+tw, y1+th))

            # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            dataL = dataL[y1:y1+th, x1:x1+tw]

            disp_pre_img = np.ascontiguousarray(disp_pre_img, dtype=np.float32) / 256
            disp_pre_img = disp_pre_img[y1:y1+th, x1:x1+tw]

            processed = preprocess.get_transform(augment=False)
            left_img  = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, disp_pre_img, dataL
        else:
            w, h = left_img.size

            left_img  = left_img.crop((w-1232, h-368, w, h))
            right_img = right_img.crop((w-1232, h-368, w, h))

            dataL = dataL.crop((w-1232, h-368, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            disp_pre_img = disp_pre_img.crop((w-1232, h-368, w, h))
            disp_pre_img = np.ascontiguousarray(disp_pre_img, dtype=np.float32) / 256

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, disp_pre_img, dataL

    def __len__(self):
        return len(self.left)