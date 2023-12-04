# **A Fast Stereo Matching Network Based on Temporal Attention and 2D Convolution**  

This repository contains the implementation of the paper: [A Fast Stereo Matching Network Based on Temporal Attention and 2D Convolution](https://doi.org/10.1016/j.patcog.2023.109808), Youchen Zhao, Hua Zhong*, Boyuan Jia, Haixiong Li (Pattern Recognition).

### Citation
```
@article{zhao2023fast,
  title={A fast stereo matching network based on temporal attention and 2D convolution},
  author={Zhao, Youchen and Zhong, Hua and Jia, Boyuan and Li, Haixiong},
  journal={Pattern Recognition},
  volume={144},
  pages={109808},
  year={2023},
  publisher={Elsevier}
}
```

## Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Usage](#usage)

## Introduction

We propose a fast stereo matching network based on temporal attention and 2D convolution (TANet). Due to the high similarity of the disparity between consecutive frames in an image sequence, we propose a temporal attention (TA) module that uses the disparity map of the previous frame to guide the disparity search range in the current frame, thus significantly improving the efficiency of disparity calculation in the cost volume module. Additionally, we propose a hierarchical cost construction and 2D convolution aggregation module that constructs a pyramid cost volume by fusing edge cues to establish detail constraints. This overcomes the problem of difficult convergence caused by information loss when replacing 3D convolution with 2D convolution. Experimental results show that the TA module effectively optimizes the cost volume and, together with 2D convolution, improves the computational speed. Compared with state-of-the-art algorithms, TANet achieves a speedup of nearly 4x, with a running time of 0.061s, and reduces the parameter count by nearly half while decreasing accuracy by 1.1%. 


## Architecture

<img align="center" src="https://github.com/Y0uchenZ/TANet/blob/main/Architecture.png?raw=true">

## Usage

### Dependencies

- [Python 3.7](https://www.python.org/downloads/)
- [PyTorch(1.6.0+)](http://pytorch.org)
- torchvision 0.5.0
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)

### Train

As an example, use the following command to train a TANet on KITTI Stereo

```python
python train.py	--maxdisp 192 \
			    --datapath (your KITTI Stereo data folder) \
                --dataset kitti2015 (kitti2012) \
			    --epochs 500 \
 			    --loadmodel (optional) \
                --savemodel (path for saving model) \
                --no-cuda (enable CUDA training) \
                --seed 1 \
                --logdir (path for saving log)
```

### Test

Use the following command to test the trained TANet on a group of images in KITTI 2015

```python
python test_1_img.py --maxdisp 192 \
                     --datapath (your KITTI Stereo data folder) \
                     --loadmodel (optional) \
                     --error_vis (save error visualization) \
                     --pred_disp (save pred_disp)
                     --no-cuda (enables CUDA training) \
                     --seed 1 
```

### Evaluation

Use the following command to evaluate the trained TANet on KITTI 2015 test data

```python
python test_val_img.py --maxdisp 192 \
                       --datapath (your KITTI Stereo data folder) \
                       --loadmodel (optional) \
                       --no-cuda (enables CUDA training) \
                       --seed 1
```
