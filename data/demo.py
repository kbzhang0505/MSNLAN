import os

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=True):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark
        self.filelist_hr = []      #修改
        self.filelist_lr = []

        hr_path = os.path.join(args.dir_demo, 'HR')
        lr_path = os.path.join(args.dir_demo, 'LR')
        lr_path = os.path.join(lr_path, 'X{}'.format(self.scale[0]))
        for f in os.listdir(lr_path):
            if f.find('.png') >= 0 or f.find('.jp') >= 0:
                self.filelist_lr.append(os.path.join(lr_path, f))
        self.filelist_lr.sort()
        for q in os.listdir(hr_path):                                    #修改 添加HR图像路径
            if q.find('.png') >= 0 or q.find('.jp') >= 0:
                self.filelist_hr.append(os.path.join(hr_path, q))
        self.filelist_hr.sort()
    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist_hr[idx]))[0]
        lr = imageio.imread(self.filelist_lr[idx])
        hr = imageio.imread(self.filelist_hr[idx])
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
        hr, = common.set_channel(hr, n_channels=self.args.n_colors)
        hr_t, = common.np2Tensor(hr, rgb_range=self.args.rgb_range)

        return lr_t, hr_t, filename

    def __len__(self):
        return len(self.filelist_lr)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

