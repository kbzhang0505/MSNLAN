import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        # self.apath = os.path.join(dir_data, 'benchmark', self.name)
        # self.apath = os.path.join(dir_data, 'Classical_SR_Test', self.name)
        self.apath = os.path.join(dir_data, self.name)

        # self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_hr = os.path.join(self.apath, 'GTmod12')
        if self.input_large:
            # self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
            self.dir_lr = os.path.join(self.apath, 'LRbicx4')
        else:
            # self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
            self.dir_lr = os.path.join(self.apath, 'LRbicx4')
        self.ext = ('.png', '.png')

