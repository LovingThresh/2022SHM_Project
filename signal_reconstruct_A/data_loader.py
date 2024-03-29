# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 10:55
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm

import random

import numpy as np
from tsai.data.core import TSTensor
from tsai.data.preprocessing import TSStandardize
from tsai.data.transforms import TSIdentity, TSGaussianNoise, TSMaskOut, RandAugment, TSMagAddNoise, TSRandomTimeScale, \
TSRandomLowRes, TSInputDropout, TSRandomTrend, TSVarOut, TSCutOut

all_TS_randaugs = [
    TSIdentity,
    (TSVarOut, 0.05, 0.5),
    (TSCutOut, 0.05, 0.5),
]


class Signal_transform:

    def __init__(self, mode='train'):

        self.mode = mode

    def __call__(self, data_info):
        signal = np.loadtxt(data_info['file_path'])
        signal = TSTensor(signal)

        # TSStandardize

        signal = TSStandardize(mean=3.0235e-07, std=0.0144)(signal)

        # 将信号进行分离，分为前四通道与第五个通道

        top_four_signal = signal[:4, :]
        fifth_signal = signal[-1:, :]

        if self.mode == 'train':

            # TSRandAugment
            probability = random.random()
            if probability < 0.2:
                pass
            elif probability < 0.6:
                top_four_signal = TSGaussianNoise(.1, additive=True)(TSTensor(top_four_signal))
            elif probability < 0.8:
                top_four_signal = TSMaskOut(.1, compensate=True)(TSTensor(top_four_signal))
            else:
                top_four_signal = RandAugment(all_TS_randaugs, N=5, M=10)(TSTensor(top_four_signal))

        elif self.mode == 'val' or 'test':
            pass

        return top_four_signal.permute(1, 0), fifth_signal.permute(1, 0)


class Signal_transform_B:

    def __init__(self, mode='train'):

        self.mode = mode

    def __call__(self, data_info):
        signal = np.loadtxt(data_info['file_path'])
        signal = TSTensor(signal)

        # TSStandardize

        signal = TSStandardize(mean=3.0235e-07, std=0.0144)(signal)

        # 将信号进行分离，分为前二通道与后三通道

        top_two_signal = signal[:2, :]
        rear_three_signal = signal[2:, :]

        if self.mode == 'train':

            # TSRandAugment
            probability = random.random()
            if probability < 0.2:
                pass
            elif probability < 0.6:
                top_two_signal = TSGaussianNoise(.1, additive=True)(TSTensor(top_two_signal))
            elif probability < 0.8:
                top_two_signal = TSMaskOut(.1, compensate=True)(TSTensor(top_two_signal))
            else:
                top_two_signal = RandAugment(all_TS_randaugs, N=5, M=10)(top_two_signal, split_idx=0)

        elif self.mode == 'val' or 'test':
            pass

        return top_two_signal.permute(1, 0), rear_three_signal.permute(1, 0)
