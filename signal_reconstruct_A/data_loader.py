# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 10:55
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_loader.py
# @Software: PyCharm
import os
import json
import random

import numpy as np
import torch.nn.functional
from mmengine.dataset import BaseDataset

from tsai.data.core import TSTensor
from tsai.data.preprocessing import TSStandardize
from tsai.data.transforms import TSIdentity, TSMagAddNoise, TSGaussianNoise, TSInputDropout, TSMagMulNoise, \
    TSRandomFreqNoise, TSShuffleSteps, TSTimeNoise, TSBlur, TSSmooth, TSCutOut, TSTimeStepOut, TSRandomResizedCrop, \
    TSMaskOut, RandAugment

from functools import partial

all_TS_randaugs = [

    TSIdentity,

    # Noise
    (TSMagAddNoise, 0.1, 1.),
    (TSGaussianNoise, .01, 1.),
    (partial(TSMagMulNoise, ex=0), 0.1, 1),
    (partial(TSTimeNoise, ex=0), 0.1, 1.),
    (partial(TSRandomFreqNoise, ex=0), 0.1, 1.),
    partial(TSShuffleSteps, ex=0),
    (TSInputDropout, 0.05, .5),

    # Magnitude

    partial(TSBlur, ex=0),
    partial(TSSmooth, ex=0),
    (TSCutOut, 0.05, 0.5),

    # Time
    (TSTimeStepOut, 0.01, 0.2),
    (TSRandomResizedCrop, 0.05, 0.5),
    (TSMaskOut, 0.01, 0.2),
]

a = {
    "metainfo": {"dataset_type": "test_dataset", "task_name": "test_task"},
    "data_list": []}


def get_info_meta(path, save_path):
    dict_info_meta = {
        "metainfo": {"dataset_type": "signal_reconstruct_dataset", "task_name": "A_task"},
        "data_list": []
    }

    signal_file_list = os.listdir(path)
    for i in signal_file_list:
        i_dict = {'file_path': path + '/' + i}
        dict_info_meta["data_list"].append(i_dict)

    json_str = json.dumps(dict_info_meta)

    with open(save_path, 'w') as f:
        f.write(json_str)

    return None


def load_signal(data_info):

    signal = np.loadtxt(data_info['file_path'])
    TS_signal = TSTensor(signal)

    # TSStandardize

    TS_signal = TSStandardize(mean=3.0235e-07, std=0.0144)(TS_signal)

    # TSRandAugment
    probability = random.random()

    if probability < 0.5:
        pass
    elif probability < 0.8:
        TS_signal = TSGaussianNoise(.1, additive=True)(TSTensor(TS_signal))
    elif probability < 0.9:
        TS_signal = TSMaskOut(.1, additive=True)(TSTensor(TS_signal))
    else:
        TS_signal = RandAugment(all_TS_randaugs, N=5, M=10)(
            TS_signal, split_idx=0)

    return TS_signal


A_data_root = 'V:/2022SHM-dataset/'
A_path = ''
save_A_path = "signal_reconstruct_dataset_A.json"
# get_info_meta(A_path, save_A_path)

# the build function of runner class

dataset = BaseDataset(ann_file=save_A_path, data_root=A_data_root, pipeline=[load_signal],
                      data_prefix=dict(file_path=A_path), serialize_data=True)
a = next(iter(dataset))
len(dataset)

# class Signal_reconstruct_A(BaseDataset):
#     def get_cat_ids(self, idx: int) -> List[int]:
#         pass
#
#     def __init__(self):
torch.nn.functional.cross_entropy(a[0], a[1])
