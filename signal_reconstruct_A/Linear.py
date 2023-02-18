# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 9:29
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Linear.py
# @Software: PyCharm

# 任务：通过前四个传感器复原第五个传感器的信号
# 有两组信息：第一是时间；第二是位置
# 所以我的几个想法是
# 1、信号本身做一组编码；时间做一组编码；位置做一组编码
# 2、根据Linear的写法，直接用信号本身做编码就足够了，到时候将位置用通道的方式进行隔开

# 先尝试一下第二种
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from typing import Optional, Union, Dict


class Linear(BaseModel):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Linear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Union[Optional[list], torch.Tensor] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list, tuple, None]:

        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([inputs.size(0), self.pred_len, inputs.size(2)], dtype=inputs.dtype)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](inputs[:, :, i])
            x = output
        else:
            x = self.Linear(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        if mode == 'loss':
            return {'loss': F.cross_entropy(x, data_samples)}
        elif mode == 'predict':
            return x, data_samples
        elif mode == 'tensor':
            return x  # [Batch, Output length, Channel]
        else:
            return None
