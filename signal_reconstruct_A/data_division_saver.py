# -*- coding: utf-8 -*-
# @Time    : 2023/2/4 9:44
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_division_saver.py
# @Software: PyCharm
import os
import json
import numpy as np
import scipy.io as scio


clean_data_path = './a/data_clean.mat'
noise_data_path = './a/data_noised.mat'

mean = 3.0235e-07
std = 0.0144


def load_mat_data(data_path):

    data_dict = scio.loadmat(data_path)

    return data_dict['data_noised']  # return (5, 1000001) for task A 100 Hz


loading_data = load_mat_data(data_path=clean_data_path)
# loading_data = load_mat_data(data_path=noise_data_path)


def sliding_window(data, sw_width=256, n_out=128, in_start=0):

    data_list = []

    for i in range(data.shape[1]):

        crop_data_begin = in_start + n_out * i
        crop_data_end = crop_data_begin + sw_width

        if crop_data_end < data.shape[1]:
            data_list.append(data[:, crop_data_begin:crop_data_end])

    return data_list


X = sliding_window(loading_data, sw_width=256, n_out=128, in_start=0)
X_1 = sliding_window(loading_data, sw_width=256, n_out=64, in_start=50)
X_2 = sliding_window(loading_data, sw_width=256, n_out=128, in_start=50)


# prex='start_0_n_128' ; 'start_50_n_64' ; 'start_50_n_128'
#

def save_crop_data(data_list, prex='start_0_n_128'):

    for num, i in enumerate(data_list):

        np.savetxt('./crop_dataset_Task_2_A/noise_{}_{}.csv'.format(prex, num), i)

# save_crop_data(X)
# save_crop_data(X_2, prex='start_50_n_64')
# save_crop_data(X_2, prex='start_50_n_128')

# position_encoding = []


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
