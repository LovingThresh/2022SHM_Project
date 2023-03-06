# -*- coding: utf-8 -*-
# @Time    : 2023/2/4 9:44
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : data_division_saver.py
# @Software: PyCharm
import os
import json
import random
import shutil
import numpy as np
import scipy.io as scio


clean_data_path = 'V:/2022SHM_Project/a/data_clean.mat'
noise_data_path = 'V:/2022SHM_Project/a/data_noised.mat'

mean = 3.0235e-07
std = 0.0144


def load_mat_data(data_path):

    data_dict = scio.loadmat(data_path)

    return data_dict['data']  # return (5, 1000001) for task A 100 Hz


loading_data = load_mat_data(data_path=clean_data_path)
# loading_data = load_mat_data(data_path=noise_data_path)

train_loading_data = loading_data[:, :int(0.95 * 1000001)]
val_loading_data = loading_data[:, int(0.95 * 1000001) : int(0.975 * 1000001)]
test_loading_data = loading_data[:, int(0.975 * 1000001):]


def sliding_window(data, sw_width=256, n_out=128, in_start=0):

    data_list = []

    for i in range(data.shape[1]):

        crop_data_begin = in_start + n_out * i
        crop_data_end = crop_data_begin + sw_width

        if crop_data_end < data.shape[1]:
            data_list.append(data[:, crop_data_begin:crop_data_end])

    return data_list


loading_data = test_loading_data
X = sliding_window(loading_data, sw_width=256, n_out=128, in_start=0)
X_1 = sliding_window(loading_data, sw_width=256, n_out=256, in_start=100)


# prex='start_0_n_128' ; 'start_100_n_256'
#

def save_crop_data(data_list, mode='train', prex='start_0_n_128'):

    for num, i in enumerate(data_list):

        np.savetxt('V:/2022SHM-dataset/crop_data_project_2_A_dataset/{}/{}_{}.csv'.format(mode, prex, num), i)


save_crop_data(X, mode='test')
save_crop_data(X_1, mode='test', prex='start_100_n_256')

# save_crop_data(X_2, prex='start_50_n_64')
# save_crop_data(X_2, prex='start_50_n_128')

# position_encoding = []


def data_division(path, train_path, val_path, test_path):

    signal_file_list = os.listdir(path)
    random.shuffle(signal_file_list)
    train_file_list =  signal_file_list[:int(len(signal_file_list) * 0.9)]
    val_file_list = signal_file_list[int(len(signal_file_list) * 0.9) : int(len(signal_file_list) * 0.95)]
    test_file_list = signal_file_list[int(len(signal_file_list) * 0.95):]

    for file_list, dst_path in zip([train_file_list, val_file_list, test_file_list], [train_path, val_path, test_path]):
        for file in file_list:
            shutil.copyfile(os.path.join(path, file), os.path.join(dst_path, file))


def get_info_meta(path, save_path):
    dict_info_meta = {
        "metainfo": {"dataset_type": "signal_reconstruct_dataset", "task_name": "A_task"},
        "data_list": []
    }

    signal_file_list = os.listdir(path)
    for i in signal_file_list:
        i_dict = {'file_path': i}
        dict_info_meta["data_list"].append(i_dict)

    json_str = json.dumps(dict_info_meta)

    with open(save_path, 'w') as f:
        f.write(json_str)

    return None


get_info_meta('V:/2022SHM-dataset/crop_data_project_2_A_dataset/val',
              'V:/2022SHM-dataset/crop_data_project_2_A_dataset/val.json')
