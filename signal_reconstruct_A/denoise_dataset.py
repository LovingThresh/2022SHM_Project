import os
import json
import numpy as np
from tsai.data.core import TSTensor
from torch.utils.data import Dataset, DataLoader
from tsai.data.preprocessing import TSStandardize


def make_noise(N):
    a = -0.0025
    b = 0.0025
    noise = a + (b - a) * np.random.rand(N)

    return noise


#
# 在原先的基础上增强一个降噪模型
class Noise_Dataset(Dataset):
    def __init__(self, data_root, data_info):
        self.data_root = data_root
        with open(data_info) as f:
            self.data = json.load(f)
        # data: dict{'metainfo', 'data_list'}
        self.data_list = self.data['data_list']
        # data_list: [dict{'file_path'}]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        file_name = self.data_list[item]['file_path']
        if 'noise' in file_name:
            clean_data_file = file_name[6:]
            noise_data_file = file_name
        else:
            clean_data_file = file_name
            noise_data_file = None

        clean_data = np.loadtxt(os.path.join(self.data_root, clean_data_file))

        if noise_data_file is not None:
            noise_data = np.loadtxt(os.path.join(self.data_root, noise_data_file))
        else:
            noise = make_noise(clean_data.shape)
            noise_data = clean_data + noise

        return TSStandardize(mean=3.0235e-07, std=0.0144)(TSTensor(noise_data)).permute(1, 0), \
            TSStandardize(mean=3.0235e-07, std=0.0144)(TSTensor(clean_data)).permute(1, 0)


train_dataset = Noise_Dataset(data_root='V:/2022SHM-dataset/crop_data_project_2_A_dataset/train',
                              data_info='V:/2022SHM-dataset/crop_data_project_2_A_dataset/train.json')
val_dataset = Noise_Dataset(data_root='V:/2022SHM-dataset/crop_data_project_2_A_dataset/val',
                            data_info='V:/2022SHM-dataset/crop_data_project_2_A_dataset/val.json')
test_dataset = Noise_Dataset(data_root='V:/2022SHM-dataset/crop_data_project_2_A_dataset/test',
                             data_info='V:/2022SHM-dataset/crop_data_project_2_A_dataset/test.json')

train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
