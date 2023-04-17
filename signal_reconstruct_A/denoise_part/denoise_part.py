# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 14:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : denoise_part.py
# @Software: PyCharm
import os
import numpy as np
import matplotlib.pyplot as plt

data_root = r'V:\2022SHM-dataset\crop_data_project_2_A_dataset\train'
clean_data_file = '0_start_0_n_128_0_0.csv'
clean_data = np.loadtxt(os.path.join(data_root, clean_data_file))
# clean_data中有五个通道的时序信号
# 建立五个子图，将这五个通道的信号分别画在各自的子图上
# 使用matplotlib画图
plt.figure()
plt.subplot(5, 1, 1)
plt.plot(clean_data[0, :])
plt.subplot(5, 1, 2)
plt.plot(clean_data[1, :])
plt.subplot(5, 1, 3)
plt.plot(clean_data[2, :])
plt.subplot(5, 1, 4)
plt.plot(clean_data[3, :])
plt.subplot(5, 1, 5)
plt.plot(clean_data[4, :])
plt.show()


# 生成噪声
def make_noise(N):
    a = -0.0025
    b = 0.0025
    noise = a + (b - a) * np.random.rand(N)

    return noise


# 给每个通道增加噪声，然后再将带噪声的信号画出来
# 为每个通道生成噪声
noise_data = clean_data + make_noise(clean_data.shape[1])
# 使用matplotlib画图
plt.figure()
plt.subplot(5, 1, 1)
plt.plot(noise_data[0, :])
plt.subplot(5, 1, 2)
plt.plot(noise_data[1, :])
plt.subplot(5, 1, 3)
plt.plot(noise_data[2, :])
plt.subplot(5, 1, 4)
plt.plot(noise_data[3, :])
plt.subplot(5, 1, 5)
plt.plot(noise_data[4, :])
plt.show()

# 使用python评价信号的相似性
# 使用numpy的corrcoef函数
# corrcoef函数的返回值是一个矩阵，矩阵的对角线上的元素是1，其他元素是两个向量的相关系数
# 由于clean_data和noise_data的shape都是(5, 256)，所以corrcoef函数的返回值是一个5*5的矩阵
np.corrcoef(clean_data, noise_data)
# 热力图的形式可视化corrcoef函数的返回值
import seaborn as sns
sns.heatmap(np.corrcoef(clean_data, noise_data))
plt.show()