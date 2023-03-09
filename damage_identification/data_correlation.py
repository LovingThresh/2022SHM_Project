import numpy as np
import scipy.io as scio


def calc_euclidean(actual, pred):
    return np.sqrt(np.sum((actual - pred) ** 2))


def calc_mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual))


def calc_correlation(actual, pred):
    a_diff = actual - np.mean(actual)
    p_diff = pred - np.mean(pred)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    return numerator / denominator


def load_mat_data(data_path):

    data_dict = scio.loadmat(data_path)

    return data_dict['A']  # return (5, 1000001) for task A 100 Hz


data = load_mat_data(r'V:\2022SHM-dataset\project3\Damage_identification\train_dataset\train_1.mat')

data_1 = data['A'][0, :]
data_5 = data['A'][-1, :]
