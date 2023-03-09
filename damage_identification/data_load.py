import scipy.io as scio

data_path = r'V:\2022SHM-dataset\project3\Damage_identification\train_dataset\train_1.mat'

data_dict = scio.loadmat(data_path)

# 实验思路
# 明确信号识别是一个六分类问题（0 - 5） + 一个回归问题
# 给信号添加位置
