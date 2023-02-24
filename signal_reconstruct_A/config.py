from utils import dict2cls
from data_loader import Signal_transform


# dataset setting
data_root = 'V:/2022SHM-dataset/crop_dataset_Task_2_A_dataset'
train_path = 'train'
train_ann_file = "train.json"
val_path = 'val'
val_ann_file = "val.json"
test_path = 'val'
test_ann_file = "val.json"

# transform setting
train_transform = [Signal_transform(mode='train')]
val_transform = [Signal_transform(mode='val')]
test_transform = [Signal_transform(mode='test')]

# batch_size setting
batch_size = 32

# model_cfg setting
dic = {"seq_len": 256, "pred_len": 256, "individual": True, 'enc_in': 4}
DLiner_model_cfg = dict2cls(dic)

dic = {"seq_len": 256, "pred_len": 256, "freq": 'h', 'enc_in': 4, 'dec_in': 4, 'd_model': 256, 'embed': 'fixed',
       'dropout': 0.1, 'e_layers': 2, 'c_out': 1, 'd_ff': 512, 'num_kernels': 6, 'top_k': 5}
TimesNet_model_cfg = dict2cls(dic)

max_epoch = 100
init_lr = 0.001
# init_lr = 0.00005

param_scheduler = [
    dict(type='LinearLR',
         start_factor=init_lr,
         by_epoch=True,
         begin=0,
         end=5),
    dict(type='CosineAnnealingLR',
         T_max=100,
         by_epoch=True,
         begin=5,
         end=100)]
