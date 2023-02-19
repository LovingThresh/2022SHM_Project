import json

from torch.optim import AdamW
from collections import namedtuple

from mmengine.runner import Runner
from mmengine.dataset import BaseDataset

from DLinear import MM_DLinear
from data_loader import load_signal

A_data_root = 'V:/2022SHM-dataset/'
A_path = ''
save_A_path = "signal_reconstruct_dataset_A.json"

dic = {"seq_len": 256, "pred_len": 256, "individual": True, 'enc_in': 4}
json_str = json.dumps(dic)
model_cfg = json.loads(json_str, object_hook=lambda d: namedtuple("X", d.keys())(*d.values()))

param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=True,
         begin=0,
         end=5),
    dict(type='CosineAnnealingLR',
         T_max=100,
         by_epoch=False,
         begin=5,
         end=100)
]

# the build function of runner class
runner = Runner(
    model=MM_DLinear(configs=model_cfg),
    work_dir='./work_dir',
    train_dataloader=dict(
        batch_size=256,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=BaseDataset(ann_file=save_A_path, data_root=A_data_root,
                            pipeline=[load_signal], data_prefix=dict(file_path=A_path), serialize_data=True),
        collate_fn=dict(type='default_collate')
    ),
    optim_wrapper=dict(type='AmpOptimWrapper', optimizer=dict(type=AdamW, lr=0.001)),
    param_scheduler=param_scheduler,
    train_cfg=dict(by_epoch=True, max_epochs=100, val_interval=1),
    val_dataloader=None,
    val_cfg=None,
    val_evaluator=None,
    default_hooks=dict(
        timer=dict(type='IterTimerHook'),
        checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=10),
        logger=dict(type='LoggerHook')),
)
runner.train()
