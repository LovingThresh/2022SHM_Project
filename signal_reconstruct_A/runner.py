import torch
import torch.nn.functional as F

from torch.optim import AdamW

from mmengine import MODELS, METRICS
from mmengine.runner import Runner
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.dataset import BaseDataset

from config import *
from DLinear import DLinear
from TimesNet import TimesNet
from typing import Optional, Union, Dict


# ------------------------------------------------------------- #
#                        register_module                        #
# ------------------------------------------------------------- #


@MODELS.register_module()
class MM_DLinear(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.net = DLinear(configs=configs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Union[Optional[list], torch.Tensor] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list, tuple, None]:
        x = self.net(inputs)
        if mode == 'loss':
            return {'loss': F.mse_loss(x, data_samples)}
        elif mode == 'predict':
            return x, data_samples
        elif mode == 'tensor':
            return x
        else:
            return None


@MODELS.register_module()
class MM_TimesNet(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.net = TimesNet(configs=configs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Union[Optional[list], torch.Tensor] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list, tuple, None]:
        x = self.net(inputs)
        if mode == 'loss':
            return {'loss': F.mse_loss(x, data_samples)}
        elif mode == 'predict':
            return x, data_samples
        elif mode == 'tensor':
            return x
        else:
            return None


@METRICS.register_module()
class MM_MSELoss(BaseMetric):

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'mse_loss': F.mse_loss(score, gt),
        })

    def compute_metrics(self, results):
        total_loss = sum(item['mse_loss'] for item in results) / len(results)

        return dict(mse_loss=total_loss)


@METRICS.register_module()
class MM_MAELoss(BaseMetric):

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'mae_loss': F.l1_loss(score, gt),
        })

    def compute_metrics(self, results):
        total_loss = sum(item['mae_loss'] for item in results) / len(results)

        return dict(mae_loss=total_loss)


# -------------------------------------------------------- #
#                          Runner                          #
# -------------------------------------------------------- #
runner = Runner(
    model=dict(type='MM_DLinear', configs=DLiner_model_cfg),
    work_dir='S:/work_dir/DLinerNet',
    train_dataloader=dict(
        batch_size=batch_size,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=BaseDataset(ann_file=train_ann_file, data_root=data_root, data_prefix={'file_path': train_path},
                            pipeline=train_transform),
        collate_fn=dict(type='default_collate')),
    val_dataloader=dict(
        batch_size=batch_size,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=BaseDataset(ann_file=val_ann_file, data_root=data_root, data_prefix={'file_path': val_path},
                            pipeline=val_transform),
        collate_fn=dict(type='default_collate')),
    test_dataloader=dict(
        batch_size=batch_size,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=BaseDataset(ann_file=test_ann_file, data_root=data_root, data_prefix={'file_path': test_path},
                            pipeline=test_transform),
        collate_fn=dict(type='default_collate')),
    optim_wrapper=dict(type='OptimWrapper', optimizer=dict(type=AdamW, lr=init_lr)),
    param_scheduler=param_scheduler,
    train_cfg=dict(by_epoch=True, max_epochs=max_epoch, val_interval=1),
    val_cfg=dict(),
    val_evaluator=[dict(type='MM_MSELoss', prefix='val'), dict(type='MM_MAELoss', prefix='val')],
    test_cfg=dict(),
    test_evaluator=[dict(type='MM_MSELoss', prefix='test'), dict(type='MM_MAELoss', prefix='val')],
    default_hooks=dict(
        timer=dict(type='IterTimerHook'),
        checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=10, rule='less'),
        logger=dict(type='LoggerHook')),
    # load_from='./work_dir/20230219_171830/epoch_100.pth',
)
runner.train()
