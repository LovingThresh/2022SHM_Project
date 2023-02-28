_base_ = [
    '../_base_/models/liteflownet/liteflownet_pre_M5S5R5.py',
    '../_base_/datasets/flyingchairs_320x448.py',
    '../_base_/schedules/liteflownet_pre_240k.py',
    '../_base_/default_runtime.py'
]

custom_hooks = [
    dict(
        type='LiteFlowNetStageLoadHook',
        src_level='level6',
        dst_level='level5')
]

# Weights are initialized from model of previous stage
load_from = 'https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M6S6R6_8x1_flyingchairs_320x448.pth'  # noqa
