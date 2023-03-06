model = dict(
    type='LiteFlowNet',
    data_preprocessor=dict(
        type='FlowDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=False,
        sigma_range=(0, 0.04),
        clamp_range=(0., 1.)),
    encoder=dict(
        type='NetC',
        in_channels=3,
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(32, 32, 64, 96, 128, 192),
        strides=(1, 2, 2, 2, 2, 2),
        num_convs=(1, 3, 2, 2, 1, 1),
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=None),
    decoder=dict(
        type='NetE',
        in_channels=dict(level6=192),
        corr_channels=dict(level6=49),
        sin_channels=dict(level6=386),
        rin_channels=dict(level6=195),
        feat_channels=64,
        mfeat_channels=(128, 128, 96, 64, 32),
        sfeat_channels=(128, 128, 96, 64, 32),
        rfeat_channels=(128, 128, 64, 64, 32, 32),
        patch_size=dict(level6=3),
        corr_cfg=dict(level6=dict(type='Correlation', max_displacement=3)),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        flow_div=20.,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled_corr=False,
        regularized_flow=False,
        extra_training_loss=False,
        flow_loss=dict(
            type='MultiLevelEPE',
            weights=dict(level6=0.32),
            p=2,
            reduction='sum'),
        init_cfg=None),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
    init_cfg=dict(
        type='Kaiming',
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0))
randomness = dict(seed=0, diff_rank_seed=True)