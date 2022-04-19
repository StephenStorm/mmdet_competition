# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'

albu_train_transforms = [
    dict(
        type='RandomRotate90',
        always_apply=False, p=0.5),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        # img_scale=(1333, 800),
        # img_scale=[(1200, 600), (1200, 1000)],
        # img_scale=[(1333, 800), (1333, 1200)],
        img_scale=[(2048, 800), (2048, 1400)],

        multiscale_mode='range',
        keep_ratio=True),
        # keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [dict(type='Shear', prob=0.5, level=0)],
            [dict(type='ColorTransform', prob=0.5, level=6)]
        ]
    ),
    dict(
            type='Albu',
            transforms=albu_train_transforms,
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_labels'],
                min_visibility=0.0,
                filter_lost_elements=True),
            keymap={
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            },
            update_pad_shape=False,
            skip_img_without_anno=True),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        # img_scale=[(1200, 600), (1200, 800), (1200, 1000)],
        # img_scale=[(1333, 800),(1333, 933), (1333, 1066),(1333, 1200)],
        img_scale=[(2048, 800), (2048, 1000), (2048, 1200), (2048, 1400)],
        # img_scale = [(2048, 1536), (2048, 1706), (2048, 1876), (2048, 2048)],

        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
datasetA = dict(
    type=dataset_type,
    # ann_file=data_root + 'train_val_split/train.json',            #80% data
    # img_prefix=data_root + 'train_val_split/train',
    ann_file=data_root + 'trademark/train/annotations/instances_train2017.json',   #all data
    img_prefix=data_root + 'trademark/train/images',
    pipeline=train_pipeline)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,

    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            datasets=[datasetA]
        )
    ),

    val=dict(
        # samples_per_gpu=4,
        # workers_per_gpu=4,
        type=dataset_type,
        ann_file=data_root + 'train_val_split/val.json',
        img_prefix=data_root + 'train_val_split/val',
        pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'train_val_split/val.json',
    #     img_prefix=data_root + 'train_val_split/val',
    #     pipeline=test_pipeline),
    test=dict(
        # samples_per_gpu=4,
        # workers_per_gpu=4,
        type=dataset_type,
        ann_file=data_root + 'trademark/test/annotations/instances_val2017.json',
        img_prefix=data_root + 'trademark/test/images',
        pipeline=test_pipeline)
    )
# evaluation = dict(interval=1, metric='bbox', start=5)
