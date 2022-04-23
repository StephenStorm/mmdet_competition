dataset_type = 'CocoDataset'
data_root = 'data/'
albu_train_transforms = [
    dict(type='RandomRotate90', always_apply=False, p=0.5),
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
        p=0.1)
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(2048, 800), (2048, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Shear',
            'prob': 0.5,
            'level': 0
        }], [{
            'type': 'ColorTransform',
            'prob': 0.5,
            'level': 6
        }]]),
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomRotate90', always_apply=False, p=0.5),
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
            dict(
                type='JpegCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(2048, 800), (2048, 1000), (2048, 1200), (2048, 1400)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
datasetA = dict(
    type='CocoDataset',
    ann_file='data/train_val_split/train.json',
    img_prefix='data/train_val_split/train',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(2048, 800), (2048, 1400)],
            multiscale_mode='range',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='AutoAugment',
            policies=[[{
                'type': 'Shear',
                'prob': 0.5,
                'level': 0
            }], [{
                'type': 'ColorTransform',
                'prob': 0.5,
                'level': 6
            }]]),
        dict(
            type='Albu',
            transforms=[
                dict(type='RandomRotate90', always_apply=False, p=0.5),
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
                dict(
                    type='JpegCompression',
                    quality_lower=85,
                    quality_upper=95,
                    p=0.2),
                dict(type='ChannelShuffle', p=0.1),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(type='Blur', blur_limit=3, p=1.0),
                        dict(type='MedianBlur', blur_limit=3, p=1.0)
                    ],
                    p=0.1)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_labels'],
                min_visibility=0.0,
                filter_lost_elements=True),
            keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
            update_pad_shape=False,
            skip_img_without_anno=True),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='CocoDataset',
                    ann_file='data/train_val_split/train.json',
                    img_prefix='data/train_val_split/train',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='Resize',
                            img_scale=[(2048, 800), (2048, 1400)],
                            multiscale_mode='range',
                            keep_ratio=True),
                        dict(type='RandomFlip', flip_ratio=0.5),
                        dict(
                            type='AutoAugment',
                            policies=[[{
                                'type': 'Shear',
                                'prob': 0.5,
                                'level': 0
                            }],
                                      [{
                                          'type': 'ColorTransform',
                                          'prob': 0.5,
                                          'level': 6
                                      }]]),
                        dict(
                            type='Albu',
                            transforms=[
                                dict(
                                    type='RandomRotate90',
                                    always_apply=False,
                                    p=0.5),
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
                                dict(
                                    type='JpegCompression',
                                    quality_lower=85,
                                    quality_upper=95,
                                    p=0.2),
                                dict(type='ChannelShuffle', p=0.1),
                                dict(
                                    type='OneOf',
                                    transforms=[
                                        dict(type='Blur', blur_limit=3, p=1.0),
                                        dict(
                                            type='MedianBlur',
                                            blur_limit=3,
                                            p=1.0)
                                    ],
                                    p=0.1)
                            ],
                            bbox_params=dict(
                                type='BboxParams',
                                format='pascal_voc',
                                label_fields=['gt_labels'],
                                min_visibility=0.0,
                                filter_lost_elements=True),
                            keymap=dict(
                                img='image',
                                gt_masks='masks',
                                gt_bboxes='bboxes'),
                            update_pad_shape=False,
                            skip_img_without_anno=True),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='Pad', size_divisor=32),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels'])
                    ])
            ])),
    val=dict(
        type='CocoDataset',
        ann_file='data/train_val_split/val.json',
        img_prefix='data/train_val_split/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(2048, 800), (2048, 1000), (2048, 1200),
                           (2048, 1400)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/trademark/test/annotations/instances_val2017.json',
        img_prefix='data/trademark/test/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(2048, 800), (2048, 1000), (2048, 1200),
                           (2048, 1400)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='DetectoRS_ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'),
            style='pytorch')),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[1.0, 1.5, 2.5, 3.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor_custom',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            gc_context=True),
        bbox_head=[
            dict(
                type='SABLHead',
                num_classes=50,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                bbox_coder=dict(
                    type='BucketingBBoxCoder',
                    num_buckets=14,
                    scale_factor=1.7),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox_reg=dict(
                    type='SmoothL1Loss', beta=0.1, loss_weight=1.0)),
            dict(
                type='SABLHead',
                num_classes=50,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                bbox_coder=dict(
                    type='BucketingBBoxCoder',
                    num_buckets=14,
                    scale_factor=1.5),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox_reg=dict(
                    type='SmoothL1Loss', beta=0.1, loss_weight=1.0)),
            dict(
                type='SABLHead',
                num_classes=50,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                bbox_coder=dict(
                    type='BucketingBBoxCoder',
                    num_buckets=14,
                    scale_factor=1.3),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox_reg=dict(
                    type='SmoothL1Loss', beta=0.1, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
evaluation = dict(interval=1, metric='bbox', start=8)
auto_resume = False
gpu_ids = range(0, 8)
work_dir = './work_dirs/cascade_rcnn_r101_rfp_1x_all_trick'
