_base_ = './tianchi_htc_without_semantic_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
)


# train_pipeline = [
#     dict(
#         type='Collect',
#         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
# ]
data = dict(
    samples_per_gpu=4,
)
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
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)

checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metric='bbox', start = 1)
work_dir = './work_dirs/htc_r101_20e_fpn'
