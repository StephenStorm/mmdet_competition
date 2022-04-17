_base_ = [
    '../_base_/models/cascade_rcnn_swinb_fpn.py',
    '../_base_/datasets/tianchi_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    rpn_head=dict(
        loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
        # reg_decoded_bbox=True,      # 使用GIoUI时注意添加
        # loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

optimizer = dict(
    type='AdamW', 
    lr=0.0000125*8*4, 
    betas=(0.9, 0.999), 
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)}))
            
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=18)

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox', start=8)


load_from = None
resume_from = None
auto_resume = False

gpu_ids = range(8)

optimizer_config = dict(
    type='Fp16OptimizerHook', grad_clip=None, coalesce=True, bucket_size_mb=-1)
work_dir = './work_dirs/cascade_rcnn_swinb_fpn_3x_ms_albu_2lr_gc_context_rotate_autoaug_3batchsize_rpn_giou_fp16'