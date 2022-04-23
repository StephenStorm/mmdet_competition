_base_ = [
    '../_base_/models/cascade_rcnn_swinb_fpn_sobel.py',
    '../_base_/datasets/tianchi_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# swin can't run because of this command 
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)

optimizer = dict(
    type='AdamW', 
    lr=0.0000125*8*2, 
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

# optimizer_config = dict(
#     type='Fp16OptimizerHook', grad_clip=None, coalesce=True, bucket_size_mb=-1)
work_dir = './work_dirs/cascade_rcnn_swinb_fpn_all_trick_sobel'

# best workdir : mmdetection/work_dirs/cascade_rcnn_swinb_fpn_3x_ms_albu_2lr_gc_context_rotate_autoaug_2batchsize_250warmup