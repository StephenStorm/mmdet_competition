_base_ = [
    '../_base_/models/cascade_rcnn_r101_fpn.py',
    '../_base_/datasets/tianchi_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
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

evaluation = dict(interval=1, metric='bbox', start = 5)
checkpoint_config = dict(interval=4)

work_dir = './work_dirs/cascade_rcnn_r101_fpn_3x_ms_giou_alltrick'
gpu_ids = range(8)