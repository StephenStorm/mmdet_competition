_base_ = './tianchi_htc_swinb_base_model.py'




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
            
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=250,
#     warmup_ratio=0.001,
#     step=[16, 19])

# learning policy
# lr_config = dict(step=[16, 19])
# runner = dict(type='EpochBasedRunner', max_epochs=20)

checkpoint_config = dict(interval=4)
evaluation = dict(interval=1, metric='bbox', start = 8)
work_dir = './work_dirs/htc_swinb_1x_fpn'
