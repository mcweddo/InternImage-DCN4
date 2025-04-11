# yapf:disable
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True,
                    save_best=['segm_mAP', 'acc'], rule='greater',
                    max_keep_ckpts=5,
                    filename_tmpl='internimage_L_epoch{}.pth'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='NeptuneVisBackend',
         init_kwargs={
            'project': 'ClaimCompanion/CoreModels',
            'api_token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiYjY0YWE5MC1kM2JmLTQ2MTYtODUwYy1lNTRkOWFjM2U2NzQifQ=='
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Optional: set moving average window size
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
# yapf:enable

#
# log_level = 'INFO'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork',
                opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_level = 'INFO'
load_from = None
resume = False
