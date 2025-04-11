# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth'
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
default_scope = 'mmdet'
backend_args = None
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=160,
        depths=[5, 5, 22, 5],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[160, 320, 640, 1280],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
]))

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='RandomChoiceResize',
                      scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      keep_ratio=True)
             ],
             [
                 dict(type='RandomChoiceResize',
                      scales=[(400, 1333), (500, 1333), (600, 1333)],
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='RandomChoiceResize',
                      scales=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      keep_ratio=True)
             ]
         ]),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
print('dosao')
# we use 4 nodes to train this model, with a total batch size of 64
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_valid.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
# optimizer
# optimizer = dict(
#     _delete_=True, type='AdamW', lr=0.0001 * 2, weight_decay=0.05,
#     constructor='CustomLayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(num_layers=37, layer_decay_rate=0.90,
#                        depths=[5, 5, 22, 5], offset_lr_scale=0.01))
# optimizer_config = dict(grad_clip=None)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg={
        'num_layers': 37,
        'layer_decay_rate': 0.90,
        'depths': [5, 5, 22, 5],
        'offset_lr_scale': 0.01
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001 * 2,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ))
# fp16 = dict(loss_scale=dict(init_scale=512))
# evaluation = dict(save_best='auto')
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=5,
    save_last=True,
)
resume_from = None

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    dict(type='SyncBuffersHook'),
    dict(type='ProfilerHook', on_trace_ready=dict(type='tb_trace'))
]
