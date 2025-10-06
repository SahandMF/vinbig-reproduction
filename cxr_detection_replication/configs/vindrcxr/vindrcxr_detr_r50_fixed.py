# Fixed DETR config for CXR detection - addressing gradient explosion and learning issues
# This config fixes the major issues identified in the training logs

# Custom imports to ensure all modules are registered
custom_imports = dict(
    imports=[
        'mmdet.models.detectors.detr',
        'mmdet.models.backbones.resnet',
        'mmdet.models.necks.channel_mapper',
        'mmdet.models.dense_heads.detr_head',
        'mmdet.models.task_modules.assigners.hungarian_assigner',
        'mmdet.models.task_modules.assigners.match_cost',
        'mmdet.models.losses.cross_entropy_loss',
        'mmdet.models.losses.l1_loss',
        'mmdet.models.losses.giou_loss',
        'mmdet.datasets.coco'
    ],
    allow_failed_imports=True
)

# Model configuration
model = dict(
    type='DETR',
    num_queries=100,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(  # DetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)))),
    decoder=dict(  # DetrTransformerDecoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, normalize=True),
    bbox_head=dict(
        type='DETRHead',
        num_classes=14,  # CXR classes
        embed_dims=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))

# Training configuration
seed = 0
device = "cuda"
log_level = 'INFO'

# Data paths - these will be overridden by command line arguments
data_root = None
annotation_root = None
backend_args = None

# Training parameters - FIXED for DETR
base_lr = 0.0001  # Increased back to reasonable value
max_epochs = 50   # Back to 50 epochs
val_interval = 5
save_checkpoint_intervals = 5
max_keep_ckpts = 3

# Image scale for DETR
img_scale = (800, 800)

# SIMPLE Train pipeline for DETR (using only available transforms)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='PackDetInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Default hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts,
    ),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'),
)

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Work directory
work_dir = None

# Dataloaders
train_dataloader = dict(
    batch_size=4,  # Reduced batch size to help with gradient stability
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        ann_file=None,
        data_root=None,
        data_prefix=dict(img=None),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=[
                'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
                'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
                'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                'Pulmonary fibrosis'
            ],
            palette=[(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
                     (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
                     (255, 96, 55), (50, 183, 250)]
        ),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,  # Reduced batch size
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        ann_file=None,
        data_root=None,
        data_prefix=dict(img=None),
        test_mode=True,
        metainfo=dict(
            classes=[
                'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
                'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
                'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                'Pulmonary fibrosis'
            ],
            palette=[(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
                     (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
                     (255, 96, 55), (50, 183, 250)]
        ),
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=4,  # Reduced batch size
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        ann_file=None,
        data_root=None,
        data_prefix=dict(img=None),
        test_mode=True,
        metainfo=dict(
            classes=[
                'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
                'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
                'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                'Pulmonary fibrosis'
            ],
            palette=[(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
                     (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
                     (255, 96, 55), (50, 183, 250)]
        ),
        pipeline=test_pipeline,
        backend_args=backend_args))

# Evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=None,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=None,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

# FIXED Optimizer - addressing gradient explosion
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),  # Stronger gradient clipping
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# FIXED Learning policy
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# FIXED Learning rate scheduler - simpler and more stable
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 40],  # Simpler milestones
        gamma=0.1)
]

# Auto scale learning rate
auto_scale_lr = dict(base_batch_size=4)  # Adjusted for new batch size

# Load from pretrained model (optional)
load_from = None
resume = False 