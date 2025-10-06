# Aggressive DETR config for CXR detection - optimized for maximum learning
seed = 0
# Dataset settings
dataset_type = 'CocoDataset'
data_root = None
annotation_root = None
backend_args = None

# Image scale and batch size
img_scale = (800, 800)
batch_size = 1  # Single image per batch for maximum gradient precision

# Model settings
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
        out_indices=(3,),
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
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
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
    decoder=dict(
        return_intermediate=True,
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(
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
    positional_encoding=dict(num_feats=128, normalize=True),
    bbox_head=dict(
        type='DETRHead',
        num_classes=14,
        embed_dims=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ])),
    test_cfg=dict(max_per_img=100))

# AGGRESSIVE Data pipeline with more augmentation
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Data loaders
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,  # Reduced for single batch
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=None,
        ann_file=None,  # Will be set by train.py
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
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=None,
        ann_file=None,  # Will be set by train.py
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
        pipeline=val_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=None,
        ann_file=None,  # Will be set by train.py
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

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=None,  # Will be set by train.py
    metric='bbox',
    backend_args=None)

test_evaluator = dict(
    type='CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=None,  # Will be set by train.py
    metric='bbox',
    backend_args=None)

# AGGRESSIVE Training settings
base_lr = 0.0005  # Higher learning rate for faster learning
max_epochs = 30   # Shorter training for faster iteration
val_interval = 1  # Validate every epoch
save_checkpoint_intervals = 1
max_keep_ckpts = 3

# AGGRESSIVE Optimizer settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.0001,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1, decay_mult=1.0),
        )),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# AGGRESSIVE Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500),  # Shorter warmup
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        eta_min=base_lr * 0.01)  # Cosine annealing to 1% of base_lr
]

# Load pretrained model
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
resume = False

# Auto scale learning rate
auto_scale_lr = dict(base_batch_size=batch_size)

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),  # More frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Logging
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)  # Smaller window for faster feedback

# Seed
seed = 0
deterministic = False

# Custom imports for DETR components
custom_imports = dict(
    allow_failed_imports=True,
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
        'mmdet.datasets.coco',
    ])

# Runtime settings
launcher = 'none'
work_dir = None
backend_args = None

# Training and testing settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer') 