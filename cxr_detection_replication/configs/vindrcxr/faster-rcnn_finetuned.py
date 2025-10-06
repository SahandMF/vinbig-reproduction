load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'

_base_ = ['mmdet::faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py']

# Dataset config
classes = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
            'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
            'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
            'Pulmonary fibrosis'
]

seed = 0
num_classes = 14
palette = [(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
           (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
           (255, 96, 55), (50, 183, 250)]
data_root = '/home/sahand/cxr-detection-replication/cxr_detection_replication/data/vinbig_cxr2'
annotation_root = '/home/sahand/Retmed-Latest/retmed/data/sample_cv/wbf_data/cv_folds'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
   dict(type='RandomFlip', prob=0.0),

    dict(
        type='PhotoMetricDistortion',  # RandomBrightnessContrast equivalent
        brightness_delta=32,  # Adjust brightness variation
        contrast_range=(0.5, 1.5),  # Adjust contrast variation
    ),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(
        type='RandomFlip',
        prob=0.0,  # No actual flipping
        direction=['horizontal']
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        data_root='',
        ann_file= 'train.json',
        data_prefix=dict(img=data_root + 'train/'),
        metainfo = dict(classes = classes, palette = palette),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root='',
        ann_file='val.json',
        data_prefix=dict(img=data_root + 'train/'),
        metainfo = dict(classes = classes, palette = palette),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# Model config
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=14)
    )
)

# Training config
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
val_cfg = dict(type='ValLoop')

# Evaluator config for validation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)

# Override the optimizer settings from base config
_base_.optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001,
        eps=1e-8,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=35, norm_type=2))


# Learning rate config
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',  # Replace WarmupCosineLR with this
        T_max=1,  # Total epochs
        eta_min=0,  # Minimum learning rate
        by_epoch=True
    )
]



# Runtime config
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)


test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        ann_file='data/test.json',
        data_root='/home/sahand/cxr-detection-replication/cxr_detection_replication/data/vinbig_cxr2',
        pipeline=test_pipeline,
        metainfo=dict(classes=classes, palette=palette),
        test_mode=True
    )
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/test.json',
    metric='bbox',
    classwise=True
)