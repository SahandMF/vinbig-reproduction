

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth'

seed = 0



_base_ = ['mmdet::vfnet/vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco.py']

# Dataset config
classes = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
            'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
            'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
            'Pulmonary fibrosis'
]


num_classes = 14
palette = [(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
           (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
           (255, 96, 55), (50, 183, 250)]

data_root = '/home/sahand/Retmed-Latest/retmed/data/vinbig_cxr2'
annotation_root = '/home/sahand/Retmed-Latest/retmed/data/sample_cv/wbf_data/cv_folds'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),  # Add this line
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

val_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        data_root='',
        ann_file= '/home/saifh/snap/snapd-desktop-integration/253/Desktop/retmed_saifh/data/folds2/train_fold_0.json',
        data_prefix=dict(img='/media/arndt_ro/public_datasets/vindr_cxr/train/'),
        metainfo=dict(classes=classes, palette=palette),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        serialize_data=False,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root='',
        ann_file='/home/saifh/snap/snapd-desktop-integration/253/Desktop/retmed_saifh/data/folds2/val_fold_0.json',
        data_prefix=dict(img='/media/arndt_ro/public_datasets/vindr_cxr/train/'),
        metainfo=dict(classes=classes, palette=palette),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        serialize_data=False))

# Model config
model = dict(
    bbox_head=dict(num_classes=14)  # Update number of classes for VFNet
)# Update number of classes

# Training config
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=18, val_interval=1)
val_cfg = dict(type='ValLoop')

# Evaluator config for validation
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/saifh/snap/snapd-desktop-integration/253/Desktop/retmed_saifh/data/folds2/val_fold_0.json',
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
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
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
    dist_cfg=dict(backend='nccl'),
    # Memory optimization settings
    deterministic=True,
    seed=0)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

work_dir = '/home/sahand/vinbig_outputs/vindr_cxr_vfnet_nb'


test_dataloader = dict(
    batch_size=8,  # Reduced from 8 to 4 for TTA memory efficiency (effective batch size = 8)
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

# --- TTA Configuration ---
tta_model = dict(
    type='mmdet.models.test_time_augs.DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=100,  # Reduced from 300 for VFNet memory efficiency
        multi_label=False  # VFNet is single-label
    )
)

# VFNet-specific TTA image scales (adapted from YOLOv5 scales)
img_scales = [(1333, 800), (1600, 960), (1000, 600)]  # VFNet base scale + variations

# VFNet multiscale resize transforms for TTA
_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='Resize', scale=s, keep_ratio=True),
            dict(type='Pad', size_divisor=32)
        ]
    ) for s in img_scales
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type='mmdet.RandomFlip', prob=0.0),  # Original
                dict(type='mmdet.RandomFlip', prob=0.0)   # Flipped
            ],
            [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param',
                               'flip', 'flip_direction')
                )
            ]
        ]
    )
]