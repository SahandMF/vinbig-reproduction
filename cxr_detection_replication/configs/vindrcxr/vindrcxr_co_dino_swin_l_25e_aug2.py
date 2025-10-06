_base_ = ['/home/sahand/cxr-detection-replication/cxr_detection_replication/projects/CO_DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py']

# Keep URL so it auto-downloads
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'


seed = 0
# Override data_preprocessor to bbox-only (no masks required)
model = dict(
    data_preprocessor=dict(
        pad_mask=False,
        batch_augments=None,
    ),
    query_head=dict(
        # Hardcode dataset classes
        num_classes=14,
        num_query=300,
        transformer=dict(
            decoder=dict(num_layers=4)
        ),
        # Stronger classification loss to overcome class imbalance
        loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=2.0),
        # Slightly stronger bbox losses to get gradients flowing
        loss_bbox=dict(type='L1Loss', loss_weight=10.0),
        loss_iou=dict(type='GIoULoss', loss_weight=4.0)
    )
)

# Dataloader paths are injected by tools/train_co_detr.py; override train pipeline to bbox-only (no masks)
backend_args = None

# Augmented training pipeline (aligned with last-run augmentation)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='mmdet.PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

# Force direct CocoDataset for training to avoid MultiImageMixDataset wrapper & mask requirements
train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        ann_file=None,
        data_root=None,
        data_prefix=dict(img='train/'),
        # Keep all images; some COCO exports may have sparse labels
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_evaluator = dict(type='CocoMetric', ann_file=None, metric='bbox')
test_evaluator = dict(type='CocoMetric', ann_file=None, metric='bbox')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

max_epochs = 25
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[6, 12], gamma=0.5)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3),
)

# Add early stopping via custom_hooks
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=5,
        min_delta=0.001,
    )
]

visualizer = dict(type='mmdet.DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

auto_scale_lr = dict(base_batch_size=1)


# Note: `roi_head` and `bbox_head` class counts are set by the training launcher
# to avoid overriding the list-structured heads from the base. If you want them
# hardcoded in the config as well, I can inline the full head definitions.

# Test pipeline mirrors 16e minimal setup
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        ann_file=None,  # Will be set by test.py
        data_root=None,  # Will be set by test.py
        data_prefix=dict(img='test/'),  # Assuming test images are in 'test/' subfolder
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(800, 800), keep_ratio=True),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ],
        metainfo=dict(
            classes=[
                'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
                'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
                'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                'Pulmonary fibrosis'
            ],
            palette=[(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
                     (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
                     (255, 96, 55), (50, 183, 250)]),
        backend_args=backend_args))
