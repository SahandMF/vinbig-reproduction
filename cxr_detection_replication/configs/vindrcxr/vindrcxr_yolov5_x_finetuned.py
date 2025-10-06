custom_imports = dict(
    imports=[
        'mmdet.datasets',
        'mmdet.evaluation.metrics'
    ],
    allow_failed_imports=False
)

_base_ = ["../custom_imports.py", "architectures/yolov5/yolov5_x-v61_syncbn_fast_8xb16-300e_coco.py"]
load_from = "https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_x-v61_syncbn_fast_8xb16-300e_coco/yolov5_x-v61_syncbn_fast_8xb16-300e_coco_20230305_152943-00776a4b.pth"  # YOLOv5x pretrained weights


seed = 0
device = "cuda"
log_level = 'INFO'  # The level of logging.

data_root = ''
annotation_root = ''
backend_args = None


persistent_workers = True
mixup_prob = 0.5

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.001
max_epochs = 50
val_interval = 5

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 5
# The maximum checkpoints to keep.
max_keep_ckpts = 3

img_scale = (640, 640)

pre_transform = _base_.pre_transform
albu_train_transforms = _base_.albu_train_transforms


affine_scale = 0.9
mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0,
        max_shear_degree=0,
        scaling_ratio_range=(0.5, 1.5),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

# enable mixup
train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_pipeline]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
            'img_path': 'img_path'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction')
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        save_best='auto',
        max_keep_ckpts=3,
    ),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='YOLOv5ParamSchedulerHook', scheduler_type='linear', lr_factor=0.01, max_epochs=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'),
)

train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=val_interval)

work_dir = ''

# --- Model definition ---
model = dict(
    type='YOLODetector',
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=1.33,
        widen_factor=1.25,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=1.33,
        widen_factor=1.25,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=14,
            in_channels=[256, 512, 1024],
            featmap_strides=[8, 16, 32],
            num_base_priors=3,
            widen_factor=1.25,
        ),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]
            ],
            strides=[8, 16, 32],
        ),
        prior_match_thr=4,
        obj_level_weights=[4, 1, 0.4],
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.0875,
            reduction='mean',
        ),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            loss_weight=0.05,
            reduction='mean',
            return_iou=True,
            eps=1e-7,
        ),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='mean',
        ),
    ),
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
    ),
    test_cfg=dict(
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300,
        multi_label=True,
    ),
)

# --- Optimizer ---
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.843,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=8,
    ),
    constructor='YOLOv5OptimizerConstructor',
    paramwise_cfg=dict(
        norm_decay_mult=0,
        custom_keys=dict(
            backbone=dict(lr_mult=0.5, decay_mult=1),
        ),
        base_total_batch_size=8,
    ),
)

# --- Scheduler ---
# --- Custom hooks ---
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49),
]

# --- Data pipelines ---
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
]

mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114,
        pre_transform=pre_transform,
    ),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0,
        max_shear_degree=0,
        scaling_ratio_range=(0.5, 1.5),
        border=(-320, -320),
        border_val=(114, 114, 114),
    ),
]

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01),
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=0.5,
        pre_transform=[*pre_transform, *mosaic_affine_pipeline],
    ),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmdet.PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), hue_delta=18, saturation_range=(0.5, 1.5)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction'),
    ),
]

# --- Dataloaders ---
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='YOLOv5CocoDataset',
        ann_file='data/ann_agreement_orig/label-4/train_fold_2.json',
        data_root='',
        data_prefix=dict(img='/media/klee_ro/public_datasets/vindr_cxr/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        ann_file='',
        data_root='',
        data_prefix=dict(img=''),
        batch_shapes_cfg=dict(type='BatchShapePolicy', batch_size=1, img_size=640, size_divisor=32, extra_pad_ratio=0.5),
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
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(type='LetterResize', scale=(640, 640), allow_scale_up=False, pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')),
        ],
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        ann_file='data/test.json',
        data_root='',
        batch_shapes_cfg=dict(type='BatchShapePolicy', batch_size=1, img_size=640, size_divisor=32, extra_pad_ratio=0.5),
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
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(type='LetterResize', scale=(640, 640), allow_scale_up=False, pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')),
        ],
        test_mode=True,
    ),
)

# --- Evaluators ---
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='',
    metric='bbox',
    proposal_nums=[100, 1, 10],
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='data/test.json',
    metric='bbox',
    proposal_nums=[100, 1, 10],
)

# --- TTA Configuration ---
tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300,
        multi_label=True
    )
)

# TTA image scales
img_scales = [(640, 640), (800, 800), (480, 480)]

# Multiscale resize transforms for TTA
_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=s),
            dict(
                type='LetterResize',
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114)
            )
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
                dict(type='mmdet.RandomFlip', prob=0.0),
                dict(type='mmdet.RandomFlip', prob=0.0)
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