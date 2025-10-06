_base_ = ["../custom_imports.py",
          "architectures/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco.py"]

default_scope = 'mmyolo'

num_classes = 14
# img_scale = (640, 640)
save_epoch_intervals = 5
max_epochs = 5
close_mosaic_epochs = 10
weight_decay = 0.0005
lr_factor = 0.01  # Learning rate scaling factor
affine_scale = 0.9  # YOLOv5RandomAffine scaling ratio
max_keep_ckpts = 3
base_lr = 0.0005
batch_size = 8
persistent_workers = True

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            ),
        ),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            )))

#  Augmentation pipeline
last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=_base_.albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', direction='horizontal', prob=0.5),
    dict(type='mmdet.RandomFlip', direction='vertical', prob=0.0),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [
    *_base_.pre_transform,
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.1, 1.9),
        max_aspect_ratio=100,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]


train_pipeline_stage2 = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)), *last_transform
]

dataset_type = 'YOLOv5VinDRCXRDataset'

# YOLO-specific dataset parameters
batch_size = 8
num_workers = 8
test_ann_file = 'data/test.json'
val_ann_file = test_ann_file
data_root = ''  # Root path of data
train_data_prefix = 'train/'
val_data_prefix = train_data_prefix  # Prefix of val image path
test_data_prefix = 'test/'


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # metainfo=dict(classes=classes, palette=palette),
        data_prefix=dict(img=train_data_prefix),
        ann_file=test_ann_file,
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=val_data_prefix),
        # metainfo=dict(classes=classes, palette=palette),
        ann_file=test_ann_file,
        pipeline=_base_.test_pipeline,
        # batch_shapes_cfg=batch_shapes_cfg
    ))

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=test_data_prefix),
        # metainfo=dict(classes=classes, palette=palette),
        ann_file=test_ann_file,
        pipeline=_base_.test_pipeline,
        # batch_shapes_cfg=batch_shapes_cfg
    ))

# Model training and testing settings

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=batch_size),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.0001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.5),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + test_ann_file,
    metric='bbox', 
    format_only=False)
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

gpu_ids = [0]
seed = 0
device = "cuda"
log_level = 'INFO'  # The level of logging.


load_from = "https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth"  # YOLOv8x pretrained weights
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)] 
work_dir = 'logs/vindrcxr_yolov8_x'
