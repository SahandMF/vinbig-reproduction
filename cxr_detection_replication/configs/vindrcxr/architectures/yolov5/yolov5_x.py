# This is modified file from mmyolo repository:
_base_ = ['yolov5_x-v61_syncbn_fast_8xb16-300e_coco.py']


load_from = "https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_x-v61_syncbn_fast_8xb16-300e_coco/yolov5_x-v61_syncbn_fast_8xb16-300e_coco_20230305_152943-00776a4b.pth"  # YOLOv5x pretrained weights

# ========================Frequently modified parameters======================
data_root = ''  # Root path of data
train_data_prefix = '/media/klee_ro/public_datasets/vindr_cxr/train/'  # Prefix of train image path
test_ann_file = 'data/test.json'
val_data_prefix = train_data_prefix  # Prefix of val image path
test_data_prefix = '/media/klee_ro/public_datasets/vindr_cxr/test/'

num_classes = 14  # Number of classes for classification
batch_size = 8
# Batch size of a single GPU during training
train_batch_size_per_gpu = batch_size
# Worker to pre-fetch data for each single GPU during training
train_num_workers = batch_size
# persistent_workers must be False if num_workers is 0
persistent_workers = True

mixup_prob = 0.5

classes = ("Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD",
           "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
           "Pneumothorax", "Pulmonary fibrosis")

palette = [(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
           (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
           (255, 96, 55), (50, 183, 250)]


# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.001
max_epochs = 1

# ========================Possible modified parameters========================
# Batch size of a single GPU during validation
val_batch_size_per_gpu = batch_size
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = batch_size
val_interval = 5

# -----train val related-----
# affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio

loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0

# The obj loss weights of the three output layers
obj_level_weights = [4., 1., 0.4]
lr_factor = 0.1  # Learning rate scaling factor

weight_decay = 0.0005
# Save model checkpoint and validation intervals
save_checkpoint_intervals = 5
# The maximum checkpoints to keep.
max_keep_ckpts = 10

img_scale = (640, 640)

# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)

# ===============================Unmodified in most cases====================
num_det_layers = _base_.num_det_layers

model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight * (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight *
                        ((img_scale[0] / 640) ** 2 * 3 / num_det_layers)),
    )
)


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
        max_rotate_degree=5.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
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
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))

train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=val_interval)
train_dataloader = dict(
    batch_size = train_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo = dict(classes=classes, palette=palette),
        pipeline = train_pipeline,
        data_prefix=dict(img=train_data_prefix),
    ))

val_dataloader = dict(
    batch_size = val_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo = dict(classes=classes, palette=palette),
        data_prefix=dict(img=val_data_prefix),
        # batch_shapes_cfg=batch_shapes_cfg
    ))

test_batch_size_per_gpu = 16

# Worker to pre-fetch data for each single GPU during training
test_num_workers = 2

test_dataloader = dict(
    batch_size = test_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes, palette=palette),
        data_prefix=dict(img=test_data_prefix),
        ann_file=test_ann_file,
    ))


# `paramwise_cfg` allows to apply stratified rates to the selected part of the model
optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        base_total_batch_size=train_batch_size_per_gpu,
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]


# In order to enable test.py, test_evaluator should be explicitly defined with an annotation file containing the names of images to be processed during inference
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + test_ann_file,
    metric='bbox',
    format_only=False)
