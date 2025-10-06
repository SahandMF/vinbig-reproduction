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

data_root = '/home/sahand/Retmed-Latest/retmed/data/vinbig_cxr2'
annotation_root = 'home/sahand/Retmed-Latest/retmed/data/sample_wbf/wbf_data/train_val_split'
backend_args = None


persistent_workers = True
mixup_prob = 0.5

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.001
max_epochs = 150
val_interval = 2

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
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))

train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=val_interval)

work_dir = '/home/sahand/vinbig_outputs/vindrcxr_yolov5_x'

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=14)
    )
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='YOLOv5CocoDataset',
        ann_file='data/test.json',
        data_root='/home/sahand/cxr-detection-replication/cxr_detection_replication/data/vinbig_cxr2',
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(640, 640), keep_ratio=False),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
            )
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
                     (255, 96, 55), (50, 183, 250)]
        ),
        test_mode=True
    )
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='data/test.json',
    metric='bbox',
    classwise=True
)