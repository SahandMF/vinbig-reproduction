load_from = 'yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth'

_base_ = 'yolov8_x_syncbn_fast_8xb16-500e_coco.py'

dataset_type = 'VindrCXRDataset'
data_root = '/media/arndt_ro/public_datasets/vindr_cxr/'

classes = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
            'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
            'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
            'Pulmonary fibrosis'
]

base_lr = 0.0005

num_classes = 14
palette = [(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
           (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
           (255, 96, 55), (50, 183, 250)]

img_scale = (768, 768)  # width, height

seed = 42

max_aspect_ratio = 100

affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio

max_epochs = 40

close_mosaic_epochs = 5

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.0),
    # HSV color space augmentation
    dict(
        type='mmdet.ColorTransform',
        prob=0.4,
        level=3
    ),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    # Geometric transformations
    dict(type='mmdet.Resize',
         scale=(768, 768),
         keep_ratio=False),
    # Random affine transformations

    dict(type='mmdet.Pad', size_divisor=32),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape')
    )
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize',
         scale=(768, 768),
         keep_ratio=False),
    dict(
        type='mmdet.RandomFlip',
        prob=0.0,  # No actual flipping
        direction=['horizontal']
    ),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')
    )
]

val_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=6,
    num_workers=20,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_root=data_root,
        ann_file='train.json',
        metainfo = dict(classes = classes, palette = palette),
        data_prefix=dict(img='train/'),
        pipeline = train_pipeline
    ))

val_dataloader = dict(
    batch_size=6,
    num_workers=20,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        ann_file='val.json',
        metainfo = dict(classes = classes, palette = palette),
        data_prefix=dict(img='train/'),
        pipeline = val_pipeline
    ))


model = dict(
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            ),
        loss_cls=dict(
           type='mmdet.FocalLoss',
           use_sigmoid=True,
           gamma=2.0,
           alpha=0.25
       )
    ),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            )))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.001,  # lr0: 0.001
        momentum=0.937,  # momentum: 0.937
        weight_decay=0.0005,  # weight_decay: 0.0005
        nesterov=True
    )
)

# Parameter scheduler configuration
param_scheduler = [
    dict(
        # Warmup settings
        type='mmdet.LinearLR',
        start_factor=0.1,  # warmup_bias_lr: 0.1
        by_epoch=True,
        begin=0,
        end=3.0,  # warmup_epochs: 3.0
        convert_to_iter_based=True
    ),
    dict(
        # Main training phase - OneCycleLR
        type='OneCycleLR',
        eta_max=0.01,  # lr0: 0.01
        final_div_factor=100,  # lrf: 0.01 (final_lr = lr0 * lrf)
        pct_start=0.3,
        by_epoch=True,
        begin=3.0,  # starts after warmup
        end=1  # total epochs
    )
]

num_classes = 14

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'val.json',
    metric=['bbox'],
    format_only=False)

default_scope = 'mmyolo'


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': 'vindr-cxr-detection',
            'name': 'vfnet-experiment',
            'entity': 'husain-qasim-ali-saif-bauhaus-universit-t-weimar',  # Replace with your wandb username/entity
        }
    )
]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

work_dir = './work_dirs/vindr_cxr_yolov8'