# _backend_args = None

tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=300))

img_scales = [(640, 640), (480, 480), (320, 320)]

#                                LoadImageFromFile
#                     /                 |                     \
# (RatioResize,LetterResize) (RatioResize,LetterResize) (RatioResize,LetterResize) # noqa
#        /      \                    /      \                    /        \
#  RandomFlip RandomFlip      RandomFlip RandomFlip        RandomFlip RandomFlip # noqa
#      |          |                |         |                  |         |
#  LoadAnn    LoadAnn           LoadAnn    LoadAnn           LoadAnn    LoadAnn
#      |          |                |         |                  |         |
#  PackDetIn  PackDetIn         PackDetIn  PackDetIn        PackDetIn  PackDetIn # noqa

_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=s),
            dict(
                type='LetterResize',
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]) for s in img_scales
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='mmdet.RandomFlip', prob=0., direction='vertical')
            ],
            [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param',
                               'flip',
                               'flip_direction')
                )
            ]
        ])
]
