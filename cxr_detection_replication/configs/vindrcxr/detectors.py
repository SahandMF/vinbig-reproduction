load_from = 'https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth'

_base_ = ['mmdet::detectors/detectors_cascade-rcnn_r50_1x_coco.py']

default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=5, type='CheckpointHook'))

backend_args = None
max_epochs = 5
train_cfg = dict(
    _scope_='mmdet', max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
