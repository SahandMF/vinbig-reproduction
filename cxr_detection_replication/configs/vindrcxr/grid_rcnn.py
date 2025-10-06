_base_ = ['mmdet::grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py']
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth'

gpu_ids = [1]
seed = 0
device = "cuda"
log_level = 'INFO'  # The level of logging.