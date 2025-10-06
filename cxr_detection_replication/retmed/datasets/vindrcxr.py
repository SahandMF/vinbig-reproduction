from mmengine.registry import MODELS, DATASETS
from mmyolo.datasets import YOLOv5CocoDataset
from mmdet.datasets import CocoDataset

@DATASETS.register_module()
@MODELS.register_module()
class VinDRCXRDataset(CocoDataset):
    METAINFO = {
        'classes': ("Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD",
                "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
                "Pneumothorax", "Pulmonary fibrosis"),

        'palette': [(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
                (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
                (255, 96, 55), (50, 183, 250)] 
    }

@DATASETS.register_module()
@MODELS.register_module()
class YOLOv5VinDRCXRDataset(YOLOv5CocoDataset):
    METAINFO = {
        'classes': ("Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD",
                "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
                "Pneumothorax", "Pulmonary fibrosis"),

        'palette': [(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
                (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
                (255, 96, 55), (50, 183, 250)] 
    }
