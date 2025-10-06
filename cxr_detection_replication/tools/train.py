import argparse
import logging
import os
import os.path as osp
import sys
import subprocess
import torch
from torch import nn
import wandb
import json
from typing import Any
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower, register_all_modules
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS, MODELS
from mmdet.datasets.coco import CocoDataset
from cxr_detection_replication.retmed.datasets.vindrcxr import VinDRCXRDataset, YOLOv5VinDRCXRDataset
from cxr_detection_replication.retmed.utils.utils import (
    update_num_classes,
    update_dataset_type,
    add_root_dir_to_dataset_config,
    update_annotation_files,
    update_gpu_config,
    setup_amp_config,
    setup_resume_config,
    setup_wandb_config,
    update_batch_size,
    setup_config
)

# This train.py now supports multiple model types:
# - DETR: Standard DETR models with manual component registration
# - CoDETR: CO-DETR models with custom_imports and MultiImageMixDataset handling
# - Other models: Standard MMDetection models



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--trainer', choices=['mmdet', 'effdet'], default='mmdet',
                        help='Training backend to use: mmdet or effdet')
    parser.add_argument('config', nargs='?', default=None, type=str,
                        help='Path to the training configuration file (required for mmdet)')
    parser.add_argument('--ann-dir', type=str, default=None,
                        help='Directory where the annotation files are stored. For cross-validation, the directory should contain train_fold_{fold}.json and val_fold_{fold}.json files; for regular train-validation split, the directory should contain train.json and val.json files.')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Root path to the dataset directory. The directory should contain the folder train and test, which are used for both training and validation.')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Directory to save logs and models')
    parser.add_argument('--use-cv', action='store_true', help='Enable cross-validation training strategy')
    parser.add_argument('--cv', type=int, default=4,
                        help='Number of cross-validation splits')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable automatic-mixed-precision training')
    parser.add_argument('--resume', nargs='?', type=str, const='auto',
                        help='Resume from checkpoint path or auto-resume from latest')
    parser.add_argument('--gpu-ids', default=None, type=int, nargs="+",
                        help="Override GPU settings from config")
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='retmed',
                        help='Weights & Biases project name (default: retmed)')
    parser.add_argument('--use-yolo', action='store_true',
                        help='Use YOLOv5 dataset format (default: False)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for training and validation (default: use config value)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')

    # EfficientDet options (used when --trainer effdet)
    effdet = parser.add_argument_group('EfficientDet options')
    effdet.add_argument('--effdet-root', type=str, default=None,
                        help='Dataset root for EfficientDet (e.g., .../data/vinbig_cxr2)')
    effdet.add_argument('--effdet-dataset', type=str, default='coco',
                        help='Dataset name for EfficientDet (default: coco)')
    effdet.add_argument('--effdet-model', type=str, default=None,
                        help='EfficientDet model name (e.g., tf_efficientdet_d2)')
    effdet.add_argument('--effdet-num-classes', type=int, default=None,
                        help='Number of classes for EfficientDet')
    effdet.add_argument('--effdet-initial-checkpoint', type=str, default='',
                        help='Initial checkpoint for EfficientDet fine-tuning')
    effdet.add_argument('--effdet-batch-size', type=int, default=8,
                        help='Batch size for EfficientDet')
    effdet.add_argument('--effdet-epochs', type=int, default=25,
                        help='Epochs for EfficientDet')
    effdet.add_argument('--effdet-workers', type=int, default=4,
                        help='Num workers for EfficientDet')
    effdet.add_argument('--effdet-amp', action='store_true', default=False,
                        help='Use AMP for EfficientDet')
    effdet.add_argument('--effdet-output', type=str, default='',
                        help='Output dir for EfficientDet run')
    effdet.add_argument('--effdet-ann-train-file', type=str, default=None,
                        help='Optional COCO train JSON override for EfficientDet')
    effdet.add_argument('--effdet-ann-val-file', type=str, default=None,
                        help='Optional COCO val JSON override for EfficientDet')
    effdet.add_argument('--effdet-use-cv', action='store_true', default=False,
                        help='Run EfficientDet cross-validation loop')
    effdet.add_argument('--effdet-cv', type=int, default=5, help='Number of folds for EfficientDet CV')
    effdet.add_argument('--effdet-cv-dir', type=str, default=None,
                        help='Directory containing train_fold_{k}.json and val_fold_{k}.json for EfficientDet CV')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def train_model(cfg, model_name):
    """Train model using the configured runner."""
    # Register modules and ensure proper metainfo
    register_all_modules()
    
    # Debug: Check if DETR is available before building
    if cfg.model.type == 'DETR':
        from mmdet.registry import MODELS
        print(f"Before Runner.from_cfg - DETR in MODELS: {'DETR' in MODELS.module_dict}")
        print(f"Model type: {cfg.model.type}")
        print(f"Available models (first 10): {list(MODELS.module_dict.keys())[:10]}")
        print("DETR components should already be registered from main()")
    
    is_metainfo_lower(cfg)

    # Build and start the runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.train()

def main():
    args = parse_args()
    if args.trainer == 'mmdet':
        register_all_modules()
        setup_cache_size_limit_of_dynamo()

    if args.trainer == 'mmdet':
        if not args.ann_dir or not args.data_path:
            raise ValueError('--ann-dir and --data-path are required for mmdet trainer')
        # Load base configuration
        if not args.config:
            raise ValueError('Config file is required for mmdet trainer')
        base_cfg = Config.fromfile(args.config)
        base_cfg.launcher = args.launcher
    
    if args.trainer == 'mmdet':
        # Debug: Check what model type we have
        print(f"Model type in config: {base_cfg.model.type}")
    
    # Register DETR components early if it's a DETR model
    if args.trainer == 'mmdet' and base_cfg.model.type == 'DETR':
        print("Registering DETR components early...")
        from mmdet.models.detectors import DETR
        from mmdet.models.data_preprocessors import DetDataPreprocessor
        from mmdet.models.backbones import ResNet
        from mmdet.models.necks import ChannelMapper
        from mmdet.models.dense_heads import DETRHead
        from mmdet.models.task_modules.assigners import HungarianAssigner
        from mmdet.models.task_modules.assigners.match_cost import ClassificationCost, BBoxL1Cost, IoUCost
        from mmdet.models.losses import CrossEntropyLoss, L1Loss, GIoULoss
        from mmdet.datasets import CocoDataset
        from mmdet.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
        from mmdet.datasets.transforms import RandomFlip, Resize, PackDetInputs
        from mmdet.evaluation import CocoMetric
        from mmengine.registry import MODELS, TASK_UTILS, DATASETS, TRANSFORMS, METRICS
        
        # Register all necessary components for DETR
        if 'DETR' not in MODELS.module_dict:
            MODELS.register_module(name='DETR', module=DETR)
        if 'DetDataPreprocessor' not in MODELS.module_dict:
            MODELS.register_module(name='DetDataPreprocessor', module=DetDataPreprocessor)
        if 'ResNet' not in MODELS.module_dict:
            MODELS.register_module(name='ResNet', module=ResNet)
        if 'ChannelMapper' not in MODELS.module_dict:
            MODELS.register_module(name='ChannelMapper', module=ChannelMapper)
        if 'DETRHead' not in MODELS.module_dict:
            MODELS.register_module(name='DETRHead', module=DETRHead)
        if 'HungarianAssigner' not in TASK_UTILS.module_dict:
            TASK_UTILS.register_module(name='HungarianAssigner', module=HungarianAssigner)
        if 'ClassificationCost' not in TASK_UTILS.module_dict:
            TASK_UTILS.register_module(name='ClassificationCost', module=ClassificationCost)
        if 'BBoxL1Cost' not in TASK_UTILS.module_dict:
            TASK_UTILS.register_module(name='BBoxL1Cost', module=BBoxL1Cost)
        if 'IoUCost' not in TASK_UTILS.module_dict:
            TASK_UTILS.register_module(name='IoUCost', module=IoUCost)
        if 'CrossEntropyLoss' not in MODELS.module_dict:
            MODELS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
        if 'L1Loss' not in MODELS.module_dict:
            MODELS.register_module(name='L1Loss', module=L1Loss)
        if 'GIoULoss' not in MODELS.module_dict:
            MODELS.register_module(name='GIoULoss', module=GIoULoss)
        if 'CocoDataset' not in DATASETS.module_dict:
            DATASETS.register_module(name='CocoDataset', module=CocoDataset)
        
        # Register transforms
        transforms_to_register = [
            ('LoadImageFromFile', LoadImageFromFile),
            ('LoadAnnotations', LoadAnnotations),
            ('RandomFlip', RandomFlip),
            ('Resize', Resize),
            ('PackDetInputs', PackDetInputs)
        ]
        for name, transform in transforms_to_register:
            if name not in TRANSFORMS.module_dict:
                TRANSFORMS.register_module(name=name, module=transform)
        
        # Register metrics
        if 'CocoMetric' not in METRICS.module_dict:
            METRICS.register_module(name='CocoMetric', module=CocoMetric)
        
        print("✓ All DETR components registered early")
    
    # Register CO-DETR components early if it's a CO-DETR model
    elif args.trainer == 'mmdet' and base_cfg.model.type == 'CoDETR':
        print("Registering CO-DETR components early...")
        # Import CO-DETR components directly to ensure they're registered
        proj_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
        pkg_root = osp.abspath(osp.join(proj_root, '..'))
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
        
        # Import CO-DETR components to register them
        try:
            from cxr_detection_replication.projects.CO_DETR.codetr import (
                CoDETR, CoDINOHead, CoStandardRoIHead, CoATSSHead
            )
            from cxr_detection_replication.projects.CO_DETR.codetr.transformer import (
                CoDinoTransformer, DetrTransformerEncoder, DinoTransformerDecoder
            )
            print("✓ CO-DETR components imported and registered successfully")
            print(f"  - Model type: {base_cfg.model.type}")
            print(f"  - Components: CoDETR, CoDINOHead, CoStandardRoIHead, CoATSSHead")
        except ImportError as e:
            print(f"⚠ Warning: Could not import CO-DETR components: {e}")
            print("  - Will rely on custom_imports in config")
    
    if args.trainer == 'mmdet':
        # Update number of classes
        base_cfg = update_num_classes(base_cfg, len(VinDRCXRDataset.METAINFO['classes']))
        base_cfg = update_dataset_type(base_cfg)
        base_cfg = add_root_dir_to_dataset_config(base_cfg, args.data_path)
        base_cfg = update_batch_size(base_cfg, args.batch_size)
    
    # Extract model name
    model_name = osp.splitext(osp.basename(args.config))[0] if args.trainer == 'mmdet' else args.effdet_model or 'effdet'
    
    # Create default work directory with timestamp if not specified (mmdet only)
    if args.trainer == 'mmdet' and args.work_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Get strategy from annotation directory name
        strategy = osp.basename(args.ann_dir)
            
        # Structure: work_dirs/model_name/strategy_timestamp
        args.work_dir = osp.join('work_dirs', 
                               model_name,
                               f'{strategy}_{timestamp}')
        print(f"No work directory specified. Using default: {args.work_dir}")
        os.makedirs(args.work_dir, exist_ok=True)
    
    # Train with chosen strategy
    if args.trainer == 'mmdet' and args.use_cv:
        print_log(f"Starting cross-validation training with {args.cv} folds for model {model_name}",
                  logger='current', level=logging.INFO)
        
        # Train each fold
        for fold in range(args.cv):
            print_log(f"Training fold {fold + 1}/{args.cv} for model {model_name}",
                      logger='current', level=logging.INFO)
            
            # Setup configuration for this fold
            cfg = setup_config(base_cfg, args, fold=fold, model_name=model_name)
            
            # Train the model for this fold
            train_model(cfg, model_name)
            
            print_log(f"Completed training for fold {fold + 1}/{args.cv}",
                      logger='current', level=logging.INFO)
    elif args.trainer == 'mmdet':
        print_log(f"Starting regular train-validation training for model {model_name}",
                  logger='current', level=logging.INFO)
        
        # Setup configuration for regular training
        cfg = setup_config(base_cfg, args, model_name=model_name)
        
        # Debug: Check model type after setup_config
        print(f"Model type after setup_config: {cfg.model.type}")
        
        # Train the model
        train_model(cfg, model_name)
        
        print_log(f"Completed training for model {model_name}",
                  logger='current', level=logging.INFO)

    else:
        # EfficientDet path - invoke the native trainer (orchestrated here)
        eff_root = args.effdet_root or args.data_path
        if eff_root is None:
            raise ValueError('Provide --effdet-root or --data-path for EfficientDet')

        train_py = osp.abspath(osp.join(osp.dirname(__file__), '..', 'efficientdet', 'train.py'))
        cv_py = osp.abspath(osp.join(osp.dirname(__file__), '..', 'efficientdet', 'train_cv.py'))

        base_cmd = [sys.executable]
        if args.effdet_use_cv:
            if not args.effdet_cv_dir:
                raise ValueError('--effdet-cv-dir is required when --effdet-use-cv is set')
            cmd = base_cmd + [cv_py, eff_root, '--cv', str(args.effdet_cv), '--cv-dir', args.effdet_cv_dir]
            if args.effdet_output:
                cmd += ['--output-base', args.effdet_output]
        else:
            cmd = base_cmd + [train_py, eff_root]
            if args.effdet_output:
                cmd += ['--output', args.effdet_output]

        # Shared EfficientDet args
        if args.effdet_dataset:
            cmd += ['--dataset', args.effdet_dataset]
        if args.effdet_model:
            cmd += ['--model', args.effdet_model]
        if args.effdet_num_classes is not None:
            cmd += ['--num-classes', str(args.effdet_num_classes)]
        if args.effdet_initial_checkpoint:
            cmd += ['--initial-checkpoint', args.effdet_initial_checkpoint]
        if args.effdet_batch_size is not None:
            cmd += ['-b', str(args.effdet_batch_size)]
        if args.effdet_epochs is not None:
            cmd += ['--epochs', str(args.effdet_epochs)]
        if args.effdet_workers is not None:
            cmd += ['--workers', str(args.effdet_workers)]
        if args.effdet_amp:
            cmd += ['--amp']
        if args.effdet_ann_train_file and not args.effdet_use_cv:
            cmd += ['--ann-train-file', args.effdet_ann_train_file]
        if args.effdet_ann_val_file and not args.effdet_use_cv:
            cmd += ['--ann-val-file', args.effdet_ann_val_file]

        print('Launching EfficientDet:', ' '.join(cmd))
        raise SystemExit(subprocess.call(cmd))

if __name__ == '__main__':
    main()
