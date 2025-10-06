import json
import mmengine
import os
import os.path as osp
import logging
import torch
import wandb
from typing import Any
from mmengine.config import Config
from mmengine.logging import print_log
from cxr_detection_replication.retmed.datasets.vindrcxr import VinDRCXRDataset

# Training utility functions
def _update(cfg: dict, key: str, value: Any):
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == key:
                cfg[k] = value
            elif isinstance(v, (dict, list)):
                _update(v, key, value)
    elif isinstance(cfg, list):
        for item in cfg:
            if isinstance(item, (dict, list)):
                _update(item, key, value)

def update_num_classes(config: Config, num_classes: int) -> Config:
    """
    Recursively finds and updates all occurrences of 'num_classes' in the configuration.

    Args:
        config (Config): The MMDetection configuration object.
        num_classes (int): The new number of classes.

    Returns:
        Config: The updated MMDetection configuration object.
    """
    _update(config._cfg_dict, 'num_classes', num_classes)
    return config

def update_dataset_type(config: Config) -> Config:
    """
    Updates the dataset type in the configuration based on the model type.
    - YOLO models use 'YOLOv5VinDRCXRDataset'
    - DETR models use 'CocoDataset'
    - Other models use 'VinDRCXRDataset'

    Args:
        config (Config): The MMDetection/MMYOLO configuration object.

    Returns:
        Config: The updated MMDetection/MMYOLO configuration object.
    """
    if 'model' in config and 'type' in config.model:
        model_type = config.model.type.lower()
        if 'yolo' in model_type:
            dataset_type_to_use = 'YOLOv5VinDRCXRDataset'
            print(f"Detected YOLO model type: '{config.model.type}'. Using dataset type: '{dataset_type_to_use}'.")
        elif 'detr' in model_type:
            dataset_type_to_use = 'CocoDataset'
            print(f"Detected DETR model type: '{config.model.type}'. Using dataset type: '{dataset_type_to_use}'.")
        else:
            dataset_type_to_use = 'VinDRCXRDataset'
            print(f"Detected non-YOLO/DETR model type: '{config.model.type}'. Using general dataset type: '{dataset_type_to_use}'.")
    else:
        dataset_type_to_use = 'VinDRCXRDataset'
        print(f"Could not determine model type. Using general dataset type: '{dataset_type_to_use}'.")

    config.dataset_type = dataset_type_to_use
    print(f"Updated dataset_type to: {dataset_type_to_use}")

    for dataloader_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        if dataloader_key in config and 'dataset' in config[dataloader_key]:
            config[dataloader_key].dataset.type = dataset_type_to_use
            print(f"Updated {dataloader_key}.dataset.type to: {dataset_type_to_use}")

    return config

def add_root_dir_to_dataset_config(config, data_path):
    """
    This function provides the root directory path and appends it in front of the image and annotations paths.
    
    Args:
        config (Config): The MMDetection configuration object
        data_path (str): The root path to the dataset directory
        
    Returns:
        Config: The updated configuration object
    """
    # Convert to absolute path
    data_path = osp.abspath(data_path)
    print(f"Updating dataset config with data_path: {data_path}")
    
    # Update data_root for each dataloader if it exists
    for dataloader_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        if hasattr(config, dataloader_key) and hasattr(getattr(config, dataloader_key), 'dataset'):
            dataset = getattr(config, dataloader_key).dataset
            # Set data_root to the base directory
            dataset.data_root = data_path
            print(f"Updated {dataloader_key}.dataset.data_root to: {data_path}")
            
            # Set appropriate data_prefix based on dataloader type
            if dataloader_key == 'train_dataloader':
                dataset.data_prefix = dict(img=data_path + '/train')
            elif dataloader_key == 'val_dataloader':
                dataset.data_prefix = dict(img=data_path + '/train')  # Validation uses train images
            elif dataloader_key == 'test_dataloader':
                dataset.data_prefix = dict(img=data_path + '/test')
            print(f"Updated {dataloader_key}.dataset.data_prefix to: {dataset.data_prefix}")

            # Ensure ann_file is not affected by data_root (only if it's not None)
            if hasattr(dataset, 'ann_file') and dataset.ann_file is not None:
                dataset.ann_file = osp.abspath(dataset.ann_file)
                print(f"Updated {dataloader_key}.dataset.ann_file to absolute path: {dataset.ann_file}")
            elif hasattr(dataset, 'ann_file') and dataset.ann_file is None:
                print(f"Note: {dataloader_key}.dataset.ann_file is None (will be set later)")
        else:
            print(f"Warning: {dataloader_key} or its dataset not found in config")

    return config


def update_annotation_files(config: Config, ann_dir: str, fold: int = None) -> Config:
    """
    Update annotation files in the configuration based on the training strategy.
    
    Args:
        config (Config): The MMDetection configuration object.
        ann_dir (str): Directory containing the annotation files.
        fold (int, optional): Fold number for cross-validation. Defaults to None.
    
    Returns:
        Config: The updated configuration object.
    """
    if not ann_dir:
        return config

    # Convert to absolute path if not already
    ann_dir = osp.abspath(ann_dir)

    # Determine annotation file paths based on training strategy
    if fold is not None:
        train_ann_file = osp.join(ann_dir, f'train_fold_{fold}.json')
        val_ann_file = osp.join(ann_dir, f'val_fold_{fold}.json')
        print(f"Using cross-validation annotation files for fold {fold}:")
    else:
        train_ann_file = osp.join(ann_dir, 'train.json')
        val_ann_file = osp.join(ann_dir, 'val.json')
        print("Using regular train-validation annotation files:")

    print(f"  Train: {train_ann_file}")
    print(f"  Val: {val_ann_file}")

    # Update dataset configurations
    if hasattr(config, 'train_dataloader'):
        # Ensure we're using absolute path for annotation file
        config.train_dataloader.dataset.ann_file = train_ann_file
        print(f"Updated train_dataloader.dataset.ann_file to: {train_ann_file}")

    if hasattr(config, 'val_dataloader'):
        # Ensure we're using absolute path for annotation file
        config.val_dataloader.dataset.ann_file = val_ann_file
        print(f"Updated val_dataloader.dataset.ann_file to: {val_ann_file}")

    if hasattr(config, 'val_evaluator'):
        # Ensure we're using absolute path for annotation file
        config.val_evaluator.ann_file = val_ann_file
        print(f"Updated val_evaluator.ann_file to: {val_ann_file}")

    return config

def update_gpu_config(config: Config, gpu_ids: list) -> Config:
    """
    Update GPU configuration in the config.
    
    Args:
        config (Config): The MMDetection configuration object.
        gpu_ids (list): List of GPU IDs to use.
    
    Returns:
        Config: The updated configuration object.
    """
    if gpu_ids is None:
        return config

    print(f"Updating GPU configuration to use GPUs: {gpu_ids}")
    
    config.gpu_ids = gpu_ids
    
    if hasattr(config, 'env'):
        if hasattr(config.env, 'device'):
            config.env.device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
            print(f"Updated env.device to: {config.env.device}")
    
    if len(gpu_ids) > 1:
        if not hasattr(config, 'env'):
            config.env = dict()
        config.env.dist_cfg = dict(backend='nccl')
        print("Enabled distributed training with NCCL backend")
    
    return config

def setup_amp_config(cfg: Config) -> None:
    """Setup Automatic Mixed Precision (AMP) configuration."""
    optim_wrapper = cfg.optim_wrapper.type
    if optim_wrapper == 'AmpOptimWrapper':
        print_log('AMP training is already enabled in your config.',
                  logger='current', level=logging.WARNING)
    else:
        assert optim_wrapper == 'OptimWrapper', (
            '`--amp` is only supported when the optimizer wrapper type is '
            f'`OptimWrapper` but got {optim_wrapper}.')
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

def setup_resume_config(cfg: Config, resume: str) -> None:
    """Setup resume training configuration."""
    if resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif resume is not None:
        cfg.resume = True
        cfg.load_from = resume

def setup_wandb_config(cfg: Config, args, model_name: str, fold: int = None) -> None:
    """Setup Weights & Biases configuration."""
    cfg.visualizer.vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend')
    ]

    model_prefix = model_name.lower() if model_name else osp.splitext(osp.basename(args.config))[0]
    run_name = f"{model_prefix}_{fold if fold is not None else 'full'}"

    model_arch = {
        'backbone': cfg.model.backbone.type if hasattr(cfg.model, 'backbone') else 'unknown',
        'neck': cfg.model.neck.type if hasattr(cfg.model, 'neck') else 'unknown',
        'head': cfg.model.roi_head.type if hasattr(cfg.model, 'roi_head') else 
                cfg.model.bbox_head.type if hasattr(cfg.model, 'bbox_head') else 'unknown'
    }

    wandb_config = {
        "model_name": model_name,
        "model_architecture": model_arch,
        "fold": fold,
        "amp": args.amp,
        "cv_splits": args.cv if args.use_cv else None,
        "use_yolo": args.use_yolo,
        "num_classes": len(VinDRCXRDataset.METAINFO['classes'])
    }

    wandb.login()
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=wandb_config,
        settings=wandb.Settings(allow_val_change=True)
    )

def update_batch_size(config: Config, batch_size: int) -> Config:
    """
    Update the batch size in the configuration for both training and validation dataloaders,
    and adjust the auto_scale_lr configuration accordingly.

    Args:
        config (Config): The MMDetection configuration object.
        batch_size (int): The new batch size to use.

    Returns:
        Config: The updated configuration object.
    """
    if batch_size is None:
        return config

    print(f"Updating batch size to {batch_size}")
    
    for dataloader_key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        if hasattr(config, dataloader_key):
            if hasattr(getattr(config, dataloader_key), 'batch_size'):
                getattr(config, dataloader_key).batch_size = batch_size
                print(f"Updated {dataloader_key}.batch_size to {batch_size}")
            elif hasattr(getattr(config, dataloader_key), 'batch_size_per_gpu'):
                getattr(config, dataloader_key).batch_size_per_gpu = batch_size
                print(f"Updated {dataloader_key}.batch_size_per_gpu to {batch_size}")

    if hasattr(config, 'auto_scale_lr'):
        if isinstance(config.auto_scale_lr, dict):
            if 'base_batch_size' in config.auto_scale_lr:
                config.auto_scale_lr['base_batch_size'] = batch_size
                print(f"Updated auto_scale_lr.base_batch_size to {batch_size}")
        elif hasattr(config.auto_scale_lr, 'base_batch_size'):
            config.auto_scale_lr.base_batch_size = batch_size
            print(f"Updated auto_scale_lr.base_batch_size to {batch_size}")

    return config


def setup_config(base_cfg, args, fold=None, model_name=None):
    """Setup configuration with common settings across all models."""
    cfg = base_cfg.copy()

    # Set data root path for images
    cfg.data_root = osp.abspath(args.data_path)

    # Update annotation files
    cfg = update_annotation_files(cfg, args.ann_dir, fold)

    # Update dataset type and metainfo (only if not already set by update_dataset_type)
    if hasattr(cfg, 'train_dataloader'):
        if not hasattr(cfg.train_dataloader.dataset, 'type') or cfg.train_dataloader.dataset.type == 'VinDRCXRDataset':
            # Only override if it's the default or not set
            cfg.train_dataloader.dataset.type = 'YOLOv5VinDRCXRDataset' if args.use_yolo else 'VinDRCXRDataset'
        # Use appropriate metainfo based on dataset type
        if 'CocoDataset' in cfg.train_dataloader.dataset.type:
            # For CocoDataset, use the metainfo from the config
            if hasattr(cfg, 'metainfo'):
                cfg.train_dataloader.dataset.metainfo = cfg.metainfo
        else:
            # For VinDRCXRDataset, use the standard metainfo
            cfg.train_dataloader.dataset.metainfo = VinDRCXRDataset.METAINFO

    if hasattr(cfg, 'val_dataloader'):
        if not hasattr(cfg.val_dataloader.dataset, 'type') or cfg.val_dataloader.dataset.type == 'VinDRCXRDataset':
            # Only override if it's the default or not set
            cfg.val_dataloader.dataset.type = 'YOLOv5VinDRCXRDataset' if args.use_yolo else 'VinDRCXRDataset'
        # Use appropriate metainfo based on dataset type
        if 'CocoDataset' in cfg.val_dataloader.dataset.type:
            # For CocoDataset, use the metainfo from the config
            if hasattr(cfg, 'metainfo'):
                cfg.val_dataloader.dataset.metainfo = cfg.metainfo
        else:
            # For VinDRCXRDataset, use the standard metainfo
            cfg.val_dataloader.dataset.metainfo = VinDRCXRDataset.METAINFO

    # Setup work directory
    if fold is not None:
        # For cross-validation, create a directory for this specific fold
        work_dir_full = osp.join(args.work_dir, f'fold_{fold}')
    else:
        # For regular training, use the base work directory
        work_dir_full = args.work_dir
    
    cfg.work_dir = work_dir_full
    
    # Create the work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f"Created work directory: {cfg.work_dir}")

    # Apply configuration overrides if provided
    if hasattr(args, 'cfg_options') and args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Setup AMP training
    if args.amp:
        setup_amp_config(cfg)

    # Update GPU configuration
    cfg = update_gpu_config(cfg, args.gpu_ids)

    # Setup resuming
    setup_resume_config(cfg, args.resume)

    # Setup wandb if requested
    if args.use_wandb:
        setup_wandb_config(cfg, args, model_name, fold)

    return cfg
