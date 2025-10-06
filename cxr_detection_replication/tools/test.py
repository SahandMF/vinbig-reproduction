from ensemble_boxes import weighted_boxes_fusion
import argparse
import os
import os.path as osp

import mmengine
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower

from cxr_detection_replication.retmed.utils.ensemble import ensemble_predictions, load_image_sizes
from cxr_detection_replication.retmed.utils.convert import convert_predictions_format
from cxr_detection_replication.retmed.utils.vindrcxr2kaggle import vindrcxr_to_kaggle_submission_format
from cxr_detection_replication.retmed.datasets.vindrcxr import VinDRCXRDataset, YOLOv5VinDRCXRDataset
from cxr_detection_replication.retmed.utils.tta_patch import patch_tta_model

import pickle
import json
import numpy as np

import mmdet.datasets
import mmdet.evaluation.metrics

def convert_pkl_to_json(pkl_path, json_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    def numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_list(item) for item in obj]
        return obj

    # Apply the conversion
    data = numpy_to_list(data)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test multiple checkpoints with optional ensembling')
    parser.add_argument('config', nargs='?', default=None, help='Test config file path (required for MMDet models)')
    parser.add_argument('checkpoints', nargs='+',
                      help='List of checkpoint files to test')
    parser.add_argument('--model-types', nargs='+', required=True, choices=['mmdet', 'effdet'],
                      help='List of model types for each checkpoint (mmdet or effdet)')
    parser.add_argument('--effdet-configs', nargs='+', default=None,
                      help='List of YAML config files for each effdet checkpoint (use "" for mmdet checkpoints)')
    parser.add_argument('--test-ann-file', required=True,
                      help='Path to the test annotation file')
    parser.add_argument('--data-root', type=str, default=None,
                      help='Root directory containing the dataset images. The default is the directory which was specified in the config file.')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for testing (default: 8)')
    
    # Output configuration
    output_group = parser.add_argument_group('Output configuration')
    output_group.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save all output files')
    output_group.add_argument(
        '--work-dir', type=str, default=None,
        help='Subdirectory for temporary working files, the default is the directory which was specified in the config file.')
    output_group.add_argument(
        '--out-prefix', type=str, default='',
        help='Prefix for output result files, the default is an empty string.')
    output_group.add_argument(
        '--json-prefix',
        type=str,
        required=True,
        help='Prefix for output JSON files')
    output_group.add_argument(
        '--csv',
        action='store_true',
        help='Enable CSV output generation')
    output_group.add_argument(
        '--csv-output',
        type=str,
        default='submission.csv',
        help='Filename for CSV output (default: submission.csv)')
    
    # Ensemble configuration
    ensemble_group = parser.add_argument_group('Ensemble configuration')
    ensemble_group.add_argument(
        '--ensemble',
        action='store_true',
        help='Enable model ensembling')
    ensemble_group.add_argument(
        '--weights',
        type=float,
        nargs='+',
        help='Weights for each model in ensemble')
    ensemble_group.add_argument(
        '--iou-thr',
        type=float,
        default=0.6,
        help='IoU threshold for WBF')
    ensemble_group.add_argument(
        '--skip-box-thr',
        type=float,
        default=0.0001,
        help='Skip box threshold for WBF')
    ensemble_group.add_argument(
        '--score-thr',
        type=float,
        default=0.2,
        help='Score threshold for WBF')
    
    # Runtime options
    runtime_group = parser.add_argument_group('Runtime options')
    runtime_group.add_argument(
        '--tta',
        action='store_true',
        help='Enable test time augmentation')
    
    parser.add_argument('--launcher', default='none', help='job launcher')
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config, e.g., key=val key2="[a,b]" nested.key=val or nested.dict="dict(k=v)"')
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show results during inference')
    parser.add_argument(
        '--show-dir',
        type=str,
        default=None,
        help='Directory where painted images will be saved')
    
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Whether to use deploy mode (default: False)')
    
    parser.add_argument('--effdet-mean', type=float, nargs='+', default=[0.485, 0.456, 0.406],
                      help='Mean for EffDet normalization (default: ImageNet mean)')
    parser.add_argument('--effdet-std', type=float, nargs='+', default=[0.229, 0.224, 0.225],
                      help='Std for EffDet normalization (default: ImageNet std)')
    
    return parser.parse_args()

def run_effdet_inference(checkpoint, config_yaml, test_ann_file, data_root, batch_size, output_dir, out_prefix, json_prefix, idx, args):
    """
    Run inference for an EfficientDet model and save results in MMDet-compatible format.
    """
    import yaml
    import torch
    from effdet import create_model
    from effdet.data import create_dataset, create_loader
    import torchvision
    import os
    import numpy as np
    import json

    # Load config
    with open(config_yaml, 'r') as f:
        effdet_args = yaml.safe_load(f)

    # Set up model
    model = create_model(
        effdet_args['model'],
        bench_task='predict',
        num_classes=effdet_args['num_classes'],
        pretrained=False,
        checkpoint_path=checkpoint
    )
    model.eval()
    model.cuda()

    # Use mean/std from model config if available, else from CLI args
    mean = getattr(model.config, 'mean', None)
    std = getattr(model.config, 'std', None)
    used_default_mean = False
    used_default_std = False
    if mean is None:
        mean = args.effdet_mean
        used_default_mean = True
    if std is None:
        std = args.effdet_std
        used_default_std = True
    if used_default_mean or used_default_std:
        print(f"[EffDet WARNING] Using fallback normalization values: mean={mean}, std={std}. If your model was trained with different normalization, please specify them using --effdet-mean and --effdet-std.")

    # Load test dataset
    dataset = create_dataset(
        effdet_args['dataset'],
        data_root,
        splits=('test',),
        ann_file_override=test_ann_file  # Pass your actual test annotation file here
    )
    if isinstance(dataset, (list, tuple)):
        dataset = dataset[0]
        
    loader = create_loader(
        dataset,
        input_size=model.config.image_size,
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation='bilinear',
        mean=mean,
        std=std,
        num_workers=2,
        pin_mem=False
    )

    # Load COCO annotation file to map dataset index to image_id
    with open(test_ann_file, 'r') as f:
        coco_data = json.load(f)
    # Assume images are in the same order as dataset indices
    idx_to_image_id = [img['id'] for img in coco_data['images']]

    all_results = []
    dataset_idx = 0
    # Use the score threshold from args (which has default 0.001 from argument parser)
    score_thr = args.score_thr
    
    # Debug: Check category mapping
    # print(f"[DEBUG] EffDet model num_classes: {model.config.num_classes}")
    # print(f"[DEBUG] Using score threshold: {score_thr}")
    
    for images, targets in loader:
        images = images.cuda()
        with torch.no_grad():
            detections = model(images)  # [batch, max_det_per_image, 6]
        detections = detections.cpu().numpy()
        batch_size_actual = detections.shape[0]
        for i in range(batch_size_actual):
            image_id = idx_to_image_id[dataset_idx]
            for det in detections[i]:
                x1, y1, x2, y2, score, label = det.tolist()
                if score < score_thr:
                    continue
                bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
                
                # Debug: Print first few detections to check category mapping
                if len(all_results) < 5:
                    print(f"[DEBUG] Detection: bbox={bbox}, score={score:.3f}, label={label}")
                
                # EffDet adds +1 to class indices (class 0 is background), but we need to map to 0-based indices
                # for the VinDR-CXR dataset which has classes 0-13
                adjusted_label = int(label) - 1  # Convert from 1-based to 0-based
                
                # Ensure label is within valid range (0-13 for VinDR-CXR)
                if 0 <= adjusted_label <= 13:
                    all_results.append({
                        'image_id': image_id,
                        'bbox': bbox,
                        'score': float(score),
                        'category_id': adjusted_label
                    })
                else:
                    print(f"[WARNING] Skipping detection with invalid label {label} (adjusted: {adjusted_label})")
            dataset_idx += 1

    # Save results in MMDet-compatible format
    json_out = os.path.join(output_dir, f'{json_prefix}_{idx}.json')
    with open(json_out, 'w') as f:
        json.dump(all_results, f)
    print(f'Saved EffDet results to {json_out}')
    # Return results for ensembling
    return all_results

def main():
    args = parse_args()
    
    # Apply TTA patch to fix data type issues
    patch_tta_model()
    
    # Validate paths and create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
    
    # Convert paths to absolute paths
    test_ann_file = os.path.abspath(args.test_ann_file)
    # data_root = os.path.abspath(args.data_root)
    
    if not os.path.exists(test_ann_file):
        raise FileNotFoundError(f"Test annotation file not found: {test_ann_file}")
   
    # Validate weights if provided
    if args.ensemble and args.weights and len(args.weights) != len(args.checkpoints):
        raise ValueError(f"Number of weights ({len(args.weights)}) must match "
                        f"number of checkpoints ({len(args.checkpoints)})")

    # Validate model-types and effdet-configs
    if len(args.model_types) != len(args.checkpoints):
        raise ValueError('Length of --model-types must match number of checkpoints')
    if args.effdet_configs is not None and len(args.effdet_configs) != len(args.checkpoints):
        raise ValueError('Length of --effdet-configs must match number of checkpoints')

    # If any model is MMDet, config is required
    if 'mmdet' in args.model_types:
        if not args.config:
            raise ValueError('Config file is required for MMDet models')
        base_cfg = Config.fromfile(args.config)
    else:
        base_cfg = None

    # Reduce the number of repeated compilations and improve training speed
    setup_cache_size_limit_of_dynamo()

    predictions = []  # Store predictions from all checkpoints
    image_sizes = load_image_sizes(args.test_ann_file)
    # Process each checkpoint
    for idx, (checkpoint, model_type) in enumerate(zip(args.checkpoints, args.model_types)):
        print(f"\n--- Processing Checkpoint {idx}/{len(args.checkpoints)}: {checkpoint} (type: {model_type}) ---")
        if model_type == 'mmdet':
            cfg = base_cfg.copy()
            cfg.launcher = args.launcher

            # Update dataset config (handle possible nested dataset wrappers)
            test_data_cfg = cfg.test_dataloader.dataset
            # Set on top-level dataset
            test_data_cfg.ann_file = test_ann_file
            if args.data_root is not None:
                test_data_cfg.data_root = args.data_root
                test_data_cfg.data_prefix = dict(img='test/')

            # Descend to innermost dataset if wrapped
            inner = test_data_cfg
            while isinstance(inner, (dict, ConfigDict)) and 'dataset' in inner:
                inner = inner['dataset']

            try:
                inner.ann_file = test_ann_file
                if args.data_root is not None:
                    inner.data_root = args.data_root
                    inner.data_prefix = dict(img='test/')
            except Exception:
                # Fallback for plain dict inner datasets
                inner['ann_file'] = test_ann_file
                if args.data_root is not None:
                    inner['data_root'] = args.data_root
                    inner['data_prefix'] = dict(img='test/')

            cfg.test_dataloader.batch_size = args.batch_size
            cfg.test_dataloader.num_workers = 4

            # Update test evaluator config
            if hasattr(cfg, 'test_evaluator'):
                cfg.test_evaluator.ann_file = test_ann_file

            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)

            # Set up work directory
            if args.work_dir is not None:
                cfg.work_dir = args.work_dir
            elif cfg.get('work_dir', None) is None:
                # Use a consistent work directory for all test files
                cfg.work_dir = osp.join('work_dirs', 'test_results')
                os.makedirs(cfg.work_dir, exist_ok=True)

            # Add checkpoint-specific output prefix if multiple checkpoints
            if len(args.checkpoints) > 1 and args.out_prefix:
                cfg.work_dir = osp.join(cfg.work_dir, f'{args.out_prefix}_{idx}')

            # Load the specific checkpoint
            cfg.load_from = checkpoint

            if args.show or args.show_dir:
                cfg = trigger_visualization_hook(cfg, args)

            if args.deploy:
                cfg.custom_hooks.append(dict(type='SwitchToDeployHook'))

            # Handle JSON output prefix for multiple checkpoints
            if args.json_prefix:
                json_prefix = f'{args.json_prefix}_{idx}' if len(args.checkpoints) > 1 else args.json_prefix
                # Include the output directory in the prefix path to ensure all files are saved there
                full_prefix = osp.join(args.output_dir, json_prefix)
                cfg_json = {
                    'test_evaluator.format_only': True,
                    'test_evaluator.outfile_prefix': full_prefix
                }
                cfg.merge_from_dict(cfg_json)

            # Determine whether the custom metainfo fields are all lowercase
            is_metainfo_lower(cfg)

            # Test Time Augmentation (TTA) handling
            if args.tta:
                assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config. Cannot use TTA!'
                assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config. Cannot use TTA!'

                cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
                test_data_cfg = cfg.test_dataloader.dataset
                while 'dataset' in test_data_cfg:
                    test_data_cfg = test_data_cfg['dataset']

                # Remove batch_shapes_cfg for TTA
                if 'batch_shapes_cfg' in test_data_cfg:
                    test_data_cfg.batch_shapes_cfg = None
                test_data_cfg.pipeline = cfg.tta_pipeline

            # Remove training-only hooks that break test-only runs (e.g., EarlyStoppingHook)
            if hasattr(cfg, 'custom_hooks') and cfg.custom_hooks:
                try:
                    cfg.custom_hooks = [
                        h for h in cfg.custom_hooks
                        if not (isinstance(h, (dict, ConfigDict)) and h.get('type') == 'EarlyStoppingHook')
                    ]
                except Exception:
                    pass

            # Add debug prints before building the runner
            print("Test dataset type:", cfg.test_dataloader.dataset.get('type', None))
            print("Test dataset ann_file:", cfg.test_dataloader.dataset.ann_file)
            # If possible, print dataset length after runner is built
            # Build the runner
            runner = Runner.from_cfg(cfg) if 'runner_type' not in cfg else RUNNERS.build(cfg)
            # Try to print dataset length if accessible
            try:
                test_dataset = runner.test_dataloader.dataset
                print("Test dataset length:", len(test_dataset))
            except Exception as e:
                print("Could not print test dataset length:", e)

            # Test and collect results
            results = runner.test()
            
            # Save results to output directory
            json_out = osp.join(args.output_dir, f'{args.json_prefix}_{idx}.json')
            mmengine.dump(results, json_out)
            print(f'Saved results to {json_out}')
            
            # Check for bbox.json file created by the runner
            bbox_json_path = osp.join(args.output_dir, f'{json_prefix}.bbox.json')
            if osp.exists(bbox_json_path) and osp.getsize(bbox_json_path) > 0:
                # Read the bbox.json file created by the runner
                with open(bbox_json_path, 'r') as f:
                    bbox_results = json.load(f)
                
                # Only add non-empty results to predictions list
                if bbox_results:
                    print(f'Using bbox results from {bbox_json_path} for ensembling')
                    predictions.append(bbox_results)
            else:
                print(f'Warning: No valid bbox results found at {bbox_json_path}')
        elif model_type == 'effdet':
            if args.effdet_configs is None:
                raise ValueError('Must provide --effdet-configs for effdet checkpoints')
            config_yaml = args.effdet_configs[idx]
            effdet_results = run_effdet_inference(
                checkpoint, config_yaml, test_ann_file, args.data_root, args.batch_size, args.output_dir, args.out_prefix, args.json_prefix, idx, args
            )
            predictions.append(effdet_results)
            
            # Generate CSV for single EffDet model if requested
            if args.csv:
                csv_path = osp.join(args.output_dir, f'{args.csv_output}_{idx}' if len(args.checkpoints) > 1 else args.csv_output)
                converted_json_path = osp.join(args.output_dir, f'converted_predictions_{idx}.json')
                
                # Convert predictions format using the actual test annotation file
                converted = convert_predictions_format(
                    effdet_results,
                    test_ann_file,  # Use actual test annotation file instead of hardcoded path
                    score_threshold=args.score_thr
                )
                
                # Save converted predictions to file
                with open(converted_json_path, 'w') as f:
                    json.dump(converted, f)
                
                # Generate Kaggle CSV from file
                vindrcxr_to_kaggle_submission_format(
                    converted_json_path,
                    csv_path,
                    args.data_root + '/test' if args.data_root else None
                )
                print(f"Kaggle CSV saved to {csv_path}")
        else:
            raise ValueError(f'Unknown model type: {model_type}')

    if args.ensemble:
        # Perform ensembling with WBF
        print("\n--- Performing Weighted Boxes Fusion (WBF) ---")
        ensembled_results = ensemble_predictions(
            predictions,
            image_sizes,
            iou_thr=args.iou_thr,
            weights=args.weights,
            skip_box_thr=args.skip_box_thr,
            score_thr=args.score_thr
        )

        # Save the ensembled results
        ensembled_json_path = osp.join(args.output_dir, f'{args.json_prefix}_ensemble.json')
        mmengine.dump(ensembled_results, ensembled_json_path)
        print(f"Ensembled results saved to {ensembled_json_path}")

        # Generate CSV if requested
        if args.csv:
            csv_path = osp.join(args.output_dir, args.csv_output)
            converted_json_path = osp.join(args.output_dir, 'converted_predictions.json')
            
            # Convert predictions format using the actual test annotation file
            converted = convert_predictions_format(
                ensembled_results,
                test_ann_file,  # Use actual test annotation file instead of hardcoded path
                score_threshold=args.score_thr
            )
            
            # Save converted predictions to file
            with open(converted_json_path, 'w') as f:
                json.dump(converted, f)
            
            # Generate Kaggle CSV from file
            vindrcxr_to_kaggle_submission_format(
                converted_json_path,
                csv_path,
                args.data_root + '/test' if args.data_root else None
            )
            print(f"Kaggle CSV saved to {csv_path}")

if __name__ == '__main__':
    main()
