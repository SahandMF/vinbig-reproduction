from pycocotools.coco import COCO
import json
import os
import argparse


from ensemble_boxes import weighted_boxes_fusion
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from typing import Dict, List, Tuple
import random

def convert_to_wbf_format(boxes):
    """Convert [x, y, w, h] boxes to [x1, y1, x2, y2] format for WBF."""
    return [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]

def convert_from_wbf_format(boxes):
    """Convert [x1, y1, x2, y2] boxes back to [x, y, w, h] format."""
    return [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]

def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
        """Convert bbox to normalized coordinates (0-1 range)."""
        x1, y1, w, h = bbox
        return [
            x1 / width,
            y1 / height,
            (x1 + w) / width,
            (y1 + h) / height
        ]

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def process_annotations(data, iou_thresh=0.4):
    """
    Process annotations to create datasets for 4 different training scenarios.
    Uses ensemble-boxes WBF implementation.

    Args:
        data (dict): Dictionary containing 'images', 'annotations', and 'categories'
        iou_thresh (float): IoU threshold for WBF, default 0.4

    Returns:
        dict: Dictionary containing four processed datasets for different training scenarios:
            - model-1: labels-1 on all images
            - model-2: labels-2 on all images
            - model-3: labels-3 on all images
            - model-4: labels-1 only on images with disease
    """
    # Group images by file_name
    images_by_file = defaultdict(list)
    unique_images = set()  # Track unique file names
    for img in data['images']:
        images_by_file[img['file_name']].append(img)
        unique_images.add(img['file_name'])

    # Group annotations by file_name
    annotations_by_file = defaultdict(list)
    for ann in data['annotations']:
        img_info = next(img for img in data['images'] if img['id'] == ann['image_id'])
        annotations_by_file[img_info['file_name']].append({
            **ann,
            'rad_id': next(img['rad_id'] for img in data['images'] if img['id'] == ann['image_id'])
        })

    processed_datasets = {
        'label-1': {'images': [], 'annotations': [], 'categories': data['categories']},
        'label-2': {'images': [], 'annotations': [], 'categories': data['categories']},
        'label-3': {'images': [], 'annotations': [], 'categories': data['categories']},
        'label-4': {'images': [], 'annotations': [], 'categories': data['categories']}
    }

    # Track which images have disease annotations
    images_with_disease = set()
    next_ann_id = 1

    # First pass: Process all annotations and track images with disease
    for file_name, annotations in annotations_by_file.items():
        # unique_image = images_by_file[file_name]

        if not annotations:
            continue
        img_width = images_by_file[file_name][0]['width']
        img_height = images_by_file[file_name][0]['height']
        # Prepare data for WBF
        boxes = [ann['bbox'] for ann in annotations]
        scores = [1.0] * len(boxes)
        labels = [ann['category_id'] for ann in annotations]
        rad_ids = [ann['rad_id'] for ann in annotations]

        # Convert boxes to WBF format and normalize
        boxes_wbf = convert_to_wbf_format(boxes)
        boxes_norm = [[x/img_width if i % 2 == 0 else x/img_height
                      for i, x in enumerate(box)] for box in boxes_wbf]

        # Apply WBF
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            [boxes_norm],
            [scores],
            [labels],
            iou_thr=iou_thresh,
            skip_box_thr=0.0,
            weights=None
        )

        # Denormalize boxes and convert back to [x, y, w, h] format
        merged_boxes = [[x*img_width if i % 2 == 0 else x*img_height
                        for i, x in enumerate(box)] for box in merged_boxes]
        merged_boxes = convert_from_wbf_format(merged_boxes)

        # Count annotator consensus for each merged box
        consensus_counts = []
        merged_rad_ids = []

        for merged_box in merged_boxes:
            contributing_rads = set()
            merged_box_wbf = convert_to_wbf_format([merged_box])[0]

            for orig_box, rad_id in zip(boxes, rad_ids):
                orig_box_wbf = convert_to_wbf_format([orig_box])[0]
                iou = calculate_iou(merged_box_wbf, orig_box_wbf)
                if iou > iou_thresh:
                    contributing_rads.add(rad_id)

            consensus_counts.append(len(contributing_rads))
            merged_rad_ids.append(list(contributing_rads))


        # Create annotations based on consensus
        has_disease = False
        for idx, (box, label, count, contributing_rads) in enumerate(
            zip(merged_boxes, merged_labels, consensus_counts, merged_rad_ids)):

            area = box[2] * box[3]
            base_ann = {
                'id': next_ann_id,
                'image_id': images_by_file[file_name][0]['id'],
                'category_id': int(label),
                'bbox': box,
                'area': area,
                'iscrowd': 0,
                'rad_ids': contributing_rads,
                'segmentation': [[
                    box[0], box[1],
                    box[0] + box[2], box[1],
                    box[0] + box[2], box[1] + box[3],
                    box[0], box[1] + box[3]
                ]]
            }

            # Model-1 and Model-4: One or more annotators
            if count >= 1:
                processed_datasets['label-1']['annotations'].append(base_ann.copy())

                processed_datasets['label-4']['annotations'].append(base_ann.copy())
                has_disease = True

            # Model-2: Two or more annotators
            if count >= 2:
                processed_datasets['label-2']['annotations'].append(base_ann.copy())

            # Model-3: All three annotators
            if count >= 3:
                processed_datasets['label-3']['annotations'].append(base_ann.copy())

            next_ann_id += 1

        if has_disease:
            images_with_disease.add(file_name)


    # Second pass: Add images according to training scenarios
    for file_name in unique_images:
        # Models 1-3: Include all images
        processed_datasets['label-1']['images'].append(images_by_file[file_name][0])
        processed_datasets['label-2']['images'].append(images_by_file[file_name][0])
        processed_datasets['label-3']['images'].append(images_by_file[file_name][0])

        # Model-4: Include only images with disease
        if file_name in images_with_disease:
            processed_datasets['label-4']['images'].append(images_by_file[file_name][0])

    return processed_datasets

def annotators_agreement(data_root, output_dir, iou_thresh=0.4):
    """
    Process annotations for annotator agreement analysis and create cross-validation folds.

    Args:
        data_root (str): Path to the input COCO format JSON annotation file
        output_dir (str): Base directory for output files
        iou_thresh (float): IoU threshold for WBF

    Returns:
        dict: Processed datasets for different training scenarios
    """
    coco = COCO(data_root)

    images = coco.dataset['images']
    annotations = coco.dataset['annotations']
    categories = coco.dataset['categories']

    output_dir = os.path.join(output_dir, "ann_agreement")
    processed_data = process_annotations({'images': images, 'annotations': annotations, 'categories': categories},
                                         iou_thresh=iou_thresh)

    label_sets = []
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing annotator agreement data...")
    
    for each in processed_data.keys():
        label_dir = os.path.join(output_dir, each)
        os.makedirs(label_dir, exist_ok=True)
        output_file = os.path.join(label_dir, f'{each}.json')
        label_sets.append(each)
        with open(output_file, 'w') as f:
            json.dump(processed_data[each], f)
        print(f"Saved {each} dataset to: {output_file}")

    return processed_data

def train_val_split(data, output_dir, ratio=0.8):
    """
    Split COCO format JSON annotations into training and validation sets based on unique file names.

    Parameters:
    -----------
    data : dict
        COCO format data dictionary
    ratio : float, optional
        Proportion of data to use for training (default is 0.8)
    output_dir : str, optional
        Directory to save the train and validation JSON files

    Returns:
    --------
    None
        Writes train and validation JSON files to the output directory
    """
    coco_data = data

    # Create copies to modify
    train_coco = coco_data.copy()
    val_coco = coco_data.copy()

    # Get unique file names
    file_names = list(set(img['file_name'] for img in coco_data['images']))
    random.shuffle(file_names)

    # Calculate split index
    split_idx = int(len(file_names) * ratio)

    # Split file names
    train_files = set(file_names[:split_idx])
    val_files = set(file_names[split_idx:])

    # Filter images based on file names
    train_images = [img for img in coco_data['images'] if img['file_name'] in train_files]
    val_images = [img for img in coco_data['images'] if img['file_name'] in val_files]

    # Get image IDs for train and validation sets
    train_image_ids = set(img['id'] for img in train_images)
    val_image_ids = set(img['id'] for img in val_images)

    # Filter annotations based on image IDs
    train_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in train_image_ids
    ]
    val_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in val_image_ids
    ]

    # Update train and validation COCO dictionaries
    train_coco['images'] = train_images
    train_coco['annotations'] = train_annotations

    val_coco['images'] = val_images
    val_coco['annotations'] = val_annotations

    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, "train_val_split")
    os.makedirs(output_dir, exist_ok=True)

    # Write train annotations
    train_path = os.path.join(output_dir, 'train.json')
    with open(train_path, 'w') as f:
        json.dump(train_coco, f, indent=2)

    # Write validation annotations
    val_path = os.path.join(output_dir, 'val.json')
    with open(val_path, 'w') as f:
        json.dump(val_coco, f, indent=2)

    # Print some information about the split
    print(f"Train/val split summary:")
    print(f"  Total unique files: {len(file_names)}")
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Training files: {len(train_files)} ({ratio * 100}%)")
    print(f"  Validation files: {len(val_files)} ({(1 - ratio) * 100}%)")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Training annotations: {len(train_annotations)}")
    print(f"  Validation annotations: {len(val_annotations)}")
    print(f"Train/val split saved to: {output_dir}")

    if 'info' not in train_coco:
        train_coco['info'] = {}
    if 'licenses' not in train_coco:
        train_coco['licenses'] = []

    if 'info' not in val_coco:
        val_coco['info'] = {}
    if 'licenses' not in val_coco:
        val_coco['licenses'] = []

def preprocess_with_wbf(data_root: str, output_dir, iou_thresh: float = 0.4):
    """
    Preprocess the data with Weighted Box Fusion (WBF)
    """
    coco = COCO(data_root)

    images = coco.dataset['images']
    annotations = coco.dataset['annotations']
    categories = coco.dataset['categories']

    data = {'images': images,  'annotations': annotations, 'categories': categories}
    # output_dir = os.path.join(output_dir, "wbf_data")
    os.makedirs(output_dir, exist_ok=True)

    processed_data = apply_wbf(data, iou_thresh)
    
    # Save the processed data
    output_file = os.path.join(output_dir, 'wbf_processed.json')
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    print(f"Saved WBF processed data to: {output_file}")
    
    return processed_data

def apply_wbf(
        data: Dict,
        iou_thresh: float = 0.4,
        ) -> Dict:
    # Group images by file_name
    images_by_file = defaultdict(list)
    unique_images = set()  # Track unique file names
    for img in data['images']:
        images_by_file[img['file_name']].append(img)
        unique_images.add(img['file_name'])

    # Group annotations by file_name
    annotations_by_file = defaultdict(list)
    for ann in data['annotations']:
        img_info = next(img for img in data['images'] if img['id'] == ann['image_id'])
        annotations_by_file[img_info['file_name']].append({
            **ann,
            'rad_id': next(img['rad_id'] for img in data['images'] if img['id'] == ann['image_id'])
        })

    processed_dataset = {'images': [], 'annotations': [], 'categories': data['categories']}
    next_ann_id = 1  # Initialize the annotation ID counter

    # First pass: Process all annotations and track images with disease
    for file_name, annotations in annotations_by_file.items():
        if not annotations:
            continue
        img_width = images_by_file[file_name][0]['width']
        img_height = images_by_file[file_name][0]['height']
        # Prepare data for WBF
        boxes = [ann['bbox'] for ann in annotations]
        scores = [1.0] * len(boxes)
        labels = [ann['category_id'] for ann in annotations]
        rad_ids = [ann['rad_id'] for ann in annotations]

        # Convert boxes to WBF format and normalize
        boxes_wbf = convert_to_wbf_format(boxes)
        boxes_norm = [[x/img_width if i % 2 == 0 else x/img_height
                      for i, x in enumerate(box)] for box in boxes_wbf]

        # Apply WBF
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            [boxes_norm],
            [scores],
            [labels],
            iou_thr=iou_thresh,
            skip_box_thr=0.0,
            weights=None
        )

        # Denormalize boxes and convert back to [x, y, w, h] format
        merged_boxes = [[x*img_width if i % 2 == 0 else x*img_height
                        for i, x in enumerate(box)] for box in merged_boxes]
        merged_boxes = convert_from_wbf_format(merged_boxes)

        merged_rad_ids = []

        for merged_box in merged_boxes:
            contributing_rads = set()
            merged_box_wbf = convert_to_wbf_format([merged_box])[0]

            for orig_box, rad_id in zip(boxes, rad_ids):
                orig_box_wbf = convert_to_wbf_format([orig_box])[0]
                iou = calculate_iou(merged_box_wbf, orig_box_wbf)
                if iou > iou_thresh:
                    contributing_rads.add(rad_id)

            merged_rad_ids.append(list(contributing_rads))

        for idx, (box, label, contributing_rads) in enumerate(
            zip(merged_boxes, merged_labels, merged_rad_ids)):

            area = box[2] * box[3]
            base_ann = {
                'id': next_ann_id,
                'image_id': images_by_file[file_name][0]['id'],
                'category_id': int(label),
                'bbox': box,
                'area': area,
                'iscrowd': 0,
                'rad_ids': contributing_rads, 
                'segmentation': [[
                    box[0], box[1],
                    box[0] + box[2], box[1],
                    box[0] + box[2], box[1] + box[3],
                    box[0], box[1] + box[3]
                ]]
            }
            processed_dataset['annotations'].append(base_ann.copy())
            next_ann_id += 1

    # Second pass: Add images according to training scenarios
    for file_name in unique_images:
        processed_dataset['images'].append(images_by_file[file_name][0])

    return processed_dataset

def get_stratified_splits(data: Dict, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified splits for cross-validation based on image categories and file names.

    Args:
        data (Dict): Dictionary containing 'images' and 'annotations'
        n_splits (int): Number of cross-validation folds to create

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (train_indices, val_indices) for each fold
    """
    # Group images by file_name
    file_names = []
    labels = []
    seen_files = set()  # Track unique file names
    
    for img in data['images']:
        if img['file_name'] not in seen_files:
            seen_files.add(img['file_name'])
            file_names.append(img['file_name'])
            # Get the most common category for this image
            img_anns = [ann for ann in data['annotations'] if ann['image_id'] == img['id']]
            if img_anns:
                category_counts = {}
                for ann in img_anns:
                    category_counts[ann['category_id']] = category_counts.get(ann['category_id'], 0) + 1
                labels.append(max(category_counts.items(), key=lambda x: x[1])[0])
            else:
                labels.append(0)  # No annotations case

    # Convert to numpy arrays
    file_names = np.array(file_names)
    labels = np.array(labels)

    # Create stratified splits
    cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(cv_splitter.split(file_names, labels, groups=file_names))

def get_random_splits(data: Dict, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create random splits for cross-validation based on file names.

    Args:
        data (Dict): Dictionary containing 'images' and 'annotations'
        n_splits (int): Number of cross-validation folds to create

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (train_indices, val_indices) for each fold
    """
    # Group images by file_name, ensuring uniqueness
    file_names = list(set(img['file_name'] for img in data['images']))
    file_names = np.array(file_names)

    # Create random splits
    cv_splitter = GroupKFold(n_splits=n_splits)
    return list(cv_splitter.split(file_names, groups=file_names))

def create_cv_folds(data: Dict, n_splits: int, output_dir: str, stratification: bool = False) -> None:
    """
    Create cross-validation folds from the processed dataset.

    Args:
        data (Dict): Dictionary containing 'images', 'annotations', and 'categories'
        n_splits (int): Number of cross-validation folds to create
        output_dir (str): Directory to save the cross-validation folds
        stratification (bool): Whether to use stratified sampling based on categories
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreating {n_splits} cross-validation folds...")
    print(f"Using {'stratified' if stratification else 'random'} sampling")

    # Get splits using appropriate method
    splits = get_stratified_splits(data, n_splits) if stratification else get_random_splits(data, n_splits)
    file_names = np.array(list(set(img['file_name'] for img in data['images'])))

    # Create lookup dictionary for faster access
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Create and save each fold
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Get file names for this fold
        train_files = set(file_names[train_idx])
        val_files = set(file_names[val_idx])

        # Create fold-specific splits
        fold_train = {
            'images': [img for img in data['images'] if img['file_name'] in train_files],
            'annotations': [],
            'categories': data['categories']
        }

        fold_val = {
            'images': [img for img in data['images'] if img['file_name'] in val_files],
            'annotations': [],
            'categories': data['categories']
        }

        # Add annotations using the lookup dictionaries
        for img in fold_train['images']:
            fold_train['annotations'].extend(annotations_by_image[img['id']])
        
        for img in fold_val['images']:
            fold_val['annotations'].extend(annotations_by_image[img['id']])

        # Save fold-specific splits
        train_path = os.path.join(output_dir, f'train_fold_{fold_idx}.json')
        val_path = os.path.join(output_dir, f'val_fold_{fold_idx}.json')

        with open(train_path, 'w') as f:
            json.dump(fold_train, f, indent=2)

        with open(val_path, 'w') as f:
            json.dump(fold_val, f, indent=2)

        # Count unique filenames
        train_files = set(img['file_name'] for img in fold_train['images'])
        val_files = set(img['file_name'] for img in fold_val['images'])

        print(f"  Fold {fold_idx}:")
        print(f"    Training: {len(train_files)} unique images, {len(fold_train['annotations'])} annotations")
        print(f"    Validation: {len(val_files)} unique images, {len(fold_val['annotations'])} annotations")

        if 'info' not in fold_train:
            fold_train['info'] = {}
        if 'licenses' not in fold_train:
            fold_train['licenses'] = []
        if 'info' not in fold_val:
            fold_val['info'] = {}
        if 'licenses' not in fold_val:
            fold_val['licenses'] = []

    print(f"Cross-validation folds saved to: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the initial data')
    parser.add_argument('data_root',
                        type=str,
                        help='Path to the initial data (COCO formated JSON file)')
    parser.add_argument('sampling_method', type=str, 
                        choices=['train_val', 'cross_val'], help='The way the data will be used in training: '
                             'train_val (train-validation split), '
                             'cross_val (cross-validation folds)')
    parser.add_argument('--strategy',
                        type=str,
                        choices=['wbf', 'ann_agree', 'none'],
                        default='none',
                        help='The way the data will be used in training: '
                             'wbf (weighted boxes fusion), '
                             'ann_agree (annotators agreement)')
    parser.add_argument('--ratio',
                        type=float,
                        default=0.8,
                        help='Train-validation split ratio (for train_val sampling method)'
                             'Value should be between 0 and 1. Default is 0.8.')
    parser.add_argument('--cv',
                        type=int,
                        help='Number of cross-validation folds in case of cross-validation sampling method')
    parser.add_argument('--stratification',
                        action='store_true',
                        help='Enable stratification for the data sampling, which will be used in training')
    parser.add_argument('--iou-threshold',
                        type=float,
                        default=0.4,
                        help='IoU threshold for box fusion in WBF and annotator agreement strategies. '
                             'Value should be between 0 and 1. Default is 0.4.')
    parser.add_argument('--output-dir',
                        type=str,
                        default='data/',
                        help='Directory to save processed data; saved by default to data/. A directory with the name of the strategy will be created inside this directory.')

    return parser.parse_args()


def main():
    args = parse_args()
    
    data_root = args.data_root

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # First apply the chosen strategy to process the data
    if args.strategy == 'ann_agree':
        processed_data = annotators_agreement(data_root, args.output_dir, iou_thresh=args.iou_threshold)
        
        # For annotator agreement, we need to handle each label set separately
        if args.sampling_method == 'cross_val':
            if args.cv is None:
                raise ValueError("Number of cross-validation folds (--cv) must be specified for cross-validation sampling")
            
            for label_set in ['label-1', 'label-2', 'label-3', 'label-4']:
                label_dir = os.path.join(args.output_dir, 'ann_agreement', label_set)
                cv_dir = os.path.join(label_dir, 'cv_folds')
                create_cv_folds(processed_data[label_set], args.cv, cv_dir, args.stratification)
        else:  # train_val
            for label_set in ['label-1', 'label-2', 'label-3', 'label-4']:
                label_dir = os.path.join(args.output_dir, 'ann_agreement', label_set)
                train_val_split(processed_data[label_set], output_dir=label_dir, ratio=args.ratio)
                
    elif args.strategy == 'wbf':
        wbf_dir = os.path.join(args.output_dir, "wbf_data")
        processed_data = preprocess_with_wbf(data_root, output_dir=wbf_dir, iou_thresh=args.iou_threshold)
        if args.sampling_method == 'cross_val':
            if args.cv is None:
                raise ValueError("Number of cross-validation folds (--cv) must be specified for cross-validation sampling")
            cv_dir = os.path.join(wbf_dir, 'cv_folds')
            create_cv_folds(processed_data, args.cv, cv_dir, args.stratification)
        else:  # train_val
            train_val_split(processed_data, output_dir=wbf_dir, ratio=args.ratio)
            
    elif args.strategy == 'none':
        # Just load the data without any processing
        with open(data_root, 'r') as f:
            processed_data = json.load(f)
            
        if args.sampling_method == 'cross_val':
            if args.cv is None:
                raise ValueError("Number of cross-validation folds (--cv) must be specified for cross-validation sampling")
            cv_dir = os.path.join(args.output_dir, 'cv_folds')
            create_cv_folds(processed_data, args.cv, cv_dir, args.stratification)
        else:  # train_val
            train_val_split(processed_data, output_dir=args.output_dir, ratio=args.ratio)
    else:
        raise ValueError("Invalid strategy. Choose 'wbf', 'ann_agree', or 'none'.")

if __name__ == '__main__':
    main()
