import json
import argparse
from ensemble_boxes import weighted_boxes_fusion
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import mmengine


def load_image_sizes(coco_file: str) -> Dict[int, tuple]:
    """
    Load image sizes from COCO format file.

    Args: 
        coco_file (str): Path to COCO format JSON file containing image information

    Returns:
        Dict[int, tuple]: Dictionary mapping image IDs to (width, height) tuples
    """
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    return {img['id']: (img['width'], img['height'])
            for img in coco_data['images']}


def normalize_predictions(predictions: List[Dict],
                          image_sizes: Dict[int, tuple],
                          score_thr: float = 0.0001) -> Tuple[Dict[int, List], Dict[int, tuple]]:
    """
    Normalize bounding boxes in predictions and group them by image ID.

    Args:
        predictions (List[Dict]): List of predictions
        image_sizes (Dict[int, tuple]): Image size information
        score_thr (float): Score threshold for filtering

    Returns:
        Tuple containing:
        - Dict mapping image IDs to lists of normalized predictions
        - Dict mapping image IDs to original prediction indices
    """
    normalized_by_image = {}
    indices_by_image = {}

    for idx, pred in enumerate(predictions):
        img_id = pred['image_id']
        score = pred['score']

        if score < score_thr or img_id not in image_sizes:
            continue

        if img_id not in normalized_by_image:
            normalized_by_image[img_id] = {
                'boxes': [],
                'scores': [],
                'labels': []
            }
            indices_by_image[img_id] = []

        # Normalize box coordinates
        x, y, w, h = pred['bbox']
        img_w, img_h = image_sizes[img_id]

        normalized_box = [
            x / img_w,
            y / img_h,
            (x + w) / img_w,
            (y + h) / img_h
        ]

        normalized_by_image[img_id]['boxes'].append(normalized_box)
        normalized_by_image[img_id]['scores'].append(score)
        normalized_by_image[img_id]['labels'].append(pred['category_id'])
        indices_by_image[img_id].append(idx)

    return normalized_by_image, indices_by_image


def denormalize_boxes(boxes: np.ndarray,
                      image_width: int,
                      image_height: int) -> List[List[float]]:
    """
    Convert normalized boxes back to original format.

    Args:
        boxes (np.ndarray): Array of normalized boxes
        image_width (int): Image width
        image_height (int): Image height

    Returns:
        List[List[float]]: Denormalized boxes in [x, y, w, h] format
    """
    denorm_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x = x1 * image_width
        y = y1 * image_height
        w = (x2 - x1) * image_width
        h = (y2 - y1) * image_height
        denorm_boxes.append([x, y, w, h])
    return denorm_boxes


def ensemble_predictions(predictions_list: List[List[Dict]],
                         image_sizes: Dict[int, tuple],
                         iou_thr: float = 0.5,
                         weights: List[float] = None,
                         skip_box_thr: float = 0.0001,
                         score_thr: float = 0.001) -> List[Dict]:
    """
    Ensemble predictions from multiple models using weighted box fusion.

    Args:
        predictions_list (List[List[Dict]]): List of prediction lists from each model
        image_sizes (Dict[int, tuple]): Dictionary mapping image IDs to (width, height)
        iou_thr (float): IoU threshold for box fusion
        weights (List[float], optional): Weights for each model's predictions
        skip_box_thr (float): Threshold for skipping low-confidence boxes
        score_thr (float): Threshold for filtering predictions

    Returns:
        List[Dict]: Final predictions after ensembling
    """
    if not predictions_list:
        print("Error: predictions_list is empty")
        return []

    # If no weights provided, use equal weights
    if weights is None:
        weights = [1.0] * len(predictions_list)
    elif len(weights) != len(predictions_list):
        raise ValueError("Number of weights must match number of prediction files")

    # Normalize predictions from each model
    normalized_predictions = []

    for preds in predictions_list:
        norm_preds, _ = normalize_predictions(preds, image_sizes, score_thr)
        normalized_predictions.append(norm_preds)

    # Process each image
    final_predictions = []
    all_image_ids = set().union(*[set(preds.keys()) for preds in normalized_predictions])

    for img_id in all_image_ids:
        boxes_list = []
        scores_list = []
        labels_list = []

        # Collect normalized predictions for this image from all models
        for norm_preds in normalized_predictions:
            if img_id in norm_preds:
                boxes_list.append(norm_preds[img_id]['boxes'])
                scores_list.append(norm_preds[img_id]['scores'])
                labels_list.append(norm_preds[img_id]['labels'])
            else:
                # Add empty arrays for models with no predictions
                boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                scores_list.append(np.array([], dtype=np.float32))
                labels_list.append(np.array([], dtype=np.float32))
        if not boxes_list:
            continue

        # Perform weighted box fusion
        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )
        # Denormalize boxes
        image_width, image_height = image_sizes[img_id]
        denorm_boxes = denormalize_boxes(boxes_fused, image_width, image_height)

        # Create final predictions
        for box, score, label in zip(denorm_boxes, scores_fused, labels_fused):
            if score >= score_thr:
                final_predictions.append({
                    'image_id': img_id,
                    'bbox': box,
                    'score': float(score),
                    'category_id': int(label)
                })

    return final_predictions


def main():
    parser = argparse.ArgumentParser(description='Ensemble predictions from multiple JSON files')
    parser.add_argument('files', nargs='+', help='Paths to input JSON files with predictions')
    parser.add_argument('--coco-file', required=True, help='Path to COCO format file with image information')
    parser.add_argument('--weights', type=float, nargs='+',
                        help='Optional weights for each input file')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for box fusion (default: 0.5)')
    parser.add_argument('--skip-box-threshold', type=float, default=0.00001,
                        help='Threshold for skipping low-confidence boxes (default: 0.0001)')
    parser.add_argument('--score-threshold', type=float, default=0.0001,
                        help='Threshold for filtering predictions (default: 0.2)')
    parser.add_argument('--output', default='ensemble_results',
                        help='Output file prefix for results (default: ensemble_results)')

    args = parser.parse_args()

    # Validate weights if provided
    if args.weights and len(args.weights) != len(args.files):
        parser.error("Number of weights must match number of input files")

    # Load image sizes
    image_sizes = load_image_sizes(args.coco_file)

    # Read predictions from files
    combined_predictions = []
    for file_path in args.files:
        try:
            with open(file_path, 'r') as f:
                predictions = json.load(f)
                combined_predictions.append(predictions)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Perform ensembling

    final_predictions = ensemble_predictions(
        combined_predictions,
        image_sizes,
        iou_thr=args.iou_threshold,
        weights=args.weights,
        skip_box_thr=args.skip_box_threshold,
        score_thr=args.score_threshold
    )

    # Save results
    pkl_out = f'{args.output}.pkl'
    json_out = f'{args.output}.json'

    print(f'Writing results to {pkl_out}')
    mmengine.dump(final_predictions, pkl_out)

    print(f'Writing results to {json_out}')
    mmengine.dump(final_predictions, json_out)


if __name__ == '__main__':
    main()