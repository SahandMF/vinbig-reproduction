import os
import json
from typing import List, Dict
import argparse
from cxr_detection_replication.retmed.utils.vindrcxr2kaggle import vindrcxr_to_kaggle_submission_format

def convert_predictions_format(predictions: List[Dict],
                               coco_file: str,
                               score_threshold: float = 0.01) -> Dict[str, List[Dict]]:
    """
    Convert predictions from list format to image-name based dictionary format.

    Args:
        predictions (List[Dict]): List of predictions in format
            [{"image_id": id, "bbox": [...], "score": s, "category_id": c}, ...]
        coco_file (str): Path to COCO format JSON file containing image information
        score_threshold (float): Threshold for filtering predictions

    Returns:
        Dict[str, List[Dict]]: Predictions grouped by image name in format
            {"image_name": [{"bbox": [], "score": .., "category_id": ..}, ...], ...}
    """
    # Load image information from COCO file
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create mapping from image ID to file name
    id_to_name = {
        img['id']: os.path.splitext(os.path.basename(img['file_name']))[0]
        for img in coco_data['images']
    }

    # Convert predictions
    result = {}
    for pred in predictions:
        if pred['score'] < score_threshold:
            continue

        img_id = pred['image_id']
        if img_id not in id_to_name:
            print(f"Warning: No image name found for ID {img_id}")
            continue

        img_name = id_to_name[img_id]
        if img_name not in result:
            result[img_name] = []

        result[img_name].append({
            'bbox': pred['bbox'],
            'score': pred['score'],
            'category_id': pred['category_id']
        })

    return result


def main():
    parser = argparse.ArgumentParser(description='Convert detection predictions format')
    parser.add_argument('input_file', help='Path to input JSON file with predictions')
    parser.add_argument('--coco-file', required=True,
                        help='Path to COCO format file with image information')
    parser.add_argument('--score-threshold', type=float, default=0.01,
                        help='Threshold for filtering predictions (default: 0.2)')
    parser.add_argument('--output', default='converted_predictions.json',
                        help='Output file path (default: converted_predictions.json)')

    args = parser.parse_args()

    # Read predictions
    with open(args.input_file, 'r') as f:
        predictions = json.load(f)

    # Convert format
    converted_predictions = convert_predictions_format(
        predictions,
        args.coco_file,
        score_threshold=args.score_threshold
    )
    # output = os.path.splitext(os.path.basename(args.output))[0] + '.csv'
    # vindrcxr_to_kaggle_submission_format(
    #     converted_predictions,
    #     output,
    # )
    # Save results
    with open(args.output, 'w') as f:
        json.dump(converted_predictions, f, indent=2)

    print(f'Converted predictions saved to {args.output}')


if __name__ == '__main__':
    main()