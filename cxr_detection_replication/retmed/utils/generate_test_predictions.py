import os
import json
from PIL import Image
import random
import argparse


def create_mock_coco_annotations(image_folder, output_json_path):
    # Initialize the COCO format dictionary
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"name": "Aortic enlargement", "id": 0, "supercategory": None},
            {"name": "Atelectasis", "id": 1, "supercategory": None},
            {"name": "Calcification", "id": 2, "supercategory": None},
            {"name": "Cardiomegaly", "id": 3, "supercategory": None},
            {"name": "Consolidation", "id": 4, "supercategory": None},
            {"name": "ILD", "id": 5, "supercategory": None},
            {"name": "Infiltration", "id": 6, "supercategory": None},
            {"name": "Lung Opacity", "id": 7, "supercategory": None},
            {"name": "Nodule/Mass", "id": 8, "supercategory": None},
            {"name": "Other lesion", "id": 9, "supercategory": None},
            {"name": "Pleural effusion", "id": 10, "supercategory": None},
            {"name": "Pleural thickening", "id": 11, "supercategory": None},
            {"name": "Pneumothorax", "id": 12, "supercategory": None},
            {"name": "Pulmonary fibrosis", "id": 13, "supercategory": None}
        ]
    }

    # List all image files
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

    # Process each image
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # # Create three entries for each image (as in your example)
                # rad_ids = ["R8", "R9", "R10"]
                # for rad_id in rad_ids:
                image_entry = {
                    "id": idx, # * 3  + rad_ids.index(rad_id),  # Unique ID for each entry
                    "width": width,
                    "height": height,
                    # "rad_id": rad_id,
                    "file_name": image_file
                }
                coco_format["images"].append(image_entry)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    # Save the JSON file
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"Created mock COCO annotations file at: {output_json_path}")
    print(f"Processed {len(image_files)} images")
    print(f"Created {len(coco_format['images'])} image entries")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate mock COCO annotations for test images')
    parser.add_argument('--input_folder', required=True, help='Path to the folder containing test images')
    parser.add_argument('--output_json', default='data/test.json', help='Path to save the output JSON file (default: data/test.json)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create annotations
    create_mock_coco_annotations(args.input_folder, args.output_json)