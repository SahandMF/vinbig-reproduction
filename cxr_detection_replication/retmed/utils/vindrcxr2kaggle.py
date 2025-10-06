import argparse
import csv
import os
import json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Converts results file to kaggle compatible csv")
    parser.add_argument("input_file", type=str, help="input file containing the models output")
    parser.add_argument("output_file", type=str, help="output csv containing the results")
    parser.add_argument("--target_csv", type=str, help="CSV file with target values for filtering")
    parser.add_argument("--apply_target_filter", action="store_true", help="Apply target value filtering")
    parser.add_argument("--target_threshold", type=float, default=0.5, help="Threshold for target value filtering")
    return parser.parse_args()


def vindrcxr_to_kaggle_submission_format(input, output, target_csv=None, apply_target_filter=False,
                                         target_threshold=0.5):
    # Load target DataFrame if filtering is enabled
    target_df = None
    if apply_target_filter and target_csv:
        target_df = pd.read_csv(target_csv)

    with open(input, "r") as f:
        results = json.load(f)

    data_path = '/media/klee_ro/public_datasets/vindr_cxr/test/'

    # Create a set of existing image names in results
    existing_images = set(results.keys())

    # Initialize list for intermediate results
    intermediate = []

    for img_file in os.listdir(data_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name = os.path.splitext(os.path.basename(img_file))[0]

            # Apply target filtering if enabled
            if apply_target_filter and target_df is not None:
                # Find the row for the given image_id
                row = target_df[target_df['image_id'] == img_name]

                # Skip if no matching row or target value exceeds threshold
                if row.empty or row['target'].values[0] > target_threshold:
                    intermediate.append([img_name, "14 1 0 0 1 1"])
                    continue

            # Get results for the image
            if img_name in results:
                entries = results[img_name]
            else:
                entries = []

            img_name = img_name.split(".")[0]

            # Process entries
            if len(entries) != 0:
                prep_entry = ""
                for entry in entries:
                    bbox = entry['bbox']
                    label_id = entry['category_id']
                    # Assuming bbox is in the format [x, y, width, height]
                    prep_entry += "{} {} {} {} {} {} ".format(
                        label_id,
                        entry['score'],
                        bbox[0],
                        bbox[1],
                        bbox[2] + bbox[0],
                        bbox[3] + bbox[1]
                    )
                prep_entry += "14 1 0 0 1 1"
                intermediate.append([img_name, prep_entry.strip()])
            else:
                intermediate.append([img_name, "14 1 0 0 1 1"])

    # Write results to csv
    with open(output, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "PredictionString"])
        writer.writerows(intermediate)


def main():
    args = parse_args()
    vindrcxr_to_kaggle_submission_format(
        args.input_file,
        args.output_file,
        target_csv=args.target_csv,
        apply_target_filter=args.apply_target_filter,
        target_threshold=args.target_threshold
    )


if __name__ == '__main__':
    main()