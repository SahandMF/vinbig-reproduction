import argparse
import os
import sys
from types import SimpleNamespace
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Import from the effdet package
from effdet.data.dataset import DetectionDatset
from effdet.data.transforms import ImageToTensor
from pathlib import Path


def compute_mean_std(dataset, batch_size=64, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, _ in tqdm(loader, desc="Computing mean/std"):
        data = data.float() / 255.0  # Normalize to [0, 1]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # (B, C, H*W)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean.tolist(), std.tolist()

def main():
    parser = argparse.ArgumentParser(description="Compute dataset mean and std for object detection datasets.")
    parser.add_argument('--img-dir', type=str, required=True, help='Path to folder containing images (e.g., train2017/).')
    parser.add_argument('--annotation-file', type=str, required=True, help='Path to COCO annotation file (.json).')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for DataLoader (default: 1).')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers (default: 4).')
    args = parser.parse_args()

    # Create cfg as a Namespace (NOT a dict)
    cfg = SimpleNamespace(
    annotation_file=os.path.expanduser(args.annotation_file),
    ann_filename=os.path.expanduser(args.annotation_file),
    img_dir=os.path.expanduser(args.img_dir),
    remove_images_without_annotations=False,  # safe default
    bbox_yxyx=False,                           # safe default
    has_labels=True,                            # ✅ NEW
    include_masks=False,
    include_bboxes_ignore= False,
    ignore_empty_gt = False,
    min_img_size= 0
)


    dataset = DetectionDatset(
        data_dir=Path(os.path.expanduser(args.img_dir)),
        parser='coco',
        parser_kwargs={'cfg': cfg},
        transform=ImageToTensor()
    )

    mean, std = compute_mean_std(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"\n✅ Dataset Mean: {mean}")
    print(f"✅ Dataset Std: {std}")

if __name__ == '__main__':
    main()
