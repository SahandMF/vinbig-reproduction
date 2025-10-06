## cxr-detection-replication

Unified chest X-ray detection toolkit combining MMDetection-based pipelines (DETR/Co-DETR/YOLOv5) and an EfficientDet training stack, with utilities for evaluation, ensembling, and CV orchestration.

### Overview
- **Project 1 (MMDet/MMYOLO)**: configs under `cxr_detection_replication/configs`, entrypoints in `cxr_detection_replication/tools`.
- **Project 2 (EfficientDet)**: located at `cxr_detection_replication/efficientdet` with its native trainer and a CV orchestrator.
- Unified trainer `cxr_detection_replication/tools/train.py` now supports both ecosystems via `--trainer {mmdet,effdet}`.

This repo targets the VinBigData Chest X-ray Abnormalities Detection task (14 classes) and includes helpers for COCO-style evaluation and Kaggle CSV conversion.

### Installation (known-good matrix)
This project was developed with Python 3.9 (conda env `cxr-python39`). The following matrix is verified for both EfficientDet and MMDetection:
- torch 1.13.1 + cu117
- numpy 1.23.5
- mmcv 2.0.1, mmdet 3.3.0 (installed via openmim)
- timm 1.0.20 (EfficientDet)

```bash
conda create -n cxr-py39 python=3.9 -y
conda activate cxr-py39

# Core first (ensures torch’s numpy bridge works)
pip install --upgrade pip setuptools wheel
pip install "numpy==1.23.5"

# Torch (CUDA 11.7 wheels)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# MM stack via openmim (pulls matching wheels for current torch/CUDA)
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.3.0"

# EfficientDet + YOLO helpers
pip install "timm==1.0.20" "mmyolo==0.6.0"

# Project deps (pinned to avoid numpy>=2 pressure) + editable install
pip install -r requirements.txt
pip install -e .
```

Notes:
- If MMDet/MMCV install issues occur, consult MMDetection docs (`mmdet`/`mmcv` versions must be compatible).
- EfficientDet side depends on `timm`, `pycocotools`, `torchvision`, etc. See `efficientdet/requirements.txt` and `requirements.txt`.

### Data layout
Use COCO-style JSONs. For MMDet, pass `--ann-dir` pointing to a directory with `train.json` and `val.json` (or `train_fold_{k}.json`/`val_fold_{k}.json` for CV), and `--data-path` pointing to the images root. For EfficientDet with `--dataset coco`, the default expects:

```
<root>/train2017/
<root>/val2017/
<root>/annotations_coco/instances_train2017.json
<root>/annotations_coco/instances_val2017.json
```

You can also override COCO JSONs per run for EfficientDet with `--ann-train-file`/`--ann-val-file`.

### Usage

MMDet/Co-DETR training (example):
```bash
python cxr_detection_replication/tools/train.py \
  /abs/path/to/config.py \
  --ann-dir /abs/path/to/train_val_split \
  --data-path /abs/path/to/data_root \
  --work-dir /abs/path/to/work_dir
```

EfficientDet single run:
```bash
python cxr_detection_replication/tools/train.py \
  --trainer effdet \
  --effdet-root /abs/path/to/data_root \
  --effdet-dataset coco \
  --effdet-model tf_efficientdet_d2 \
  --effdet-num-classes 14 \
  --effdet-initial-checkpoint /abs/path/to/checkpoint.pth.tar \
  --effdet-batch-size 8 --effdet-epochs 25 --effdet-workers 4 --effdet-amp \
  --effdet-output /abs/path/to/output
```

EfficientDet CV (built-in orchestrator):
```bash
python cxr_detection_replication/tools/train.py \
  --trainer effdet \
  --effdet-root /abs/path/to/data_root \
  --effdet-dataset coco \
  --effdet-model tf_efficientdet_d2 \
  --effdet-num-classes 14 \
  --effdet-initial-checkpoint /abs/path/to/checkpoint.pth.tar \
  --effdet-use-cv --effdet-cv 5 \
  --effdet-cv-dir /abs/path/to/cv_folds \
  --effdet-output /abs/path/to/output_cv
```

### Testing examples (tools/test.py)

Single MMDet checkpoint
```bash
python cxr_detection_replication/tools/test.py \
  configs/vindrcxr/faster-rcnn.py \
  work_dirs/faster_rcnn/train_val/epoch_12.pth \
  --model-types mmdet \
  --test-ann-file data/test.json \
  --data-root data/vinbig_cxr2 \
  --batch-size 8 \
  --output-dir work_dirs/test_runs/faster_rcnn \
  --json-prefix faster_rcnn
```

Multiple MMDet checkpoints with ensembling (WBF)
```bash
python cxr_detection_replication/tools/test.py \
  configs/vindrcxr/vfnet_finetuned.py \
  work_dirs/vfnet/cv5/fold_0/epoch_10.pth work_dirs/vfnet/cv5/fold_1/epoch_10.pth \
  --model-types mmdet mmdet \
  --test-ann-file data/test.json \
  --data-root data/vinbig_cxr2 \
  --batch-size 8 \
  --output-dir work_dirs/test_runs/vfnet_ensemble \
  --json-prefix vfnet \
  --ensemble --weights 1.0 1.0 --iou-thr 0.6 --skip-box-thr 0.0001 --score-thr 0.2
```

Single EfficientDet checkpoint
```bash
python cxr_detection_replication/tools/test.py \
  "" \
  efficientdet/output/d2_train_val/best.pth.tar \
  --model-types effdet \
  --effdet-configs efficientdet/configs/example.yaml \
  --test-ann-file data/test.json \
  --data-root data/vinbig_cxr2 \
  --batch-size 8 \
  --output-dir work_dirs/test_runs/effdet_d2 \
  --json-prefix effdet
```

Mixed MMDet + EfficientDet ensemble
```bash
python cxr_detection_replication/tools/test.py \
  configs/vindrcxr/vindrcxr_detr_r50_improved_aug_150e.py \
  work_dirs/detr_r50/train_val/epoch_100.pth efficientdet/output/d2_train_val/best.pth.tar \
  --model-types mmdet effdet \
  --effdet-configs "" efficientdet/configs/example.yaml \
  --test-ann-file data/test.json \
  --data-root data/vinbig_cxr2 \
  --output-dir work_dirs/test_runs/mixed_ensemble \
  --json-prefix mixed \
  --ensemble --weights 1.0 1.0
```

### Command-line arguments

MMDetection trainer (`cxr_detection_replication/tools/train.py`, default `--trainer mmdet`):
- `config` (positional): path to MMDet config.
- `--ann-dir`: directory with `train.json`/`val.json` or `train_fold_{k}.json`/`val_fold_{k}.json`.
- `--data-path`: images root.
- `--work-dir`: logs/checkpoints dir (auto-created if omitted).
- `--use-cv`: enable cross-validation.
- `--cv`: number of folds (default: 4).
- `--amp`: enable mixed precision.
- `--resume [path|auto]`: resume training.
- `--gpu-ids`: override GPU IDs from config.
- `--launcher`: mmdet launcher (`none|pytorch|slurm|mpi`).
- `--use_wandb`, `--wandb-project`: enable W&B logging.
- `--use-yolo`: use YOLOv5 dataset format.
- `--batch-size`: override batch size.
- `--cfg-options`: MMDet config overrides (key=val).

EfficientDet trainer (`--trainer effdet`):
- `--effdet-root`: dataset root (COCO layout) or use `--data-path`.
- `--effdet-dataset`: dataset name (default: coco).
- `--effdet-model`: e.g., `tf_efficientdet_d2`.
- `--effdet-num-classes`: number of classes.
- `--effdet-initial-checkpoint`: pretrained checkpoint path.
- `--effdet-batch-size`, `--effdet-epochs`, `--effdet-workers`, `--effdet-amp`.
- `--effdet-output`: output dir.
- `--effdet-ann-train-file`, `--effdet-ann-val-file`: override COCO JSONs (useful for CV folds).
- `--effdet-use-cv`, `--effdet-cv`, `--effdet-cv-dir`: built-in CV loop.
- EfficientDet native flags (within trainer): `--save-samples` to save sample images.

Test harness (`cxr_detection_replication/tools/test.py`):
- Positional:
  - `config` (optional, required if any model is MMDet)
  - `checkpoints` (one or more paths)
- Required:
  - `--model-types`: one per checkpoint (`mmdet` or `effdet`)
  - `--test-ann-file`: COCO test annotation JSON
- Optional data/runtime:
  - `--effdet-configs`: YAML per checkpoint (use `""` for MMDet entries)
  - `--data-root`: dataset images root (sets `data_prefix=\u007bimg:'test/'\u007d`)
  - `--batch-size`: test-time batch size (default 8)
  - `--launcher`: job launcher (default: none)
  - `--cfg-options`: MMDet config overrides
  - `--tta`: enable TTA (requires TTA config in MMDet)
  - `--show`, `--show-dir`: visualize and/or save predictions
  - `--deploy`: switch to deploy mode (if configured)
  - `--effdet-mean`, `--effdet-std`: override EffDet normalization
- Output:
  - `--output-dir`: directory to save outputs (required)
  - `--work-dir`: temp working dir (defaults to config/work_dirs)
  - `--out-prefix`: sub-prefix when testing multiple checkpoints
  - `--json-prefix`: base prefix for saved result JSONs (required)
  - `--csv`, `--csv-output`: export Kaggle-style CSV (uses the test ann file)
- Ensembling (WBF):
  - `--ensemble`: enable WBF over per-model JSON results
  - `--weights`: one weight per model
  - `--iou-thr`, `--skip-box-thr`, `--score-thr`: WBF thresholds

### Data preprocessing (from RetMed)
Borrowed and adapted from the original RetMed repository (`https://github.com/alexkubl/retmed/tree/main`). Use `retmed/utils/preprocessing.py` to prepare datasets:

- Preprocessing strategies:
  - Weighted Boxes Fusion (`wbf`): resolves annotator disagreements via WBF to create consensus annotations.
  - Annotator Agreement (`ann_agree`): builds multiple training sets by consensus level:
    - labels-1: ≥1 annotator (inclusive)
    - labels-2: ≥2 annotators (balanced)
    - labels-3: all 3 annotators (conservative)
    - labels-4: labels-1 but only disease images (focused)

- Splitting strategies:
  - Cross validation (`cross_val`): generate CV folds (≥2).
  - Train-validation (`train_val`): create `train.json` and `val.json`.

Notes:
- `--data_root` (required) must point to your raw dataset; examples use `data/sampled_dataset.json` but you should substitute with your actual COCO JSON.
- You can re-split already preprocessed JSONs by choosing strategy `none` and pointing to an existing file.

### Training examples

MMDetection - Faster R-CNN (CV 5 folds)
```bash
python /cxr_detection_replication/tools/train.py \
  configs/vindrcxr/faster-rcnn.py \
  --ann-dir data/sample_cv/wbf_data/cv_folds  \
  --data-path data/vinbig_cxr2 \
  --use-cv --cv 5 \
  --work-dir /work_dirs/faster_rcnn/cv5
```

MMDetection - Faster R-CNN (train/val)
```bash
python /cxr_detection_replication/tools/train.py \
  /configs/vindrcxr/faster-rcnn.py \
  --ann-dir data/sample_wbf/wbf_data/train_val_split \
  --data-path /data/vinbig_cxr2 \
  --work-dir /work_dirs/faster_rcnn/train_val
```

MMDetection - VFNet finetuned (CV 5 folds)
```bash
python /cxr_detection_replication/tools/train.py \
  /configs/vindrcxr/vfnet_finetuned.py \
  --ann-dir /data/sample_cv/wbf_data/cv_folds \
  --data-path /data/vinbig_cxr2 \
  --use-cv --cv 5 \
  --work-dir /work_dirs/vfnet/cv5
```

MMDetection - VFNet finetuned (train/val)
```bash
python /cxr_detection_replication/tools/train.py \
  /configs/vindrcxr/vfnet_finetuned.py \
  --ann-dir /data/sample_wbf/wbf_data/train_val_split \
  --data-path /data/vinbig_cxr2 \
  --work-dir /work_dirs/vfnet/train_val
```

MMDetection - DETR R50 (CV 5 folds)
```bash
python /cxr_detection_replication/tools/train.py \
  /configs/vindrcxr/vindrcxr_detr_r50_improved_aug_150e.py \
  --ann-dir /data/sample_cv/wbf_data/cv_folds \
  --data-path /data/vinbig_cxr2 \
  --use-cv --cv 5 \
  --work-dir /work_dirs/detr_r50/cv5
```

MMDetection - DETR R50 (train/val)
```bash
python /cxr_detection_replication/tools/train.py \
  /configs/vindrcxr/vindrcxr_detr_r50_improved_aug_150e.py \
  --ann-dir /data/sample_wbf/wbf_data/train_val_split \
  --data-path /data/vinbig_cxr2 \
  --work-dir /work_dirs/detr_r50/train_val
```

MMDetection - YOLOv5x finetuned (CV 5 folds)
```bash
python /cxr_detection_replication/tools/train.py \
  //configs/vindrcxr/vindrcxr_yolov5_x_finetuned.py \
  --ann-dir /data/sample_cv/wbf_data/cv_folds \
  --data-path /data/vinbig_cxr2 \
  --use-yolo --use-cv --cv 5 \
  --work-dir /work_dirs/yolov5x/cv5
```

MMDetection - YOLOv5x finetuned (train/val)
```bash
python /cxr_detection_replication/tools/train.py \
  /configs/vindrcxr/vindrcxr_yolov5_x_finetuned.py \
  --ann-dir /data/sample_wbf/wbf_data/train_val_split \
  --data-path /data/vinbig_cxr2 \
  --use-yolo \
  --work-dir /work_dirs/yolov5x/train_val
```

EfficientDet D2 (no pretrained) - CV 5 folds
```bash
python /cxr_detection_replication/tools/train.py \
  --trainer effdet \
  --effdet-root /data/vinbig_cxr2 \
  --effdet-dataset coco \
  --effdet-model tf_efficientdet_d2 \
  --effdet-num-classes 14 \
  --effdet-batch-size 8 --effdet-epochs 25 --effdet-workers 4 --effdet-amp \
  --effdet-use-cv --effdet-cv 5 \
  --effdet-cv-dir /data/sample_cv/wbf_data/cv_folds \
  --effdet-output /output/d2_cv
```

EfficientDet D2 (no pretrained) - train/val
```bash
python /cxr_detection_replication/tools/train.py \
  --trainer effdet \
  --effdet-root /data/vinbig_cxr2 \
  --effdet-dataset coco \
  --effdet-model tf_efficientdet_d2 \
  --effdet-num-classes 14 \
  --effdet-batch-size 8 --effdet-epochs 25 --effdet-workers 4 --effdet-amp \
  --effdet-ann-train-file /data/sample_wbf/wbf_data/train_val_split/train.json \
  --effdet-ann-val-file   /data/sample_wbf/wbf_data/train_val_split/val.json \
  --effdet-output /cxr_detection_replication/efficientdet/output/d2_train_val
```

### GPU configuration
- Set `CUDA_VISIBLE_DEVICES` to select GPUs
  - Single GPU: `CUDA_VISIBLE_DEVICES=0`
  - Multiple GPUs: `CUDA_VISIBLE_DEVICES=0,1`

### Advanced options
- Mixed precision training: add `--amp`
- Resume training: add `--resume` (optionally a path or `auto` where supported)
- GPU selection: combine your command with `CUDA_VISIBLE_DEVICES=...`

### Weights & Biases (W&B)
Use `--use_wandb` (and optionally `--wandb-project <name>`) to:
- Track metrics (loss, mAP, etc.)
- Log model checkpoints
- Record hyperparameters
- Save configuration
- Visualize training curves

### RetMed lineage
This repository reuses and extends parts of the RetMed pipeline. For more background, see `cxr_detection_replication/retmed` and the original RetMed README structure.

### Development
- Python version: 3.9 (recommended)
- Code style: keep edits minimal and avoid breaking existing configs.
- Optional: enable W&B logging with `--use_wandb` (MMDet) or `wandb` configured in your environment.

### License
See individual subdirectories and third-party components for their respective licenses.


