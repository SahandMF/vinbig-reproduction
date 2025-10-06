import os
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
from ensemble_boxes import weighted_boxes_fusion
import random
import matplotlib.pyplot as plt

def load_model(model_name, checkpoint_path, num_classes, device):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes  # Force config to use 14 classes
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    model = DetBenchPredict(net)
    model.eval()
    model.to(device)
    return model

def draw_boxes(img, boxes, labels, color, class_names, scores=None, thickness=3, text_thickness=2):
    img = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = class_names[labels[i]]
        if scores is not None:
            label += f': {scores[i]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, text_thickness)
    return img

def main(
    model_name,
    checkpoint_path,
    coco_ann_file,
    img_dir,
    output_dir,
    num_samples=5,
    device='cuda'
):
    # Load COCO
    coco = COCO(coco_ann_file)
    img_ids = coco.getImgIds()
    # Only images with GT
    img_ids_with_anns = [img_id for img_id in img_ids if coco.getAnnIds(imgIds=img_id)]
    if len(img_ids_with_anns) < num_samples:
        print(f"Warning: Only {len(img_ids_with_anns)} images found with annotations. Visualizing all of them.")
        sample_ids = img_ids_with_anns
    else:
        sample_ids = random.sample(img_ids_with_anns, num_samples)

    # Get class names
    cats = coco.loadCats(coco.getCatIds())
    class_names = [cat['name'] for cat in cats]
    class_id_to_idx = {cat['id']: idx for idx, cat in enumerate(cats)}

    # Load model
    num_classes = 14
    model = load_model(model_name, checkpoint_path, num_classes, device)

    os.makedirs(output_dir, exist_ok=True)

    for img_id in tqdm(sample_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # GT
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            gt_boxes.append([int(x), int(y), int(x+bw), int(y+bh)])
            gt_labels.append(class_id_to_idx[ann['category_id']])

        # Prediction
        input_size = 768  # for tf_efficientdet_d2
        img_resized = cv2.resize(img_rgb, (input_size, input_size))
        input_img = img_resized.astype(np.float32) / 255.0
        input_img = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            det = model(input_img)[0]

        # If det is a tensor of shape (N, 6)
        if isinstance(det, torch.Tensor):
            if det.numel() == 0 or det.shape[0] == 0:
                print(f"No predictions for image {img_info['file_name']}, skipping.")
                continue
            det_np = det.cpu().numpy()
            pred_boxes = det_np[:, :4]
            pred_scores = det_np[:, 4]
            pred_labels = det_np[:, 5].astype(int)
        else:
            print(f"Unknown prediction output type for image {img_info['file_name']}, skipping.")
            continue

        # Scale boxes back to original image size
        scale_x = img.shape[1] / input_size
        scale_y = img.shape[0] / input_size
        pred_boxes[:, [0, 2]] = pred_boxes[:, [0, 2]] * scale_x
        pred_boxes[:, [1, 3]] = pred_boxes[:, [1, 3]] * scale_y

        # Filter predictions (optional: top N or score threshold)
        keep = pred_scores > 0.05
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # Apply WBF to predictions
        # Normalize boxes to [0, 1]
        norm_boxes = []
        for box in pred_boxes:
            norm_boxes.append([
                box[0] / img.shape[1],
                box[1] / img.shape[0],
                box[2] / img.shape[1],
                box[3] / img.shape[0]
            ])
        if len(norm_boxes) == 0:
            print(f"No predictions for image {img_info['file_name']} after filtering, skipping.")
            continue

        # WBF expects a list of predictions from different models, so we wrap in another list
        wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
            [norm_boxes], [pred_scores.tolist()], [pred_labels.tolist()],
            iou_thr=0.5, skip_box_thr=0.05
        )

        # Convert WBF boxes back to absolute coordinates
        wbf_boxes_abs = []
        for box in wbf_boxes:
            wbf_boxes_abs.append([
                int(box[0] * img.shape[1]),
                int(box[1] * img.shape[0]),
                int(box[2] * img.shape[1]),
                int(box[3] * img.shape[0])
            ])
        wbf_labels = [int(l) for l in wbf_labels]
        wbf_scores = [float(s) for s in wbf_scores]

        # Draw GT and WBF predictions
        img_vis = draw_boxes(img_rgb, gt_boxes, gt_labels, (255, 255, 0), class_names, thickness=3, text_thickness=2)
        img_vis = draw_boxes(img_vis, wbf_boxes_abs, wbf_labels, (0, 128, 255), class_names, wbf_scores, thickness=3, text_thickness=2)

        # Save
        out_filename = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        out_path = os.path.join(output_dir, f'{out_filename}_vis.png')
        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(out_path, img_bgr)
        print(f"Saving to {out_path}: {'Success' if success else 'Failed'}")

if __name__ == '__main__':
    main(
        model_name='tf_efficientdet_d2',
        checkpoint_path='/home/sahand/output/train/20250512-152724-tf_efficientdet_d2/checkpoint-19.pth.tar',
        coco_ann_file='/home/sahand/datasets/vinbig_cxr2/annotations_coco/instances_train2017.json',
        img_dir='/home/sahand/datasets/vinbig_cxr2/train2017',
        output_dir='/home/sahand/visualization_output',
        num_samples=5,
        device='cuda'
    )