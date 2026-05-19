"""
Quantitative comparison: GT crop vs YOLO-detected crop for ReID.

Usage:
    python scripts/evaluate_yolo_crop.py \
        --model_type vit \
        --checkpoint checkpoints/vit_best.pth \
        --data_dir data/VeRi \
        --output results/results_yolo_crop.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_reid.dataset import VeRiDataset
from vehicle_reid.evaluation import evaluate_reid
from vehicle_reid.transforms import get_test_transforms
from ultralytics import YOLO

VEHICLE_CLASSES = {2, 5, 7}  # car, bus, truck in COCO


def load_reid_model(checkpoint: str, model_type: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    num_classes = 576
    for key in state:
        if 'classifier.weight' in key:
            num_classes = state[key].shape[0]
            break

    if model_type == 'resnet50':
        from vehicle_reid.model import build_model
        model = build_model(num_classes=num_classes)
        img_size = (256, 128)
    else:
        from vehicle_reid.vit_model import build_vit_model
        model = build_vit_model(num_classes=num_classes, model_size='small', img_size=224)
        img_size = (224, 224)

    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model, img_size


def detect_vehicle_crop(yolo_model, img_path: str):
    """Return (PIL crop, detected) where detected=False means fallback to full image."""
    img = Image.open(img_path).convert('RGB')
    results = yolo_model(img_path, verbose=False)
    boxes = results[0].boxes

    best_conf = -1
    best_box = None
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls in VEHICLE_CLASSES and conf > 0.3 and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy().astype(int)

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        x1, y1 = max(0, x1), max(0, y1)
        crop = img.crop((x1, y1, x2, y2))
        return crop, True
    return img, False  # fallback: full image


@torch.no_grad()
def extract_features_from_dataset(model, dataset, device, batch_size=64):
    """Extract embeddings using VeRiDataset (GT crops, uses dataset's own transform)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features, labels, cameras = [], [], []
    for batch in tqdm(loader, desc="GT embeddings"):
        imgs = batch['image'].to(device)
        feats = model(imgs)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = F.normalize(feats, p=2, dim=1)
        features.append(feats.cpu().numpy())
        labels.append(batch['label'].numpy())
        cameras.append(batch['camera_id'].numpy())
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(cameras),
    )


@torch.no_grad()
def extract_features_yolo(model, yolo_model, dataset, transform, device, batch_size=32):
    """Extract embeddings using YOLO-detected crops from query images."""
    features = []
    labels = []
    cameras = []
    fallback_flags = []

    imgs_batch, labs_batch, cams_batch, fall_batch = [], [], [], []

    for item in tqdm(dataset.data, desc="YOLO crop embeddings"):
        img_path, vehicle_id, camera_id = item
        crop, detected = detect_vehicle_crop(yolo_model, img_path)
        imgs_batch.append(transform(crop))
        labs_batch.append(vehicle_id)
        cams_batch.append(camera_id)
        fall_batch.append(not detected)

        if len(imgs_batch) == batch_size:
            batch_tensor = torch.stack(imgs_batch).to(device)
            feats = model(batch_tensor)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = F.normalize(feats, p=2, dim=1)
            features.append(feats.cpu().numpy())
            labels.extend(labs_batch)
            cameras.extend(cams_batch)
            fallback_flags.extend(fall_batch)
            imgs_batch, labs_batch, cams_batch, fall_batch = [], [], [], []

    if imgs_batch:
        batch_tensor = torch.stack(imgs_batch).to(device)
        feats = model(batch_tensor)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = F.normalize(feats, p=2, dim=1)
        features.append(feats.cpu().numpy())
        labels.extend(labs_batch)
        cameras.extend(cams_batch)
        fallback_flags.extend(fall_batch)

    return (
        np.concatenate(features),
        np.array(labels),
        np.array(cameras),
        fallback_flags,
    )


def print_table(rows):
    print("\n" + "=" * 65)
    print(f"{'Method':<25} {'Rank-1':>8} {'Rank-5':>8} {'mAP':>8} {'Fallback':>10}")
    print("-" * 65)
    for row in rows:
        fb = row.get('fallback_pct', '—')
        fb_str = f"{fb:.1f}%" if isinstance(fb, float) else str(fb)
        print(f"{row['method']:<25} {row['Rank-1']:>7.2f}% {row['Rank-5']:>7.2f}% {row['mAP']:>7.2f}% {fb_str:>10}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='vit', choices=['resnet50', 'vit'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output', default='results/results_yolo_crop.json')
    parser.add_argument('--yolo_model', default='yolo11n.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    reid_model, img_size = load_reid_model(args.checkpoint, args.model_type, device)
    transform = get_test_transforms(img_size=img_size)

    query_dataset = VeRiDataset(root=args.data_dir, split='query', transform=transform)
    gallery_dataset = VeRiDataset(root=args.data_dir, split='gallery', transform=transform)
    print(f"Query: {len(query_dataset)} | Gallery: {len(gallery_dataset)}")

    print("\n[1/3] GT gallery embeddings...")
    g_feat, g_labels, g_cams = extract_features_from_dataset(
        reid_model, gallery_dataset, device, args.batch_size
    )

    print("[2/3] GT query embeddings (baseline)...")
    q_feat_gt, q_labels, q_cams = extract_features_from_dataset(
        reid_model, query_dataset, device, args.batch_size
    )

    results_gt = evaluate_reid(q_feat_gt, g_feat, q_labels, g_labels, q_cams, g_cams)
    print(f"  Rank-1={results_gt['Rank-1']:.2f}%  mAP={results_gt['mAP']:.2f}%")

    print("[3/3] YOLO-crop query embeddings...")
    yolo = YOLO(args.yolo_model)
    q_feat_yolo, q_labels_yolo, q_cams_yolo, fallback_flags = extract_features_yolo(
        reid_model, yolo, query_dataset, transform, device, args.batch_size
    )

    fallback_count = int(sum(fallback_flags))
    fallback_pct = 100.0 * fallback_count / len(fallback_flags)
    print(f"  Fallback: {fallback_count}/{len(fallback_flags)} = {fallback_pct:.1f}%")

    results_yolo = evaluate_reid(
        q_feat_yolo, g_feat, q_labels_yolo, g_labels, q_cams_yolo, g_cams
    )
    print(f"  Rank-1={results_yolo['Rank-1']:.2f}%  mAP={results_yolo['mAP']:.2f}%")

    rows = [
        {'method': f'{args.model_type.upper()} (GT crop)', **results_gt, 'fallback_pct': '—'},
        {'method': f'{args.model_type.upper()} (YOLO crop)', **results_yolo, 'fallback_pct': fallback_pct},
    ]
    print_table(rows)

    def to_json(d):
        return {k: float(v) for k, v in d.items()}

    output = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'yolo_model': args.yolo_model,
        'gt_crop': to_json(results_gt),
        'yolo_crop': to_json(results_yolo),
        'fallback_count': fallback_count,
        'fallback_pct': float(fallback_pct),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
