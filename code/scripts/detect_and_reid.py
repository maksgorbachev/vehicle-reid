"""
End-to-end Vehicle Detection + ReID Pipeline
=============================================
YOLOv11 detects vehicles in images, ReID model matches them across cameras.

Usage:
    python scripts/detect_and_reid.py \
        --query_dir data/VeRi/image_query \
        --gallery_dir data/VeRi/image_test \
        --checkpoint checkpoints/vit_best.pth \
        --model_type vit \
        --n_queries 5 --top_k 3 \
        --output ../images/yolo_reid_demo.png
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Run YOLOv11 on image, return (PIL crop, bbox) or (full image, None) if no detection."""
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
        return crop, best_box
    return img, None  # fallback: full image


@torch.no_grad()
def embed_images(model, image_paths, transform, device, batch_size=32):
    """Extract L2-normalized embeddings for list of file paths."""
    all_feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Gallery embeddings"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)
        feats = model(batch)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = F.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def embed_crops(model, crops, transform, device, batch_size=32):
    """Extract embeddings for list of PIL Images."""
    all_feats = []
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        imgs = torch.stack([transform(c) for c in batch_crops]).to(device)
        feats = model(imgs)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = F.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def visualize_results(query_img, query_bbox, gallery_paths, distances, top_k, ax_row):
    """Draw one row: query with bbox + top-k gallery matches."""
    ax_row[0].imshow(query_img)
    ax_row[0].set_title('Query', fontsize=8)
    if query_bbox is not None:
        x1, y1, x2, y2 = query_bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax_row[0].add_patch(rect)
    ax_row[0].axis('off')

    for j, (path, dist) in enumerate(zip(gallery_paths[:top_k], distances[:top_k])):
        img = Image.open(path).convert('RGB')
        ax_row[j + 1].imshow(img)
        ax_row[j + 1].set_title(f'd={dist:.3f}', fontsize=7)
        ax_row[j + 1].axis('off')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dir', required=True)
    parser.add_argument('--gallery_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model_type', default='vit', choices=['resnet50', 'vit'])
    parser.add_argument('--n_queries', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--output', default='../images/yolo_reid_demo.png')
    parser.add_argument('--yolo_model', default='yolo11n.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading YOLOv11...")
    yolo = YOLO(args.yolo_model)

    print("Loading ReID model...")
    reid_model, img_size = load_reid_model(args.checkpoint, args.model_type, device)
    transform = get_test_transforms(img_size=img_size)

    query_paths = sorted([
        os.path.join(args.query_dir, f)
        for f in os.listdir(args.query_dir) if f.endswith('.jpg')
    ])[:args.n_queries]

    gallery_paths = sorted([
        os.path.join(args.gallery_dir, f)
        for f in os.listdir(args.gallery_dir) if f.endswith('.jpg')
    ])

    print(f"Queries: {len(query_paths)} | Gallery: {len(gallery_paths)}")

    print("Detecting vehicles in queries...")
    query_crops, query_bboxes, fallback_count = [], [], 0
    for path in tqdm(query_paths):
        crop, bbox = detect_vehicle_crop(yolo, path)
        query_crops.append(crop)
        query_bboxes.append(bbox)
        if bbox is None:
            fallback_count += 1
    print(f"  Fallback (no detection): {fallback_count}/{len(query_paths)}")

    print("Extracting gallery embeddings...")
    gallery_feats = embed_images(reid_model, gallery_paths, transform, device)

    print("Extracting query embeddings from YOLO crops...")
    query_feats = embed_crops(reid_model, query_crops, transform, device)

    sim = query_feats @ gallery_feats.T  # [Q, G]
    dist = 1 - sim

    fig, axes = plt.subplots(
        len(query_paths), args.top_k + 1,
        figsize=(2.5 * (args.top_k + 1), 2.5 * len(query_paths))
    )
    if len(query_paths) == 1:
        axes = [axes]

    for i, (q_path, crop, bbox) in enumerate(zip(query_paths, query_crops, query_bboxes)):
        sorted_idx = np.argsort(dist[i])
        top_gallery = [gallery_paths[j] for j in sorted_idx[:args.top_k]]
        top_dist = dist[i][sorted_idx[:args.top_k]]
        q_img = Image.open(q_path).convert('RGB')
        visualize_results(q_img, bbox, top_gallery, top_dist, args.top_k, axes[i])

    plt.suptitle('YOLOv11 Detection + Vehicle ReID', fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
