"""
Generate yolo_reid_demo.png: YOLOv11 query bbox + top-3 gallery matches.

Usage (from code/ directory):
    python scripts/visualize_yolo_reid.py \
        --checkpoint checkpoints/vit_best.pth \
        --data_dir data/VeRi \
        --output ../images/yolo_reid_demo.png
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_reid.transforms import get_test_transforms
from vehicle_reid.dataset import VeRiDataset

VEHICLE_CLASSES = {2, 5, 7}  # car, bus, truck in COCO


def load_reid_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    num_classes = 576
    for key in state:
        if 'classifier.weight' in key:
            num_classes = state[key].shape[0]
            break
    from vehicle_reid.vit_model import build_vit_model
    model = build_vit_model(num_classes=num_classes, model_size='small', img_size=224)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def embed_pil(model, img_pil, device):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    t = tf(img_pil).unsqueeze(0).to(device)
    feat = model(t)
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    return F.normalize(feat, p=2, dim=1).cpu().numpy()[0]


@torch.no_grad()
def build_gallery(model, gallery_ds, device, batch_size=64):
    loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels, cams, paths = [], [], [], []
    for batch in tqdm(loader, desc='Gallery embeddings'):
        imgs = batch['image'].to(device)
        f = model(imgs)
        if isinstance(f, (tuple, list)):
            f = f[0]
        f = F.normalize(f, p=2, dim=1)
        feats.append(f.cpu().numpy())
        labels.append(batch['label'].numpy())
        cams.append(batch['camera_id'].numpy())
    # paths in same order as data
    for img_path, vid, cam in gallery_ds.data:
        paths.append(img_path)
    return (
        np.concatenate(feats),
        np.concatenate(labels),
        np.concatenate(cams),
        paths,
    )


def detect_and_crop(yolo_model, img_path):
    """Returns (pil_crop, bbox_xyxy_or_None, pil_full)."""
    img = Image.open(img_path).convert('RGB')
    results = yolo_model(img_path, verbose=False)
    boxes = results[0].boxes
    best_conf, best_box = -1, None
    if boxes is not None and len(boxes):
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls in VEHICLE_CLASSES and conf > 0.3 and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy().astype(int)
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        crop = img.crop((max(0, x1), max(0, y1), x2, y2))
        return crop, best_box, img
    return img, None, img


def make_figure(query_full, bbox, gallery_imgs, distances, out_path):
    n = len(gallery_imgs)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.8))

    ax = axes[0]
    ax.imshow(query_full)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title('Query\n(YOLOv11\ndetection)', fontsize=9)
    ax.axis('off')

    for i, (img_pil, dist) in enumerate(zip(gallery_imgs, distances)):
        ax = axes[i + 1]
        ax.imshow(img_pil)
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
        ax.set_title(f'Top-{i+1}\ndist {dist:.2f}', fontsize=9)
        ax.axis('off')

    plt.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/vit_best.pth')
    parser.add_argument('--data_dir', default='data/VeRi')
    parser.add_argument('--output', default='../images/yolo_reid_demo.png')
    parser.add_argument('--query_idx', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_reid_model(args.checkpoint, device)

    tf = get_test_transforms(img_size=(224, 224))
    query_ds = VeRiDataset(args.data_dir, split='query', transform=tf)
    gallery_ds = VeRiDataset(args.data_dir, split='gallery', transform=tf)

    gallery_feats, gallery_labels, gallery_cams, gallery_paths = \
        build_gallery(model, gallery_ds, device)

    from ultralytics import YOLO as UltralyticsYOLO
    yolo = UltralyticsYOLO('yolo11n.pt')

    # Try queries until we get a detection
    candidates = list(range(len(query_ds)))
    random.shuffle(candidates)
    if args.query_idx is not None:
        candidates = [args.query_idx] + candidates

    chosen_idx = None
    chosen_crop = chosen_bbox = chosen_full = None

    for idx in candidates[:80]:
        img_path, vid, cam = query_ds.data[idx]
        crop, bbox, full = detect_and_crop(yolo, img_path)
        if bbox is not None:
            chosen_idx = idx
            chosen_crop = crop
            chosen_bbox = bbox
            chosen_full = full
            print(f'Query #{idx}: {os.path.basename(img_path)}, bbox={bbox}')
            break

    if chosen_idx is None:
        idx = 0
        img_path, vid, cam = query_ds.data[idx]
        chosen_full = Image.open(img_path).convert('RGB')
        chosen_crop = chosen_full
        chosen_bbox = None
        chosen_idx = idx
        print(f'No detection found, using full image #{idx}')

    q_feat = embed_pil(model, chosen_crop, device)
    sims = gallery_feats @ q_feat
    dists = 1.0 - sims

    # Exclude same-vehicle same-camera (junk)
    q_vid = query_ds.data[chosen_idx][1]
    q_cam = query_ds.data[chosen_idx][2]
    valid_mask = ~((gallery_labels == q_vid) & (gallery_cams == q_cam))
    valid_idx = np.where(valid_mask)[0]
    top_local = np.argsort(dists[valid_idx])[:args.top_k]
    top_global = valid_idx[top_local]
    top_dists = dists[top_global]

    top_imgs = [Image.open(gallery_paths[i]).convert('RGB') for i in top_global]
    make_figure(chosen_full, chosen_bbox, top_imgs, top_dists, args.output)


if __name__ == '__main__':
    main()
