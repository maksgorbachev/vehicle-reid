"""
Mahalanobis Distance Evaluation for Vehicle ReID
=================================================
Compares L2 vs Mahalanobis retrieval for both checkpoints.

Covariance estimation:
  - ViT-Small  (D=384):  full covariance  + ridge regularization
  - ResNet-50  (D=2048): diagonal covariance (computational constraint)

Usage:
    cd code
    python scripts/eval_mahalanobis.py \
        --resnet_ckpt checkpoints/resnet50_best.pth \
        --vit_ckpt    checkpoints/vit_best.pth \
        --data_dir    data/VeRi
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_reid.evaluation import evaluate_reid, compute_ap, compute_cmc


# --------------------------------------------------------------------------- #
# Mahalanobis helpers
# --------------------------------------------------------------------------- #

def compute_dist_mahalanobis(query_feats, gallery_feats, train_feats,
                             use_diagonal=False, batch_size=128):
    """
    Mahalanobis distance matrix [Q, G].
    use_diagonal=True: diagonal-only covariance (fast for D>=512).
    Computed in row-batches to avoid OOM.
    """
    if use_diagonal:
        # Diagonal Mahalanobis = L2 on whitened features.
        # Use ||q-g||^2 = ||q||^2 + ||g||^2 - 2*q@g.T  — O(Q*G) memory only.
        var = train_feats.var(axis=0) + 1e-5           # [D]
        scale = 1.0 / np.sqrt(var)
        q = (query_feats   * scale).astype(np.float32) # [Q, D]
        g = (gallery_feats * scale).astype(np.float32) # [G, D]
        qq = (q ** 2).sum(axis=1, keepdims=True)        # [Q, 1]
        gg = (g ** 2).sum(axis=1, keepdims=True)        # [G, 1]
        dist2 = qq + gg.T - 2.0 * (q @ g.T)            # [Q, G]
        return np.sqrt(np.maximum(dist2, 0))            # [Q, G]

    # Full covariance with ridge regularization
    cov = np.cov(train_feats, rowvar=False)           # [D, D]
    cov += 1e-5 * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)                      # [D, D]

    Q = query_feats.shape[0]
    G = gallery_feats.shape[0]
    dist = np.zeros((Q, G), dtype=np.float32)
    for i in range(Q):
        diff = gallery_feats - query_feats[i]         # [G, D]
        tmp  = diff @ cov_inv                         # [G, D]
        dist[i] = np.einsum('gd,gd->g', tmp, diff)
    return np.sqrt(np.maximum(dist, 0))


def eval_with_dist(dist_mat, query_labels, gallery_labels,
                   query_cameras, gallery_cameras, max_rank=50):
    indices = np.argsort(dist_mat, axis=1)
    all_ap, all_cmc = [], np.zeros((len(query_labels), max_rank), dtype=np.float32)
    for qi in range(len(query_labels)):
        good  = np.where(gallery_labels == query_labels[qi])[0]
        junk  = np.where((gallery_labels == query_labels[qi]) &
                         (gallery_cameras == query_cameras[qi]))[0]
        all_ap.append(compute_ap(indices[qi], good, junk))
        all_cmc[qi] = compute_cmc(indices[qi], good, junk, max_rank)
    cmc = np.mean(all_cmc, axis=0) * 100
    return {
        'mAP':    float(np.mean(all_ap) * 100),
        'Rank-1': float(cmc[0]),
        'Rank-5': float(cmc[4]),
    }


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def extract_features(model, image_paths, transform, device, batch_size=64):
    model.eval()
    feats, labels, cameras = [], [], []

    def parse(path):
        name = os.path.basename(path)
        parts = name.split('_')
        vid = int(parts[0])
        cam = int(parts[1][1:]) if len(parts) > 1 else 0
        return vid, cam

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        imgs = []
        for p in batch_paths:
            imgs.append(transform(Image.open(p).convert('RGB')))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out / out.norm(dim=1, keepdim=True)
        feats.append(out.cpu().numpy())
        for p in batch_paths:
            vid, cam = parse(p)
            labels.append(vid)
            cameras.append(cam)
        if (start // batch_size) % 10 == 0:
            print(f"  {start + len(batch_paths)}/{len(image_paths)}", end='\r')

    print()
    return (np.concatenate(feats, axis=0),
            np.array(labels), np.array(cameras))


def list_images(folder):
    exts = {'.jpg', '.jpeg', '.png'}
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in exts])


# --------------------------------------------------------------------------- #
# Per-model evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(ckpt_path, model_type, data_dir, device):
    print(f"\n{'='*55}")
    print(f"  Model: {model_type}  |  {ckpt_path}")
    print(f"{'='*55}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)

    if model_type == 'resnet50':
        from vehicle_reid.model import VehicleReIDModel
        num_classes = state['classifier.weight'].shape[0] \
            if 'classifier.weight' in state \
            else state['bnneck.classifier.weight'].shape[0]
        model = VehicleReIDModel(num_classes=num_classes, pretrained=False)
        img_size = (256, 128)
        use_diagonal = True   # D=2048 too large for full cov
    else:
        from vehicle_reid.vit_model import build_vit_model
        num_classes = state['bnneck.classifier.weight'].shape[0]
        model = build_vit_model(num_classes=num_classes, model_size='small',
                                pretrained=False)
        img_size = (224, 224)
        use_diagonal = False  # D=384 is fine for full cov

    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Extracting train features...")
    train_paths = list_images(os.path.join(data_dir, 'image_train'))
    # ResNet D=2048: sample 5k for covariance estimation (diagonal anyway)
    # ViT D=384: sample 5k too — still accurate, much faster
    max_train = 5000
    if len(train_paths) > max_train:
        rng = np.random.default_rng(42)
        train_paths = list(rng.choice(train_paths, max_train, replace=False))
        print(f"  Sampled {max_train} train images for covariance estimation")
    train_feats, _, _ = extract_features(model, train_paths, transform, device)

    print("Extracting query features...")
    query_paths = list_images(os.path.join(data_dir, 'image_query'))
    q_feats, q_labels, q_cams = extract_features(model, query_paths, transform, device)

    print("Extracting gallery features...")
    gal_paths = list_images(os.path.join(data_dir, 'image_test'))
    g_feats, g_labels, g_cams = extract_features(model, gal_paths, transform, device)

    # --- L2 ---
    print("Computing L2 metrics...")
    sim = q_feats @ g_feats.T
    dist_l2 = 1 - sim
    res_l2 = eval_with_dist(dist_l2, q_labels, g_labels, q_cams, g_cams)
    print(f"  L2       Rank-1={res_l2['Rank-1']:.2f}  Rank-5={res_l2['Rank-5']:.2f}  mAP={res_l2['mAP']:.2f}")

    # --- Mahalanobis ---
    cov_type = "diagonal" if use_diagonal else "full"
    print(f"Computing Mahalanobis ({cov_type} cov, D={train_feats.shape[1]})...")
    if not use_diagonal:
        print("  Inverting covariance matrix... (may take ~30s)")
    dist_mah = compute_dist_mahalanobis(q_feats, g_feats, train_feats,
                                        use_diagonal=use_diagonal)
    res_mah = eval_with_dist(dist_mah, q_labels, g_labels, q_cams, g_cams)
    print(f"  Mahal    Rank-1={res_mah['Rank-1']:.2f}  Rank-5={res_mah['Rank-5']:.2f}  mAP={res_mah['mAP']:.2f}")

    delta = {k: res_mah[k] - res_l2[k] for k in res_l2}
    print(f"  Delta    Rank-1={delta['Rank-1']:+.2f}  Rank-5={delta['Rank-5']:+.2f}  mAP={delta['mAP']:+.2f}")

    return {'l2': res_l2, 'mahalanobis': res_mah, 'delta': delta,
            'cov_type': cov_type, 'D': int(train_feats.shape[1])}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet_ckpt', default='checkpoints/resnet50_best.pth')
    parser.add_argument('--vit_ckpt',    default='checkpoints/vit_best.pth')
    parser.add_argument('--data_dir',    default='data/VeRi')
    parser.add_argument('--out',         default='results/mahalanobis_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    if os.path.exists(args.resnet_ckpt):
        results['resnet50'] = evaluate_model(args.resnet_ckpt, 'resnet50',
                                             args.data_dir, device)
    else:
        print(f"ResNet checkpoint not found: {args.resnet_ckpt}")

    if os.path.exists(args.vit_ckpt):
        results['vit_small'] = evaluate_model(args.vit_ckpt, 'vit_small',
                                              args.data_dir, device)
    else:
        print(f"ViT checkpoint not found: {args.vit_ckpt}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {args.out}")

    # Summary table
    print("\n" + "="*65)
    print(f"{'Model':<14} {'Metric':<8} {'L2':>8} {'Mahal':>8} {'Delta':>8}")
    print("="*65)
    for model_name, r in results.items():
        for metric in ['Rank-1', 'Rank-5', 'mAP']:
            print(f"{model_name:<14} {metric:<8} "
                  f"{r['l2'][metric]:>8.2f} "
                  f"{r['mahalanobis'][metric]:>8.2f} "
                  f"{r['delta'][metric]:>+8.2f}")
        print("-"*65)
