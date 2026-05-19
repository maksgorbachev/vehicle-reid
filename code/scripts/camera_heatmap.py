"""
Per-camera-pair Rank-1 accuracy heatmap for VeRi-776.
For each (query_camera, gallery_camera) pair: computes Rank-1 accuracy
restricted to images from those two cameras only.
"""

import os
import sys
import re
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Embedding extraction (shared with cluster_analysis.py)
# ---------------------------------------------------------------------------

def extract_embeddings(checkpoint_path, data_dir, model_type, split, batch_size=64):
    """Extract L2-normalized embeddings + labels + cameras for a dataset split."""
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)

    key = 'bnneck.classifier.weight'
    num_classes = state[key].shape[0] if key in state else 576

    if model_type == 'resnet50':
        from vehicle_reid.model import VehicleReIDModel
        model = VehicleReIDModel(num_classes=num_classes, pretrained=False)
        img_size = (128, 256)
    else:
        from vehicle_reid.vit_model import ViTReIDModel
        model = ViTReIDModel(num_classes=num_classes, pretrained=False)
        img_size = (224, 224)

    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_dir = os.path.join(data_dir, f'image_{split}')
    pattern = re.compile(r'^(\d+)_c(\d+)_')
    img_files = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    all_features, all_labels, all_cameras = [], [], []

    import torch
    with torch.no_grad():
        for start in range(0, len(img_files), batch_size):
            batch = img_files[start:start + batch_size]
            imgs = torch.stack([
                transform(Image.open(p).convert('RGB')) for p in batch
            ]).to(device)
            feats = model(imgs)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats / feats.norm(dim=1, keepdim=True)
            all_features.append(feats.cpu().numpy())
            for p in batch:
                m = pattern.match(os.path.basename(p))
                all_labels.append(int(m.group(1)))
                all_cameras.append(int(m.group(2)))
            if (start // batch_size) % 10 == 0:
                print(f"  {start + len(batch)}/{len(img_files)}", end='\r')

    print()
    return (
        np.concatenate(all_features),
        np.array(all_labels),
        np.array(all_cameras),
    )


# ---------------------------------------------------------------------------
# Per-camera-pair Rank-1
# ---------------------------------------------------------------------------

def compute_camera_pair_rank1(q_feats, q_labels, q_cams,
                               g_feats, g_labels, g_cams):
    """
    For each (query_cam, gallery_cam) pair:
      - Restrict queries to q_cam, gallery to g_cam
      - Compute cosine similarity, find nearest gallery image
      - Rank-1 = fraction where nearest has same vehicle ID
      - Returns 20x20 matrix (NaN where no query/gallery exist)
    """
    all_cams = sorted(set(q_cams.tolist()) | set(g_cams.tolist()))
    cam_to_idx = {c: i for i, c in enumerate(all_cams)}
    n = len(all_cams)

    rank1_matrix = np.full((n, n), np.nan)
    count_matrix = np.zeros((n, n), dtype=int)

    for qc in all_cams:
        qi = cam_to_idx[qc]
        q_mask = q_cams == qc
        if not q_mask.any():
            continue
        qf = q_feats[q_mask]
        ql = q_labels[q_mask]

        for gc in all_cams:
            if qc == gc:
                continue  # skip same-camera (no-same-camera protocol)
            gi = cam_to_idx[gc]
            g_mask = g_cams == gc
            if not g_mask.any():
                continue
            gf = g_feats[g_mask]
            gl = g_labels[g_mask]

            # Cosine similarity → nearest neighbor
            sim = qf @ gf.T          # [Q_qc, G_gc]
            nn_idx = np.argmax(sim, axis=1)
            correct = (gl[nn_idx] == ql)
            rank1_matrix[qi, gi] = correct.mean() * 100
            count_matrix[qi, gi] = len(ql)

    return rank1_matrix, count_matrix, all_cams


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_heatmap(matrix, cam_labels, title, output_path, vmin=0, vmax=100):
    """Plot Rank-1 heatmap with camera labels."""
    n = len(cam_labels)
    labels = [f'C{c:02d}' for c in cam_labels]

    fig, ax = plt.subplots(figsize=(11, 9))
    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    im = ax.imshow(masked, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Камера галереи', fontsize=11)
    ax.set_ylabel('Камера запроса', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Annotate cells with value
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 30 or val > 80 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=6.5, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Rank-1 (%)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_side_by_side(mat_r, mat_v, cam_labels, output_path):
    """Side-by-side heatmap: ResNet-50 vs ViT-Small + difference."""
    n = len(cam_labels)
    labels = [f'C{c:02d}' for c in cam_labels]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    titles = ['ResNet-50', 'ViT-Small', 'ViT - ResNet (разница)']
    matrices = [mat_r, mat_v, mat_v - mat_r]
    cmaps = ['RdYlGn', 'RdYlGn', 'RdBu']
    vranges = [(0, 100), (0, 100), (-30, 30)]

    for ax, mat, title, cmap, (vmin, vmax) in zip(
            axes, matrices, titles, cmaps, vranges):
        masked = np.ma.array(mat, mask=np.isnan(mat))
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Камера галереи', fontsize=10)
        ax.set_ylabel('Камера запроса', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 60 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                            fontsize=5.5, color=color)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('%', fontsize=9)

    plt.suptitle('Rank-1 по парам камер (запрос × галерея), VeRi-776',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_stats(mat_r, mat_v, cam_labels):
    """Print top easy/hard camera pairs."""
    flat_r = [(mat_r[i, j], cam_labels[i], cam_labels[j])
              for i in range(len(cam_labels))
              for j in range(len(cam_labels))
              if not np.isnan(mat_r[i, j])]
    flat_v = [(mat_v[i, j], cam_labels[i], cam_labels[j])
              for i in range(len(cam_labels))
              for j in range(len(cam_labels))
              if not np.isnan(mat_v[i, j])]

    print("\n=== Top-5 EASY pairs (ResNet-50) ===")
    for v, qc, gc in sorted(flat_r, reverse=True)[:5]:
        print(f"  C{qc:02d} -> C{gc:02d}: {v:.1f}%")

    print("\n=== Top-5 HARD pairs (ResNet-50) ===")
    for v, qc, gc in sorted(flat_r)[:5]:
        print(f"  C{qc:02d} -> C{gc:02d}: {v:.1f}%")

    print("\n=== Top-5 EASY pairs (ViT-Small) ===")
    for v, qc, gc in sorted(flat_v, reverse=True)[:5]:
        print(f"  C{qc:02d} -> C{gc:02d}: {v:.1f}%")

    print("\n=== Top-5 HARD pairs (ViT-Small) ===")
    for v, qc, gc in sorted(flat_v)[:5]:
        print(f"  C{qc:02d} -> C{gc:02d}: {v:.1f}%")

    vals_r = [v for v, _, _ in flat_r]
    vals_v = [v for v, _, _ in flat_v]
    diff = [v - r for (v, qc, gc), (r, _, __) in zip(
        sorted(flat_v, key=lambda x: (x[1], x[2])),
        sorted(flat_r, key=lambda x: (x[1], x[2]))
    )]
    print(f"\nMean Rank-1 ResNet-50: {np.mean(vals_r):.1f}%")
    print(f"Mean Rank-1 ViT-Small: {np.mean(vals_v):.1f}%")
    print(f"Pairs where ViT > ResNet: {sum(1 for d in diff if d > 0)}/{len(diff)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='code/data/VeRi')
    parser.add_argument('--resnet_ckpt', default='code/checkpoints/resnet50_best.pth')
    parser.add_argument('--vit_ckpt', default='code/checkpoints/vit_best.pth')
    parser.add_argument('--output_dir', default='images')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Extracting ResNet-50 embeddings (query) ===")
    qf_r, ql_r, qc_r = extract_embeddings(
        args.resnet_ckpt, args.data_dir, 'resnet50', 'query', args.batch_size)

    print("=== Extracting ResNet-50 embeddings (test/gallery) ===")
    gf_r, gl_r, gc_r = extract_embeddings(
        args.resnet_ckpt, args.data_dir, 'resnet50', 'test', args.batch_size)

    print("=== Extracting ViT-Small embeddings (query) ===")
    qf_v, ql_v, qc_v = extract_embeddings(
        args.vit_ckpt, args.data_dir, 'vit', 'query', args.batch_size)

    print("=== Extracting ViT-Small embeddings (test/gallery) ===")
    gf_v, gl_v, gc_v = extract_embeddings(
        args.vit_ckpt, args.data_dir, 'vit', 'test', args.batch_size)

    print("\n=== Computing per-camera-pair Rank-1 ===")
    mat_r, cnt_r, cams_r = compute_camera_pair_rank1(qf_r, ql_r, qc_r, gf_r, gl_r, gc_r)
    mat_v, cnt_v, cams_v = compute_camera_pair_rank1(qf_v, ql_v, qc_v, gf_v, gl_v, gc_v)

    print_stats(mat_r, mat_v, cams_r)

    print("\n=== Generating heatmaps ===")
    plot_heatmap(
        mat_r, cams_r,
        'Rank-1 по парам камер: ResNet-50 (VeRi-776)',
        os.path.join(args.output_dir, 'camera_heatmap_resnet50.png')
    )
    plot_heatmap(
        mat_v, cams_v,
        'Rank-1 по парам камер: ViT-Small (VeRi-776)',
        os.path.join(args.output_dir, 'camera_heatmap_vit.png')
    )
    plot_side_by_side(
        mat_r, mat_v, cams_r,
        os.path.join(args.output_dir, 'camera_heatmap_comparison.png')
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
