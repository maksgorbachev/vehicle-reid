"""
Per-identity cluster analysis for Vehicle ReID.
Extracts gallery embeddings from both checkpoints, computes PCA 2D projections,
measures intra-class variance vs inter-class distance for ResNet-50 and ViT-Small.
"""

import os
import sys
import re
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(checkpoint_path, data_dir, model_type, split='query', batch_size=64):
    """Load checkpoint, extract L2-normalized embeddings for query or test split."""
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{model_type}] Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)

    # Infer num_classes from checkpoint to avoid size mismatch
    if model_type == 'resnet50':
        from vehicle_reid.model import VehicleReIDModel
        # classifier weight shape: [num_classes, feat_dim]
        num_classes = state.get(
            'bnneck.classifier.weight',
            state.get('classifier.weight', None)
        )
        num_classes = num_classes.shape[0] if num_classes is not None else 576
        model = VehicleReIDModel(num_classes=num_classes, pretrained=False)
        img_size = (128, 256)
    else:
        from vehicle_reid.vit_model import ViTReIDModel
        num_classes = state.get(
            'bnneck.classifier.weight',
            state.get('classifier.weight', None)
        )
        num_classes = num_classes.shape[0] if num_classes is not None else 576
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
            batch_paths = img_files[start:start + batch_size]
            imgs = []
            for p in batch_paths:
                imgs.append(transform(Image.open(p).convert('RGB')))
            imgs = torch.stack(imgs).to(device)
            feats = model(imgs)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats / feats.norm(dim=1, keepdim=True)
            all_features.append(feats.cpu().numpy())
            for p in batch_paths:
                m = pattern.match(os.path.basename(p))
                all_labels.append(int(m.group(1)))
                all_cameras.append(int(m.group(2)))
            if (start // batch_size) % 10 == 0:
                print(f"  {start + len(batch_paths)}/{len(img_files)}", end='\r')

    print()
    return (
        np.concatenate(all_features, axis=0),
        np.array(all_labels),
        np.array(all_cameras),
        img_files,
    )


# ---------------------------------------------------------------------------
# Cluster metrics
# ---------------------------------------------------------------------------

def compute_cluster_metrics(features, labels):
    """
    For each identity, compute intra-class variance.
    Also compute mean inter-class centroid distance.

    Returns:
        intra_vars: dict {id: mean pairwise L2^2 within class}
        inter_dist: scalar mean distance between class centroids
        centroids:  dict {id: centroid vector}
    """
    unique_ids = np.unique(labels)
    centroids = {}
    intra_vars = {}

    for uid in unique_ids:
        mask = labels == uid
        vecs = features[mask]
        c = vecs.mean(axis=0)
        centroids[uid] = c
        # Mean squared L2 distance to centroid = intra-class variance
        diffs = vecs - c
        intra_vars[uid] = float(np.mean(np.sum(diffs ** 2, axis=1)))

    # Mean inter-class centroid distance
    centroid_mat = np.stack(list(centroids.values()))
    # Pairwise L2 distances between centroids
    n = len(centroid_mat)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(centroid_mat[i] - centroid_mat[j]))
    inter_dist = float(np.mean(dists)) if dists else 0.0

    return intra_vars, inter_dist, centroids


def silhouette_score_fast(features, labels, max_samples=2000):
    """Approximate silhouette score using random subset."""
    from sklearn.metrics import silhouette_score
    n = len(features)
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
    # Need at least 2 classes
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(features, labels, metric='cosine', sample_size=None))


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_pca_clusters(features, labels, cameras, model_name, output_path,
                      num_ids=20, seed=42):
    """
    PCA 2D scatter of embeddings, colored by vehicle ID.
    Shows top num_ids most frequent IDs.
    """
    rng = np.random.RandomState(seed)
    unique_ids, counts = np.unique(labels, return_counts=True)
    # Pick IDs with most images for cleaner visualization
    top_ids = unique_ids[np.argsort(-counts)[:num_ids]]

    mask = np.isin(labels, top_ids)
    feats_sel = features[mask]
    labels_sel = labels[mask]
    cams_sel = cameras[mask]

    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(feats_sel)
    var_exp = pca.explained_variance_ratio_ * 100

    cmap = plt.cm.tab20
    id_color = {uid: cmap(i / max(num_ids - 1, 1)) for i, uid in enumerate(top_ids)}

    fig, ax = plt.subplots(figsize=(11, 8))

    for uid in top_ids:
        m = labels_sel == uid
        ax.scatter(
            xy[m, 0], xy[m, 1],
            c=[id_color[uid]],
            label=f'ID {uid:04d}',
            alpha=0.7, s=25, edgecolors='none'
        )
        # Mark centroid
        cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
        ax.scatter(cx, cy, c=[id_color[uid]], marker='*', s=180,
                   edgecolors='black', linewidths=0.8, zorder=5)

    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var.)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var.)', fontsize=11)
    ax.set_title(
        f'PCA-проекция эмбеддингов {model_name} (VeRi-776)\n'
        f'Показаны {num_ids} наиболее представленных ТС. '
        f'Звёздочка — центроид кластера.',
        fontsize=11
    )
    ax.legend(loc='upper right', fontsize=6.5, ncol=2,
              markerscale=1.2, framealpha=0.85)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_intra_variance_comparison(metrics_resnet, metrics_vit, output_path):
    """
    Side-by-side boxplot: intra-class variance distribution for each model.
    """
    vars_resnet = list(metrics_resnet['intra_vars'].values())
    vars_vit = list(metrics_vit['intra_vars'].values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: boxplot comparison
    ax = axes[0]
    bp = ax.boxplot(
        [vars_resnet, vars_vit],
        labels=['ResNet-50', 'ViT-Small'],
        patch_artist=True,
        notch=False,
        widths=0.5
    )
    colors = ['#4878D0', '#EE854A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Внутриклассовая дисперсия (L2²)', fontsize=11)
    ax.set_title('Распределение внутриклассовой дисперсии\nпо идентичностям ТС', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: scatter — intra variance per ID (ResNet vs ViT)
    ax2 = axes[1]
    # Align by common IDs
    common_ids = sorted(
        set(metrics_resnet['intra_vars'].keys()) & set(metrics_vit['intra_vars'].keys())
    )
    x = [metrics_resnet['intra_vars'][i] for i in common_ids]
    y = [metrics_vit['intra_vars'][i] for i in common_ids]

    ax2.scatter(x, y, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    lim = max(max(x), max(y)) * 1.05
    ax2.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='y=x')
    ax2.set_xlabel('Дисперсия ResNet-50', fontsize=11)
    ax2.set_ylabel('Дисперсия ViT-Small', fontsize=11)
    ax2.set_title('Внутриклассовая дисперсия: ResNet-50 vs ViT-Small\n'
                  'Точки ниже диагонали: ViT компактнее', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # Annotate fractions
    below = sum(1 for xi, yi in zip(x, y) if yi < xi)
    frac = below / len(common_ids) * 100
    ax2.text(0.05, 0.93, f'ViT компактнее в {frac:.0f}% ID',
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle('Компактность кластеров в пространстве эмбеддингов', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_table(metrics_resnet, metrics_vit):
    """Print LaTeX-ready summary table."""
    vars_r = list(metrics_resnet['intra_vars'].values())
    vars_v = list(metrics_vit['intra_vars'].values())

    print("\n=== Cluster Compactness Summary ===")
    print(f"{'Metric':<40} {'ResNet-50':>12} {'ViT-Small':>12}")
    print("-" * 66)
    print(f"{'Mean intra-class variance':<40} {np.mean(vars_r):>12.4f} {np.mean(vars_v):>12.4f}")
    print(f"{'Median intra-class variance':<40} {np.median(vars_r):>12.4f} {np.median(vars_v):>12.4f}")
    print(f"{'Mean inter-centroid distance':<40} {metrics_resnet['inter_dist']:>12.4f} {metrics_vit['inter_dist']:>12.4f}")
    print(f"{'Silhouette score':<40} {metrics_resnet['silhouette']:>12.4f} {metrics_vit['silhouette']:>12.4f}")

    ratio_r = metrics_resnet['inter_dist'] / (np.mean(vars_r) ** 0.5 + 1e-8)
    ratio_v = metrics_vit['inter_dist'] / (np.mean(vars_v) ** 0.5 + 1e-8)
    print(f"{'Inter/Intra ratio':<40} {ratio_r:>12.4f} {ratio_v:>12.4f}")

    print("\nLaTeX table row:")
    print(r"\hline")
    print(f"Внутрикл. дисперсия (ср.) & {np.mean(vars_r):.4f} & {np.mean(vars_v):.4f} \\\\")
    print(f"Межкл. расстояние & {metrics_resnet['inter_dist']:.4f} & {metrics_vit['inter_dist']:.4f} \\\\")
    print(f"Silhouette score & {metrics_resnet['silhouette']:.4f} & {metrics_vit['silhouette']:.4f} \\\\")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Per-identity cluster analysis')
    parser.add_argument('--data_dir', default='code/data/VeRi')
    parser.add_argument('--resnet_ckpt', default='code/checkpoints/resnet50_best.pth')
    parser.add_argument('--vit_ckpt', default='code/checkpoints/vit_best.pth')
    parser.add_argument('--output_dir', default='images')
    parser.add_argument('--split', default='query', choices=['query', 'test'],
                        help='Which split to use for embeddings')
    parser.add_argument('--num_ids', type=int, default=20,
                        help='Number of IDs to show in PCA plot')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Extracting ResNet-50 embeddings ===")
    feats_r, labels_r, cams_r, _ = extract_embeddings(
        args.resnet_ckpt, args.data_dir, 'resnet50',
        split=args.split, batch_size=args.batch_size
    )

    print("\n=== Extracting ViT-Small embeddings ===")
    feats_v, labels_v, cams_v, _ = extract_embeddings(
        args.vit_ckpt, args.data_dir, 'vit',
        split=args.split, batch_size=args.batch_size
    )

    print("\n=== Computing cluster metrics ===")
    intra_r, inter_r, centroids_r = compute_cluster_metrics(feats_r, labels_r)
    intra_v, inter_v, centroids_v = compute_cluster_metrics(feats_v, labels_v)

    print("Computing silhouette scores (may take ~30s)...")
    sil_r = silhouette_score_fast(feats_r, labels_r)
    sil_v = silhouette_score_fast(feats_v, labels_v)

    metrics_resnet = {'intra_vars': intra_r, 'inter_dist': inter_r, 'silhouette': sil_r}
    metrics_vit = {'intra_vars': intra_v, 'inter_dist': inter_v, 'silhouette': sil_v}

    print_summary_table(metrics_resnet, metrics_vit)

    print("\n=== Generating visualizations ===")
    plot_pca_clusters(
        feats_r, labels_r, cams_r,
        model_name='ResNet-50',
        output_path=os.path.join(args.output_dir, 'pca_clusters_resnet50.png'),
        num_ids=args.num_ids
    )
    plot_pca_clusters(
        feats_v, labels_v, cams_v,
        model_name='ViT-Small',
        output_path=os.path.join(args.output_dir, 'pca_clusters_vit.png'),
        num_ids=args.num_ids
    )
    plot_intra_variance_comparison(
        metrics_resnet, metrics_vit,
        output_path=os.path.join(args.output_dir, 'cluster_compactness.png')
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
