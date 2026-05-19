"""
C18 failure example.

Query images from camera C18 (worst camera, Rank-1 ~0-2% per pair-heatmap).
Find top-10 retrieval results for ResNet-50 vs ViT-Small.
Save side-by-side grid for thesis.

Usage:
    python scripts/c18_failure_example.py \
        --data_dir code/data/VeRi \
        --resnet_ckpt code/checkpoints/resnet50_best.pth \
        --vit_ckpt code/checkpoints/vit_best.pth \
        --output images/c18_failure_comparison.png
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_reid.dataset import VeRiDataset
from vehicle_reid.transforms import get_test_transforms


def build_model(model_type, num_classes):
    if model_type == 'resnet50':
        from vehicle_reid.model import build_model as _b
        return _b(num_classes=num_classes)
    from vehicle_reid.vit_model import build_vit_model
    return build_vit_model(num_classes=num_classes, model_size='small', img_size=224)


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, labels, cams, paths = [], [], [], []
    for batch in tqdm(loader, desc="extract"):
        x = batch['image'].to(device)
        f = model(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        f = f / f.norm(dim=1, keepdim=True)
        feats.append(f.cpu().numpy())
        labels.append(batch['label'].numpy())
        cams.append(batch['camera_id'].numpy())
        paths.extend(batch['path'])
    return (np.concatenate(feats), np.concatenate(labels),
            np.concatenate(cams), paths)


def load_and_extract(model_type, ckpt_path, data_dir, device, batch_size=64):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    num_classes = 576
    for k in state:
        if 'classifier.weight' in k:
            num_classes = state[k].shape[0]
            break

    model = build_model(model_type, num_classes).to(device)
    model.load_state_dict(state, strict=False)

    img_size = (224, 224) if model_type == 'vit' else (256, 128)
    tf = get_test_transforms(img_size=img_size)

    q_ds = VeRiDataset(root=data_dir, split='query', transform=tf)
    g_ds = VeRiDataset(root=data_dir, split='gallery', transform=tf)
    q_loader = DataLoader(q_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    g_loader = DataLoader(g_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[{model_type}] query={len(q_ds)} gallery={len(g_ds)}")
    qf, ql, qc, qp = extract(model, q_loader, device)
    gf, gl, gc, gp = extract(model, g_loader, device)
    return qf, ql, qc, qp, gf, gl, gc, gp


def pick_c18_query_with_most_failures(qf, ql, qc, qp, gf, gl, gc, top_k=10):
    """Pick query from C18 with most wrong matches in top-k (worst case)."""
    c18_idx = np.where(qc == 18)[0]
    worst_idx = -1
    worst_wrong = -1
    for qi in c18_idx:
        sims = gf @ qf[qi]
        order = np.argsort(-sims)
        # filter out same camera + same id (CMC convention)
        keep = ~((gc[order] == qc[qi]) & (gl[order] == ql[qi]))
        order = order[keep][:top_k]
        wrong = int((gl[order] != ql[qi]).sum())
        if wrong > worst_wrong:
            worst_wrong = wrong
            worst_idx = qi
    return worst_idx


def topk_indices(qf_i, ql_i, qc_i, gf, gl, gc, top_k=10):
    sims = gf @ qf_i
    order = np.argsort(-sims)
    keep = ~((gc[order] == qc_i) & (gl[order] == ql_i))
    return order[keep][:top_k]


def render(query_path, query_label, gallery_paths, gallery_labels,
           resnet_idx, vit_idx, save_path, top_k=10):
    fig, axes = plt.subplots(2, top_k + 1, figsize=(top_k * 1.6 + 2.5, 6))

    def show(ax, path, color, title):
        img = Image.open(path).convert('RGB')
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(3); s.set_visible(True)

    for row, (name, idx_list) in enumerate([('ResNet-50', resnet_idx), ('ViT-Small', vit_idx)]):
        show(axes[row, 0], query_path, 'blue', f'{name}\nQuery (C18)')

        correct_cnt = 0
        for j, gi in enumerate(idx_list):
            ok = gallery_labels[gi] == query_label
            correct_cnt += int(ok)
            color = 'green' if ok else 'red'
            show(axes[row, j + 1], gallery_paths[gi], color, f'#{j+1}')

        axes[row, 0].set_ylabel(f'{correct_cnt}/{top_k}', fontsize=11, rotation=0,
                                labelpad=25, va='center')

    plt.suptitle(f'Failure case: query from camera C18 (ID={query_label})\n'
                 f'Top-{top_k} retrieval. Green=correct, Red=wrong',
                 fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='code/data/VeRi')
    p.add_argument('--resnet_ckpt', default='code/checkpoints/resnet50_best.pth')
    p.add_argument('--vit_ckpt', default='code/checkpoints/vit_best.pth')
    p.add_argument('--output', default='images/c18_failure_comparison.png')
    p.add_argument('--top_k', type=int, default=10)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # ResNet pass
    r = load_and_extract('resnet50', args.resnet_ckpt, args.data_dir, device)
    rqf, rql, rqc, rqp, rgf, rgl, rgc, rgp = r

    # pick worst C18 query using ResNet (same query index used for ViT — query order identical)
    qi = pick_c18_query_with_most_failures(rqf, rql, rqc, rqp, rgf, rgl, rgc, args.top_k)
    print(f"picked query idx={qi} path={rqp[qi]} id={rql[qi]} cam={rqc[qi]}")

    resnet_top = topk_indices(rqf[qi], rql[qi], rqc[qi], rgf, rgl, rgc, args.top_k)

    # ViT pass
    v = load_and_extract('vit', args.vit_ckpt, args.data_dir, device)
    vqf, vql, vqc, vqp, vgf, vgl, vgc, vgp = v

    # sanity: query datasets are aligned (same order)
    assert vql[qi] == rql[qi] and vqc[qi] == rqc[qi], "query order mismatch"
    vit_top = topk_indices(vqf[qi], vql[qi], vqc[qi], vgf, vgl, vgc, args.top_k)

    # gallery order between models also identical → use rgp paths
    render(rqp[qi], rql[qi], rgp, rgl, resnet_top, vit_top, args.output, args.top_k)


if __name__ == '__main__':
    main()
