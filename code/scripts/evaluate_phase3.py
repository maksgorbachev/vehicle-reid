"""
Phase 3 Evaluation Script
=========================
Запускает оценку для ResNet-50, ResNet-101, ViT-Small с тремя метриками:
  - L2 / косинусное расстояние (baseline)
  - k-reciprocal Re-ranking
  - Расстояние Махаланобиса

Результаты сохраняются в code/results/results_phase3.json
и выводятся таблицей для добавления в Chapter3.tex.

Использование:
    python scripts/evaluate_phase3.py \\
        --data_dir data/VeRi-776 \\
        --resnet50_ckpt checkpoints/resnet50_best.pth \\
        --resnet101_ckpt checkpoints/resnet101_best.pth \\
        --vit_ckpt checkpoints/vit_best.pth
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_reid.model import build_model
from vehicle_reid.vit_model import build_vit_model
from vehicle_reid.dataset import VeRiDataset
from vehicle_reid.transforms import get_test_transforms
from vehicle_reid.evaluation import ReIDEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/VeRi-776')
    p.add_argument('--resnet50_ckpt', type=str, default='checkpoints/resnet50_best.pth')
    p.add_argument('--resnet101_ckpt', type=str, default='checkpoints/resnet101_best.pth')
    p.add_argument('--vit_ckpt', type=str, default='checkpoints/vit_best.pth')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output', type=str, default='results/results_phase3.json')
    p.add_argument('--skip_mahalanobis', action='store_true',
                   help='Skip Mahalanobis (slow on CPU, needs large RAM)')
    return p.parse_args()


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features_list, labels_list, cameras_list = [], [], []

    for batch in loader:
        imgs = batch['image'].to(device)
        labels = batch['label']
        cameras = batch['camera_id']
        feats = model(imgs)
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())
        cameras_list.append(cameras.numpy())

    return (
        np.concatenate(features_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(cameras_list, axis=0),
    )


def _load_state(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state:
        return state['model_state_dict']
    if 'state_dict' in state:
        return state['state_dict']
    return state


def _num_classes_from_state(state, key='bnneck.classifier.weight'):
    return state[key].shape[0]


def load_resnet(arch, ckpt_path, device):
    state = _load_state(ckpt_path, device)
    num_classes = _num_classes_from_state(state)
    model = build_model(num_classes=num_classes, arch=arch, pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def load_vit(ckpt_path, device):
    state = _load_state(ckpt_path, device)
    num_classes = _num_classes_from_state(state)
    model = build_vit_model(num_classes=num_classes, model_size='small', pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def evaluate_model(name, model, query_loader, gallery_loader, train_loader,
                   device, evaluator, skip_mahal=False):
    print(f"\n--- Extracting features: {name} ---")
    q_feat, q_lbl, q_cam = extract_features(model, query_loader, device)
    g_feat, g_lbl, g_cam = extract_features(model, gallery_loader, device)

    results = {}

    # Baseline L2
    r = evaluator.evaluate(q_feat, g_feat, q_lbl, g_lbl, q_cam, g_cam)
    results['L2'] = r
    print(f"  L2:   Rank-1={r['Rank-1']:.2f}%  Rank-5={r['Rank-5']:.2f}%  mAP={r['mAP']:.2f}%")

    # Re-ranking
    r_rr = evaluator.evaluate(q_feat, g_feat, q_lbl, g_lbl, q_cam, g_cam,
                               reranking=True, k1=20, k2=6, lambda_value=0.3)
    results['RR'] = r_rr
    print(f"  +RR:  Rank-1={r_rr['Rank-1']:.2f}%  Rank-5={r_rr['Rank-5']:.2f}%  mAP={r_rr['mAP']:.2f}%")

    # Mahalanobis
    if not skip_mahal:
        print(f"  Computing Mahalanobis (may take a few minutes)...")
        tr_feat, _, _ = extract_features(model, train_loader, device)
        r_mah = evaluator.evaluate(q_feat, g_feat, q_lbl, g_lbl, q_cam, g_cam,
                                    mahalanobis=True, train_features=tr_feat)
        results['Mahalanobis'] = r_mah
        print(f"  Mah:  Rank-1={r_mah['Rank-1']:.2f}%  Rank-5={r_mah['Rank-5']:.2f}%  mAP={r_mah['mAP']:.2f}%")

    return results


def print_latex_table(all_results):
    """Выводит строки для вставки в Table 3.2 Chapter3.tex."""
    print("\n" + "="*70)
    print("LaTeX table rows (вставить в Chapter3.tex Table 3.2):")
    print("="*70)
    header = r"\hline Model & Rank-1 & Rank-5 & mAP & Rank-1+RR & mAP+RR \\"
    print(header)
    for name, res in all_results.items():
        r1 = res['L2']['Rank-1']
        r5 = res['L2']['Rank-5']
        mp = res['L2']['mAP']
        r1rr = res['RR']['Rank-1']
        mprr = res['RR']['mAP']
        print(f"{name} & {r1:.2f} & {r5:.2f} & {mp:.2f} & {r1rr:.2f} & {mprr:.2f} \\\\")
        if 'Mahalanobis' in res:
            mah = res['Mahalanobis']
            print(f"{name}+Mah & {mah['Rank-1']:.2f} & {mah['Rank-5']:.2f} & {mah['mAP']:.2f} & -- & -- \\\\")
    print(r"\hline")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Transforms
    test_tf = get_test_transforms(img_size=(256, 128))
    test_tf_vit = get_test_transforms(img_size=(224, 224))

    # Datasets
    train_ds = VeRiDataset(args.data_dir, split='train', transform=test_tf)
    query_ds = VeRiDataset(args.data_dir, split='query', transform=test_tf)
    gallery_ds = VeRiDataset(args.data_dir, split='test', transform=test_tf)
    query_ds_vit = VeRiDataset(args.data_dir, split='query', transform=test_tf_vit)
    gallery_ds_vit = VeRiDataset(args.data_dir, split='test', transform=test_tf_vit)
    train_ds_vit = VeRiDataset(args.data_dir, split='train', transform=test_tf_vit)

    num_classes = train_ds.num_classes
    print(f"Dataset: {num_classes} train IDs, {len(query_ds)} queries, {len(gallery_ds)} gallery")

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, shuffle=False)
    train_loader = DataLoader(train_ds, **kw)
    query_loader = DataLoader(query_ds, **kw)
    gallery_loader = DataLoader(gallery_ds, **kw)
    query_loader_vit = DataLoader(query_ds_vit, **kw)
    gallery_loader_vit = DataLoader(gallery_ds_vit, **kw)
    train_loader_vit = DataLoader(train_ds_vit, **kw)

    evaluator = ReIDEvaluator(max_rank=50, remove_same_camera=True)
    all_results = {}

    # ResNet-50
    if os.path.exists(args.resnet50_ckpt):
        model = load_resnet('resnet50', args.resnet50_ckpt, device)
        all_results['ResNet-50'] = evaluate_model(
            'ResNet-50', model, query_loader, gallery_loader, train_loader,
            device, evaluator, skip_mahal=args.skip_mahalanobis
        )
        del model
    else:
        print(f"[SKIP] ResNet-50 checkpoint not found: {args.resnet50_ckpt}")

    # ResNet-101
    if os.path.exists(args.resnet101_ckpt):
        model = load_resnet('resnet101', args.resnet101_ckpt, device)
        all_results['ResNet-101'] = evaluate_model(
            'ResNet-101', model, query_loader, gallery_loader, train_loader,
            device, evaluator, skip_mahal=args.skip_mahalanobis
        )
        del model
    else:
        print(f"[SKIP] ResNet-101 checkpoint not found: {args.resnet101_ckpt}")

    # ViT-Small
    if os.path.exists(args.vit_ckpt):
        model = load_vit(args.vit_ckpt, device)
        # ViT использует 224x224
        all_results['ViT-Small'] = evaluate_model(
            'ViT-Small', model, query_loader_vit, gallery_loader_vit, train_loader_vit,
            device, evaluator, skip_mahal=args.skip_mahalanobis
        )
        del model
    else:
        print(f"[SKIP] ViT checkpoint not found: {args.vit_ckpt}")

    # Сохраняем
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    def to_py(obj):
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_py(x) for x in obj]
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    with open(args.output, 'w') as f:
        json.dump(to_py(all_results), f, indent=2)
    print(f"\nResults saved to {args.output}")

    print_latex_table(all_results)


if __name__ == '__main__':
    main()
