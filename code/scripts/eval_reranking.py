"""
Evaluation script: extract embeddings from checkpoint, compute metrics with/without re-ranking.

Usage:
    python scripts/eval_reranking.py --model_type resnet50 \
        --checkpoint checkpoints/resnet50_best.pth \
        --data_dir data/VeRi-776 \
        --output results/results_resnet50_rr.json

    python scripts/eval_reranking.py --model_type vit \
        --checkpoint checkpoints/vit_best.pth \
        --data_dir data/VeRi-776 \
        --output results/results_vit_rr.json
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_reid.dataset import VeRiDataset
from vehicle_reid.evaluation import evaluate_reid, evaluate_with_reranking
from vehicle_reid.transforms import get_test_transforms


def build_model(model_type: str, num_classes: int = 576):
    if model_type == 'resnet50':
        from vehicle_reid.model import build_model as _build
        return _build(num_classes=num_classes)
    elif model_type == 'vit':
        from vehicle_reid.vit_model import build_vit_model
        return build_vit_model(num_classes=num_classes, model_size='small', img_size=224)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features, labels, cameras = [], [], []

    for batch in tqdm(loader, desc="Extracting"):
        imgs, pids, cids = batch['image'], batch['label'], batch['camera_id']
        imgs = imgs.to(device)
        feats = model(imgs)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = feats / feats.norm(dim=1, keepdim=True)
        features.append(feats.cpu().numpy())
        labels.append(pids.numpy())
        cameras.append(cids.numpy())

    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(cameras, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['resnet50', 'vit'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--k1', type=int, default=20)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--lambda_value', type=float, default=0.3)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint and detect num_classes
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    # Detect num_classes from classifier weight shape
    for key in state:
        if 'classifier.weight' in key:
            num_classes = state[key].shape[0]
            print(f"Detected num_classes={num_classes} from checkpoint")
            break
    else:
        num_classes = 576

    model = build_model(args.model_type, num_classes=num_classes)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    img_size = (224, 224) if args.model_type == 'vit' else (256, 128)
    val_transforms = get_test_transforms(img_size=img_size)

    query_dataset = VeRiDataset(
        root=args.data_dir, split='query', transform=val_transforms
    )
    gallery_dataset = VeRiDataset(
        root=args.data_dir, split='gallery', transform=val_transforms
    )

    query_loader = DataLoader(query_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    print(f"Query: {len(query_dataset)} | Gallery: {len(gallery_dataset)}")

    q_feat, q_labels, q_cams = extract_features(model, query_loader, device)
    g_feat, g_labels, g_cams = extract_features(model, gallery_loader, device)

    print("\n--- Without Re-ranking ---")
    results = evaluate_reid(q_feat, g_feat, q_labels, g_labels, q_cams, g_cams)
    print(f"  Rank-1: {results['Rank-1']:.2f}%")
    print(f"  Rank-5: {results['Rank-5']:.2f}%")
    print(f"  mAP:    {results['mAP']:.2f}%")

    print("\n--- With Re-ranking (k1={}, k2={}, lambda={}) ---".format(
        args.k1, args.k2, args.lambda_value))
    results_rr = evaluate_with_reranking(
        q_feat, g_feat, q_labels, g_labels, q_cams, g_cams,
        k1=args.k1, k2=args.k2, lambda_value=args.lambda_value
    )
    print(f"  Rank-1: {results_rr['Rank-1']:.2f}%")
    print(f"  Rank-5: {results_rr['Rank-5']:.2f}%")
    print(f"  mAP:    {results_rr['mAP']:.2f}%")

    print(f"\n  Delta mAP:    +{results_rr['mAP'] - results['mAP']:.2f}%")
    print(f"  Delta Rank-1: +{results_rr['Rank-1'] - results['Rank-1']:.2f}%")

    output = {
        'model': args.model_type,
        'checkpoint': args.checkpoint,
        'rerank_params': {'k1': args.k1, 'k2': args.k2, 'lambda': args.lambda_value},
        'without_reranking': results,
        'with_reranking': results_rr,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # Convert numpy floats to python floats for JSON serialization
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        if hasattr(obj, 'item'):
            return obj.item()
        return obj
    with open(args.output, 'w') as f:
        json.dump(to_python(output), f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
