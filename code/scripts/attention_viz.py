"""
ViT Attention Map Visualization
================================
Visualizes where ViT-Small attends when embedding vehicle images.
Extracts CLS-token attention from the last transformer block.

Usage:
    # Demo (no dataset/checkpoint needed):
    python scripts/attention_viz.py --demo

    # Real checkpoint:
    python scripts/attention_viz.py \
        --checkpoint ../checkpoints/vit_best.pth \
        --data_dir   ../data/VeRi \
        --output_dir ../images
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Attention extraction
# --------------------------------------------------------------------------- #

def _patch_block_attn(block, storage_list):
    """
    Disable fused_attn on one transformer block and monkey-patch its forward
    to append the attention weight matrix to storage_list.
    Returns a restore callable.
    """
    attn_module = block.attn
    original_fused   = attn_module.fused_attn
    original_forward = attn_module.forward

    attn_module.fused_attn = False

    def patched_forward(x, attn_mask=None):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads, attn_module.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)

        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        storage_list.append(attn.detach().cpu())
        attn = attn_module.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, attn_module.attn_dim)
        x = attn_module.norm(x)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x

    attn_module.forward = patched_forward

    def restore():
        attn_module.fused_attn   = original_fused
        attn_module.forward      = original_forward

    return restore


def extract_cls_attention(vit_reid_model, images_tensor, device, last_k=4):
    """
    CLS-to-patch attention from the last `last_k` transformer blocks.
    Per layer: max over heads. Across layers: mean.

    Args:
        vit_reid_model: ViTReIDModel (eval mode)
        images_tensor:  [B, 3, 224, 224] on CPU
        last_k:         how many trailing blocks to use (default 4)

    Returns:
        attn_maps: np.ndarray [B, grid, grid]
    """
    import torch

    blocks = vit_reid_model.backbone.blocks
    target_blocks = list(blocks)[-last_k:]

    storages = []
    restores = []
    for block in target_blocks:
        layer_storage = []
        storages.append(layer_storage)
        restores.append(_patch_block_attn(block, layer_storage))

    vit_reid_model.eval()
    with torch.no_grad():
        _ = vit_reid_model(images_tensor.to(device))

    for r in restores:
        r()

    per_layer_maps = []
    for layer_storage in storages:
        if not layer_storage:
            continue
        A = layer_storage[0]              # [B, heads, N, N]
        cls_attn = A[:, :, 0, 1:]         # [B, heads, P]
        m = cls_attn.max(dim=1).values    # [B, P]  max over heads
        per_layer_maps.append(m)

    if not per_layer_maps:
        raise RuntimeError("No attention matrices captured — check model structure.")

    stacked = torch.stack(per_layer_maps, dim=1)   # [B, k, P]
    cls_map = stacked.mean(dim=1).numpy()          # [B, P]

    num_patches = cls_map.shape[1]
    grid = int(num_patches ** 0.5)
    return cls_map.reshape(-1, grid, grid)


# --------------------------------------------------------------------------- #
# Overlay rendering
# --------------------------------------------------------------------------- #

def _overlay_attention(pil_image, attn_map, alpha=0.55, colormap='jet'):
    """
    Overlay attention heatmap on PIL image.
    attn_map: [H, W] float array (0..1 or arbitrary positive)
    Returns PIL image (RGB).
    """
    img_np = np.array(pil_image.convert('RGB'), dtype=np.float32) / 255.0
    H, W, _ = img_np.shape

    # Percentile normalization: clip to [p5, p95] before min-max scaling.
    a = attn_map.astype(np.float32)
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    if hi - lo < 1e-8:
        a = a - a.min()
        a = a / (a.max() + 1e-8)
    else:
        a = np.clip(a, lo, hi)
        a = (a - lo) / (hi - lo + 1e-8)

    # Upsample attention to image size
    attn_pil = Image.fromarray((a * 255).astype(np.uint8))
    attn_resized = np.array(attn_pil.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(attn_resized)[:, :, :3]  # drop alpha

    blended = (1 - alpha) * img_np + alpha * heatmap
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Main visualization functions
# --------------------------------------------------------------------------- #

def visualize_attention_examples(
    vit_reid_model,
    image_paths,
    image_labels,
    device,
    num_examples=6,
    save_path='attention_maps.pdf',
    title='ViT-Small Attention Maps (последние 4 блока, max по головам, CLS-токен)',
):
    """
    Show num_examples images side-by-side: original | attention overlay.
    image_paths and image_labels must be aligned lists.
    """
    import torch
    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    chosen = np.random.choice(len(image_paths), min(num_examples, len(image_paths)),
                              replace=False)

    pil_images, tensors = [], []
    for idx in chosen:
        pil = Image.open(image_paths[idx]).convert('RGB')
        pil_images.append(pil)
        tensors.append(transform(pil))

    batch = torch.stack(tensors)
    attn_maps = extract_cls_attention(vit_reid_model, batch, device)  # [B, 14, 14]

    n = len(pil_images)
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 2.5))
    if n == 1:
        axes = [axes]

    for i, (pil, attn) in enumerate(zip(pil_images, attn_maps)):
        overlay = _overlay_attention(pil, attn)
        lid = image_labels[chosen[i]]

        axes[i][0].imshow(pil)
        axes[i][0].set_title(f'ID {lid} — original', fontsize=8)
        axes[i][0].axis('off')

        axes[i][1].imshow(overlay)
        axes[i][1].set_title(f'ID {lid} — attention', fontsize=8)
        axes[i][1].axis('off')

    plt.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {save_path}")


def visualize_attention_successes_and_failures(
    vit_reid_model,
    query_paths,
    query_labels,
    gallery_paths,
    gallery_labels,
    device,
    num_each=3,
    save_path='attention_success_fail.pdf',
):
    """
    Show attention maps for correct (Rank-1 hit) and incorrect (Rank-1 miss) queries.
    Layout: [query | attn] | [rank1_result | attn]  per row.
    """
    import torch
    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def get_rank1(q_idx):
        q_label = query_labels[q_idx]
        q_pil = Image.open(query_paths[q_idx]).convert('RGB')
        q_tensor = transform(q_pil).unsqueeze(0)

        with torch.no_grad():
            q_emb = vit_reid_model(q_tensor.to(device))  # [1, D]

        # Brute-force rank-1 over gallery (sample up to 500 for speed)
        gal_size = min(500, len(gallery_paths))
        gal_idx = np.random.choice(len(gallery_paths), gal_size, replace=False)
        gal_tensors = []
        for gi in gal_idx:
            gp = Image.open(gallery_paths[gi]).convert('RGB')
            gal_tensors.append(transform(gp))
        gal_batch = torch.stack(gal_tensors).to(device)
        with torch.no_grad():
            gal_embs = vit_reid_model(gal_batch)  # [G, D]

        dists = ((gal_embs - q_emb) ** 2).sum(dim=1).cpu().numpy()
        best = int(np.argmin(dists))
        best_global = int(gal_idx[best])
        hit = gallery_labels[best_global] == q_label
        return best_global, hit

    successes, failures = [], []
    for qi in range(len(query_paths)):
        if len(successes) >= num_each and len(failures) >= num_each:
            break
        best_gi, hit = get_rank1(qi)
        if hit and len(successes) < num_each:
            successes.append((qi, best_gi))
        elif not hit and len(failures) < num_each:
            failures.append((qi, best_gi))

    rows = successes[:num_each] + failures[:num_each]
    labels_row = ['Верно'] * len(successes[:num_each]) + ['Ошибка'] * len(failures[:num_each])

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(10, n * 2.8))
    if n == 1:
        axes = [axes]

    for i, ((qi, gi), verdict) in enumerate(zip(rows, labels_row)):
        q_pil  = Image.open(query_paths[qi]).convert('RGB')
        g_pil  = Image.open(gallery_paths[gi]).convert('RGB')

        q_tensor = transform(q_pil).unsqueeze(0)
        g_tensor = transform(g_pil).unsqueeze(0)

        q_attn = extract_cls_attention(vit_reid_model, q_tensor, device)[0]
        g_attn = extract_cls_attention(vit_reid_model, g_tensor, device)[0]

        q_overlay = _overlay_attention(q_pil, q_attn)
        g_overlay = _overlay_attention(g_pil, g_attn)

        color = 'green' if verdict == 'Верно' else 'red'
        ql = query_labels[qi]
        gl = gallery_labels[gi]

        axes[i][0].imshow(q_pil);    axes[i][0].set_title(f'Query  ID {ql}', fontsize=8)
        axes[i][1].imshow(q_overlay); axes[i][1].set_title('attention', fontsize=8)
        axes[i][2].imshow(g_pil);    axes[i][2].set_title(f'Rank-1 ID {gl}', fontsize=8,
                                                            color=color)
        axes[i][3].imshow(g_overlay); axes[i][3].set_title(f'attention ({verdict})',
                                                             fontsize=8, color=color)
        for ax in axes[i]:
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
                spine.set_visible(True)

    plt.suptitle('ViT-Small Attention: успешные и ошибочные совпадения', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {save_path}")


# --------------------------------------------------------------------------- #
# Demo mode
# --------------------------------------------------------------------------- #

def run_demo(output_dir):
    """Generate demo attention visualizations using synthetic colored rectangles."""
    import torch
    import torch.nn as nn

    try:
        import timm
    except ImportError:
        print("timm not installed. Run: pip install timm")
        return

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vehicle_reid.vit_model import build_vit_model

    os.makedirs(output_dir, exist_ok=True)
    print("Demo mode: loading pretrained ViT-Small (ImageNet weights) ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_vit_model(num_classes=576, model_size='small', pretrained=True)
    model.to(device).eval()

    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create 6 synthetic "vehicle" images with colored rectangles
    demo_imgs = []
    for seed in range(6):
        np.random.seed(seed * 7)
        arr = np.ones((128, 256, 3), dtype=np.uint8) * 200
        col = (np.random.randint(30, 220),
               np.random.randint(30, 220),
               np.random.randint(30, 220))
        arr[20:100, 30:220] = col
        arr[10:30, 60:190] = (min(col[0]+40, 255), min(col[1]+40, 255), min(col[2]+40, 255))
        # wheels
        for wx in [60, 180]:
            arr[90:110, wx-15:wx+15] = (30, 30, 30)
        pil = Image.fromarray(arr)
        demo_imgs.append(pil)

    tensors = torch.stack([transform(p) for p in demo_imgs])
    attn_maps = extract_cls_attention(model, tensors, device)  # [6, 14, 14]

    n = len(demo_imgs)
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 2.5))
    for i, (pil, attn) in enumerate(zip(demo_imgs, attn_maps)):
        overlay = _overlay_attention(pil, attn)
        axes[i][0].imshow(pil);    axes[i][0].set_title('Изображение', fontsize=8);  axes[i][0].axis('off')
        axes[i][1].imshow(overlay); axes[i][1].set_title('Attention map', fontsize=8); axes[i][1].axis('off')

    plt.suptitle('ViT-Small Attention Maps (demo, pretrained ImageNet)', fontsize=11)
    plt.tight_layout()
    out = os.path.join(output_dir, 'attention_maps_demo.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out}")


# --------------------------------------------------------------------------- #
# Real checkpoint mode
# --------------------------------------------------------------------------- #

def run_from_checkpoint(checkpoint_path, data_dir, output_dir, num_examples=6):
    import torch
    from vehicle_reid.vit_model import build_vit_model

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    # detect num_classes from checkpoint to avoid size mismatch
    num_classes = state['bnneck.classifier.weight'].shape[0]
    model = build_vit_model(num_classes=num_classes, model_size='small', pretrained=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print("Checkpoint loaded.")

    def _list_images(folder):
        exts = {'.jpg', '.jpeg', '.png'}
        return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in exts])

    def _parse_id(path):
        return int(os.path.basename(path).split('_')[0])

    query_paths   = _list_images(os.path.join(data_dir, 'image_query'))
    gallery_paths = _list_images(os.path.join(data_dir, 'image_test'))
    query_labels   = [_parse_id(p) for p in query_paths]
    gallery_labels = [_parse_id(p) for p in gallery_paths]

    # Simple random examples
    visualize_attention_examples(
        model, query_paths, query_labels, device,
        num_examples=num_examples,
        save_path=os.path.join(output_dir, 'attention_maps_vit.pdf'),
        title='ViT-Small Attention Maps — VeRi-776 query set',
    )

    # Success vs failure
    visualize_attention_successes_and_failures(
        model,
        query_paths[:200], query_labels[:200],
        gallery_paths, gallery_labels,
        device,
        num_each=3,
        save_path=os.path.join(output_dir, 'attention_success_fail_vit.pdf'),
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT Attention Map Visualization')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode — no checkpoint or dataset needed')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to vit_best.pth checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data/VeRi',
                        help='VeRi dataset root (must have image_query/ and image_test/)')
    parser.add_argument('--output_dir', type=str, default='../images',
                        help='Where to save PDF figures')
    parser.add_argument('--num_examples', type=int, default=6,
                        help='Number of examples to visualize')
    args = parser.parse_args()

    if args.demo:
        run_demo(args.output_dir)
    elif args.checkpoint:
        run_from_checkpoint(args.checkpoint, args.data_dir,
                            args.output_dir, args.num_examples)
    else:
        print("Specify --demo or --checkpoint <path>. See --help.")
