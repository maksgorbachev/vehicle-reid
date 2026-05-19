"""
Visualization Scripts for Vehicle ReID
=======================================
Визуализации для анализа результатов:
- t-SNE/UMAP проекции эмбеддингов
- Top-k retrieval grid
- Training curves
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from PIL import Image

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_tsne_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    cameras: np.ndarray = None,
    num_samples: int = 2000,
    save_path: str = 'tsne_embeddings.pdf',
    title: str = 't-SNE Visualization of Vehicle Embeddings',
    perplexity: int = 30,
    figsize: tuple = (12, 10)
):
    """
    Построение t-SNE визуализации эмбеддингов.

    Args:
        features: [N, D] эмбеддинги
        labels: [N] метки идентичностей
        cameras: [N] ID камер (опционально)
        num_samples: количество сэмплов для визуализации
        save_path: путь для сохранения
        title: заголовок графика
    """
    print(f"Computing t-SNE for {min(num_samples, len(features))} samples...")

    # Сэмплирование при необходимости
    if len(features) > num_samples:
        indices = np.random.choice(len(features), num_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
        if cameras is not None:
            cameras = cameras[indices]

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(features)

    # Уникальные идентичности
    unique_labels = np.unique(labels)
    num_ids = min(20, len(unique_labels))  # Показываем до 20 ID
    selected_ids = np.random.choice(unique_labels, num_ids, replace=False)

    # Фильтруем для отображения
    mask = np.isin(labels, selected_ids)
    embeddings_2d = embeddings_2d[mask]
    labels_filtered = labels[mask]
    if cameras is not None:
        cameras_filtered = cameras[mask]

    # Цветовая палитра
    colors = plt.cm.tab20(np.linspace(0, 1, num_ids))
    label_to_color = {lid: colors[i] for i, lid in enumerate(selected_ids)}

    # Создаём фигуру
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    for lid in selected_ids:
        mask_id = labels_filtered == lid
        ax.scatter(
            embeddings_2d[mask_id, 0],
            embeddings_2d[mask_id, 1],
            c=[label_to_color[lid]],
            label=f'ID {lid}',
            alpha=0.7,
            s=30
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE visualization to {save_path}")


def plot_tsne_with_thumbnails(
    features: np.ndarray,
    labels: np.ndarray,
    image_paths: list,
    num_samples: int = 300,
    zoom: float = 0.18,
    save_path: str = 'tsne_thumbnails.pdf',
    title: str = 't-SNE с миниатюрами транспортных средств',
    perplexity: int = 30,
    figsize: tuple = (16, 14),
    border_width: int = 3,
):
    """
    t-SNE с реальными миниатюрами изображений (OffsetImage).

    Args:
        features:     [N, D] эмбеддинги
        labels:       [N] метки идентичностей (int)
        image_paths:  [N] пути к изображениям
        num_samples:  сколько точек показывать (300 — оптимум для читаемости)
        zoom:         масштаб миниатюр (0.15–0.25 для 300 изображений)
        save_path:    путь сохранения
        title:        заголовок
        perplexity:   параметр t-SNE
        figsize:      размер фигуры
        border_width: толщина рамки вокруг миниатюры (по цвету ID)
    """
    N = len(features)
    n = min(num_samples, N)
    idx = np.random.choice(N, n, replace=False)
    features_s = features[idx]
    labels_s   = labels[idx]
    paths_s    = [image_paths[i] for i in idx]

    print(f"t-SNE: {n} samples, perplexity={perplexity} ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    xy = tsne.fit_transform(features_s)

    unique_ids = np.unique(labels_s)
    cmap = plt.cm.tab20 if len(unique_ids) <= 20 else plt.cm.hsv
    id_color = {lid: cmap(i / max(len(unique_ids) - 1, 1))
                for i, lid in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=12)

    for i, (x, y, path, lid) in enumerate(zip(xy[:, 0], xy[:, 1], paths_s, labels_s)):
        try:
            img = np.array(Image.open(path).convert('RGB').resize((64, 32)))
        except Exception:
            # Цветной прямоугольник как заглушка (demo-режим)
            color_arr = np.full((32, 64, 3),
                                (np.array(id_color[lid][:3]) * 255).astype(np.uint8),
                                dtype=np.uint8)
            img = color_arr

        # Рамка по цвету идентичности
        c = (np.array(id_color[lid][:3]) * 255).astype(np.uint8)
        img[:border_width, :] = c
        img[-border_width:, :] = c
        img[:, :border_width] = c
        img[:, -border_width:] = c

        oi = OffsetImage(img, zoom=zoom)
        oi.image.axes = ax
        ab = AnnotationBbox(oi, (x, y), frameon=False, pad=0)
        ax.add_artist(ab)

    # Scatter-точки (невидимые) для масштаба осей
    ax.scatter(xy[:, 0], xy[:, 1], s=0, alpha=0)

    # Легенда по цветам ID
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=id_color[lid], label=f'ID {lid}')
               for lid in unique_ids[:20]]
    if len(unique_ids) > 20:
        handles.append(Patch(facecolor='white', label=f'... +{len(unique_ids)-20} IDs'))
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              ncol=2, framealpha=0.8, handlelength=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE thumbnails → {save_path}")


def plot_retrieval_grid(
    query_images: list,
    gallery_images: list,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    dist_matrix: np.ndarray,
    num_queries: int = 5,
    top_k: int = 10,
    save_path: str = 'retrieval_grid.pdf',
    img_size: tuple = (64, 32)
):
    """
    Визуализация top-k retrieval результатов.

    Args:
        query_images: список путей к изображениям запросов
        gallery_images: список путей к изображениям галереи
        query_labels: метки запросов
        gallery_labels: метки галереи
        dist_matrix: [Q, G] матрица расстояний
        num_queries: количество запросов для показа
        top_k: количество top-k результатов
        save_path: путь сохранения
    """
    # Выбираем случайные запросы
    query_indices = np.random.choice(len(query_labels), num_queries, replace=False)

    fig, axes = plt.subplots(num_queries, top_k + 1, figsize=(top_k * 1.5 + 2, num_queries * 2))

    for i, q_idx in enumerate(query_indices):
        # Запрос
        try:
            q_img = Image.open(query_images[q_idx]).resize(img_size)
            axes[i, 0].imshow(q_img)
        except:
            axes[i, 0].text(0.5, 0.5, 'Query', ha='center', va='center')
        axes[i, 0].set_title('Query', fontsize=8)
        axes[i, 0].axis('off')
        axes[i, 0].spines['bottom'].set_color('blue')
        axes[i, 0].spines['bottom'].set_linewidth(3)

        # Top-k результаты
        sorted_indices = np.argsort(dist_matrix[q_idx])

        for j, g_idx in enumerate(sorted_indices[:top_k]):
            is_correct = gallery_labels[g_idx] == query_labels[q_idx]

            try:
                g_img = Image.open(gallery_images[g_idx]).resize(img_size)
                axes[i, j + 1].imshow(g_img)
            except:
                axes[i, j + 1].text(0.5, 0.5, f'#{j+1}', ha='center', va='center')

            axes[i, j + 1].axis('off')

            # Обводка: зелёная если правильно, красная если ошибка
            color = 'green' if is_correct else 'red'
            for spine in axes[i, j + 1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)

            axes[i, j + 1].set_title(f'Rank {j+1}', fontsize=7)

    plt.suptitle('Vehicle Re-ID: Top-k Retrieval Results\n(Green=Correct, Red=Incorrect)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved retrieval grid to {save_path}")


def plot_training_curves(
    train_losses: list,
    eval_results: list,
    save_path: str = 'training_curves.pdf'
):
    """
    Построение графиков обучения.

    Args:
        train_losses: список словарей с потерями по эпохам
        eval_results: список словарей с результатами оценки
        save_path: путь сохранения
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(train_losses) + 1)

    # 1. Total Loss
    ax = axes[0, 0]
    losses = [l['loss'] for l in train_losses]
    ax.plot(epochs, losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)

    # 2. CE and Triplet Loss
    ax = axes[0, 1]
    ce_losses = [l['loss_ce'] for l in train_losses]
    triplet_losses = [l['loss_triplet'] for l in train_losses]
    ax.plot(epochs, ce_losses, 'g-', label='CE Loss', linewidth=2)
    ax.plot(epochs, triplet_losses, 'r-', label='Triplet Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. mAP
    ax = axes[1, 0]
    if eval_results:
        eval_epochs = [r['epoch'] for r in eval_results]
        mAP_no_rr = [r['no_rr']['mAP'] for r in eval_results]
        mAP_rr = [r['with_rr']['mAP'] for r in eval_results]

        ax.plot(eval_epochs, mAP_no_rr, 'b-o', label='Without Re-ranking', linewidth=2)
        ax.plot(eval_epochs, mAP_rr, 'r-s', label='With Re-ranking', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP (%)')
        ax.set_title('mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Rank-1
    ax = axes[1, 1]
    if eval_results:
        rank1_no_rr = [r['no_rr']['Rank-1'] for r in eval_results]
        rank1_rr = [r['with_rr']['Rank-1'] for r in eval_results]

        ax.plot(eval_epochs, rank1_no_rr, 'b-o', label='Without Re-ranking', linewidth=2)
        ax.plot(eval_epochs, rank1_rr, 'r-s', label='With Re-ranking', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rank-1 (%)')
        ax.set_title('Rank-1 Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Vehicle ReID Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_ablation_bar_chart(
    results: dict,
    save_path: str = 'ablation_study.pdf'
):
    """
    Столбчатая диаграмма для абляционного исследования.

    Args:
        results: словарь {config_name: {'mAP': float, 'Rank-1': float}}
        save_path: путь сохранения
    """
    configs = list(results.keys())
    mAPs = [results[c]['mAP'] for c in configs]
    rank1s = [results[c]['Rank-1'] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, mAPs, width, label='mAP', color='steelblue')
    bars2 = ax.bar(x + width/2, rank1s, width, label='Rank-1', color='coral')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ablation Study: Effect of Different Components')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation chart to {save_path}")


def plot_camera_distribution(
    features: np.ndarray,
    cameras: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'camera_distribution.pdf'
):
    """
    Визуализация влияния камер на распределение эмбеддингов.
    """
    # t-SNE
    print("Computing t-SNE for camera distribution...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)

    # Сэмплируем
    num_samples = min(2000, len(features))
    indices = np.random.choice(len(features), num_samples, replace=False)
    features_sampled = features[indices]
    cameras_sampled = cameras[indices]

    embeddings_2d = tsne.fit_transform(features_sampled)

    unique_cameras = np.unique(cameras_sampled)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_cameras)))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cam in enumerate(unique_cameras):
        mask = cameras_sampled == cam
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Camera {cam}',
            alpha=0.6,
            s=20
        )

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Embedding Distribution by Camera\n(Well-mixed = Good Domain Invariance)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved camera distribution to {save_path}")


# Генерация демо-визуализаций
def generate_demo_visualizations(output_dir: str = '../fig'):
    """Генерация демонстрационных визуализаций."""
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)

    # Симулируем данные
    num_samples = 1000
    num_ids = 50
    feature_dim = 256
    num_cameras = 6

    labels = np.random.randint(0, num_ids, num_samples)
    cameras = np.random.randint(0, num_cameras, num_samples)

    # Генерируем кластеризованные признаки
    id_centers = np.random.randn(num_ids, feature_dim)
    features = id_centers[labels] + np.random.randn(num_samples, feature_dim) * 0.3
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # t-SNE (dots)
    plot_tsne_embeddings(
        features, labels, cameras,
        save_path=os.path.join(output_dir, 'tsne_embeddings.pdf'),
        title='t-SNE Visualization of Vehicle Embeddings (VeRi-776)'
    )

    # t-SNE с миниатюрами (demo: image_paths=None → цветные заглушки)
    # В demo-режиме image_paths не существуют — передаём фиктивные пути,
    # функция поймает исключение PIL и нарисует цветной прямоугольник.
    fake_paths = [f'__fake__{i}' for i in range(num_samples)]
    plot_tsne_with_thumbnails(
        features, labels, fake_paths,
        num_samples=300,
        save_path=os.path.join(output_dir, 'tsne_thumbnails.pdf'),
        title='t-SNE с миниатюрами (demo)'
    )

    # Camera distribution
    plot_camera_distribution(
        features, cameras, labels,
        save_path=os.path.join(output_dir, 'camera_distribution.pdf')
    )

    # Training curves (симуляция)
    train_losses = []
    for epoch in range(120):
        loss = 5.0 * np.exp(-epoch / 30) + 0.5 + np.random.randn() * 0.1
        ce_loss = 3.0 * np.exp(-epoch / 25) + 0.3 + np.random.randn() * 0.05
        triplet_loss = 0.3 * np.exp(-epoch / 40) + 0.1 + np.random.randn() * 0.02
        train_losses.append({
            'loss': max(loss, 0.5),
            'loss_ce': max(ce_loss, 0.3),
            'loss_triplet': max(triplet_loss, 0.05)
        })

    eval_results = []
    for epoch in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        progress = epoch / 120
        mAP_base = 35 + 45 * (1 - np.exp(-3 * progress)) + np.random.randn() * 2
        rank1_base = 85 + 10 * (1 - np.exp(-3 * progress)) + np.random.randn() * 1

        eval_results.append({
            'epoch': epoch,
            'no_rr': {'mAP': mAP_base, 'Rank-1': rank1_base},
            'with_rr': {'mAP': mAP_base + 5, 'Rank-1': rank1_base + 2}
        })

    plot_training_curves(
        train_losses, eval_results,
        save_path=os.path.join(output_dir, 'training_curves.pdf')
    )

    # Ablation study
    ablation_results = {
        'ResNet50 + CE': {'mAP': 58.3, 'Rank-1': 89.2},
        '+ Triplet': {'mAP': 65.7, 'Rank-1': 92.1},
        '+ BNNeck': {'mAP': 72.4, 'Rank-1': 94.3},
        '+ Augmentations': {'mAP': 76.8, 'Rank-1': 95.1},
        '+ Re-ranking': {'mAP': 82.5, 'Rank-1': 95.8}
    }

    plot_ablation_bar_chart(
        ablation_results,
        save_path=os.path.join(output_dir, 'ablation_study.pdf')
    )

    print(f"\nAll visualizations saved to {output_dir}/")


def run_tsne_thumbnails_from_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    model_type: str = 'resnet50',
    output_dir: str = '../images',
    num_samples: int = 300,
    zoom: float = 0.18,
):
    """
    Загружает чекпоинт, извлекает эмбеддинги query-сета, строит t-SNE с миниатюрами.
    Использует существующие модули vehicle_reid.
    """
    import torch
    from vehicle_reid.dataset import VeRiDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Модель
    if model_type == 'resnet50':
        from vehicle_reid.model import VehicleReIDModel
        model = VehicleReIDModel(num_classes=576, pretrained=False)
        img_size = (128, 256)
    else:
        from vehicle_reid.vit_model import ViTReIDModel
        model = ViTReIDModel(num_classes=576, pretrained=False)
        img_size = (224, 224)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    query_dir = os.path.join(data_dir, 'image_query')
    img_files = sorted([
        os.path.join(query_dir, f)
        for f in os.listdir(query_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Парсим ID и камеру из имени файла: vehicleID_cameraID_...
    def parse_id(path):
        name = os.path.basename(path)
        parts = name.split('_')
        return int(parts[0]), int(parts[1][1:])

    all_features, all_labels, all_paths = [], [], []
    batch_size = 64

    with torch.no_grad():
        for start in range(0, len(img_files), batch_size):
            batch_paths = img_files[start:start + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                imgs.append(transform(img))
            imgs = torch.stack(imgs).to(device)
            feats = model(imgs)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats / feats.norm(dim=1, keepdim=True)
            all_features.append(feats.cpu().numpy())
            all_labels.extend([parse_id(p)[0] for p in batch_paths])
            all_paths.extend(batch_paths)
            if (start // batch_size) % 5 == 0:
                print(f"  {start + len(batch_paths)}/{len(img_files)}")

    features = np.concatenate(all_features, axis=0)
    labels   = np.array(all_labels)

    os.makedirs(output_dir, exist_ok=True)
    tag = model_type
    plot_tsne_with_thumbnails(
        features, labels, all_paths,
        num_samples=num_samples,
        zoom=zoom,
        save_path=os.path.join(output_dir, f'tsne_thumbnails_{tag}.pdf'),
        title=f't-SNE эмбеддингов {tag.upper()} с миниатюрами (VeRi-776 query)'
    )
    # Также dot-версия для сравнения
    plot_tsne_embeddings(
        features, labels,
        save_path=os.path.join(output_dir, f'tsne_embeddings_{tag}.pdf'),
        title=f't-SNE эмбеддингов {tag.upper()} (VeRi-776 query)'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../images',
                        help='Output directory for figures')
    parser.add_argument('--demo', action='store_true',
                        help='Generate demo visualizations (no dataset needed)')
    parser.add_argument('--tsne_thumbnails', action='store_true',
                        help='Run t-SNE with image thumbnails from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, default='../data/VeRi-776',
                        help='Path to VeRi-776 dataset')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['resnet50', 'vit'])
    parser.add_argument('--num_samples', type=int, default=300,
                        help='Number of t-SNE samples')
    parser.add_argument('--zoom', type=float, default=0.18,
                        help='Thumbnail zoom factor')
    args = parser.parse_args()

    if args.tsne_thumbnails and args.checkpoint:
        run_tsne_thumbnails_from_checkpoint(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            model_type=args.model_type,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            zoom=args.zoom,
        )
    else:
        generate_demo_visualizations(args.output_dir)
