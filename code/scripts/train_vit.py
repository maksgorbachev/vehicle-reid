"""
ViT-Small + BNNeck для Vehicle Re-ID
Обучение на VeRi-776 - локальная версия
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
import timm

# ================== КОНФИГУРАЦИЯ ==================
# Путь к датасету VeRi-776
DATA_DIR = r'C:\Users\nik\.cache\kagglehub\datasets\abhyudaya12\veri-vehicle-re-identification-dataset\versions\1\VeRi'

EPOCHS = 120
BATCH_SIZE = 48
P = 12  # Количество ID в батче
K = 4   # Количество изображений на ID
LR = 1e-4
WARMUP_EPOCHS = 10
MARGIN = 0.3
LABEL_SMOOTH = 0.1

VIT_SIZE = 'small'  # 'tiny', 'small', 'base'
IMG_SIZE = (224, 224)

SAVE_DIR = './checkpoints_vit'
os.makedirs(SAVE_DIR, exist_ok=True)

# ================== УСТРОЙСТВО ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================== МОДЕЛЬ ==================
class BNNeck(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_features, num_classes, bias=False)
        nn.init.normal_(self.bn.weight, 1.0, 0.02)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        bn_x = self.bn(x)
        logits = self.classifier(bn_x)
        return bn_x, logits


class ViTReID(nn.Module):
    def __init__(self, num_classes, model_size='small', pretrained=True):
        super().__init__()

        model_names = {
            'tiny': 'vit_tiny_patch16_224',
            'small': 'vit_small_patch16_224',
            'base': 'vit_base_patch16_224'
        }
        model_name = model_names[model_size]

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.embed_dim
        self.bnneck = BNNeck(self.embed_dim, num_classes)

        print(f'ViT: {model_name}, embed_dim={self.embed_dim}')

    def forward(self, x):
        features = self.backbone(x)
        bn_features, logits = self.bnneck(features)

        if self.training:
            return features, bn_features, logits
        else:
            return F.normalize(bn_features, p=2, dim=1)


# ================== ФУНКЦИИ ПОТЕРЬ ==================
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        dist_mat = torch.cdist(features, features, p=2)
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = ~mask_pos

        dist_pos = dist_mat.clone()
        dist_pos[~mask_pos] = 0
        hard_pos = dist_pos.max(dim=1)[0]

        dist_neg = dist_mat.clone()
        dist_neg[~mask_neg] = float('inf')
        hard_neg = dist_neg.min(dim=1)[0]

        loss = F.relu(hard_pos - hard_neg + self.margin).mean()
        return loss


# ================== ДАТАСЕТ ==================
class VeRiDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform

        if split == 'train':
            self.img_dir = os.path.join(data_dir, 'image_train')
            list_file = os.path.join(data_dir, 'name_train.txt')
        elif split == 'query':
            self.img_dir = os.path.join(data_dir, 'image_query')
            list_file = os.path.join(data_dir, 'name_query.txt')
        else:  # gallery
            self.img_dir = os.path.join(data_dir, 'image_test')
            list_file = os.path.join(data_dir, 'name_test.txt')

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f'Папка не найдена: {self.img_dir}')

        self.samples = []
        self.label_to_idx = {}
        self.idx_to_samples = defaultdict(list)

        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
        else:
            filenames = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

        for fname in filenames:
            parts = fname.split('_')
            if len(parts) >= 2:
                vid = int(parts[0])
                cam = int(parts[1][1:]) if parts[1].startswith('c') else 0

                if vid not in self.label_to_idx:
                    self.label_to_idx[vid] = len(self.label_to_idx)

                label = self.label_to_idx[vid]
                self.samples.append((fname, label, cam))
                self.idx_to_samples[label].append(len(self.samples) - 1)

        self.num_classes = len(self.label_to_idx)
        print(f'{split}: {len(self.samples)} images, {self.num_classes} IDs')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label, cam = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': label, 'camera_id': cam}


class PKSampler:
    def __init__(self, dataset, p, k):
        self.p, self.k = p, k
        self.idx_to_samples = dataset.idx_to_samples
        self.labels = list(self.idx_to_samples.keys())

    def __iter__(self):
        random.shuffle(self.labels)
        batch = []
        for label in self.labels:
            indices = self.idx_to_samples[label]
            selected = random.sample(indices, self.k) if len(indices) >= self.k else random.choices(indices, k=self.k)
            batch.extend(selected)
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]

    def __len__(self):
        return len(self.labels) // self.p


# ================== ЗАГРУЗКА ДАННЫХ ==================
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
])

test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def main():
    global model, train_loader, query_loader, gallery_loader, criterion_ce, criterion_triplet, optimizer, scheduler, scaler

    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('CUDA недоступна! Обучение будет медленным.')
        sys.exit(1)

    print(f'timm version: {timm.__version__}')
    print(f'\nConfig: {EPOCHS} epochs, batch size {BATCH_SIZE} (P={P}, K={K})')
    print(f'Data dir: {DATA_DIR}')

    print('\nЗагрузка датасета...')
    train_dataset = VeRiDataset(DATA_DIR, split='train', transform=train_transform)
    query_dataset = VeRiDataset(DATA_DIR, split='query', transform=test_transform)
    gallery_dataset = VeRiDataset(DATA_DIR, split='gallery', transform=test_transform)

    NUM_CLASSES = train_dataset.num_classes
    print(f'Total classes: {NUM_CLASSES}')

    train_sampler = PKSampler(train_dataset, p=P, k=K)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print(f'Train batches per epoch: {len(train_loader)}')

    # ================== ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ==================
    print('\nИнициализация модели...')
    model = ViTReID(num_classes=NUM_CLASSES, model_size=VIT_SIZE, pretrained=True).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {num_params:,}')

    criterion_ce = CrossEntropyLabelSmooth(NUM_CLASSES, epsilon=LABEL_SMOOTH)
    criterion_triplet = TripletLoss(margin=MARGIN)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scaler = GradScaler()


    # ================== ФУНКЦИИ ОЦЕНКИ ==================
    @torch.no_grad()
    def extract_features(model, loader):
        model.eval()
        features, labels, cameras = [], [], []
        for batch in tqdm(loader, desc='Extracting features', leave=False):
            imgs = batch['image'].to(device)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.append(batch['label'].numpy())
            cameras.append(batch['camera_id'].numpy())
        return np.concatenate(features), np.concatenate(labels), np.concatenate(cameras)

    def compute_metrics(query_feats, gallery_feats, query_labels, gallery_labels, query_cams, gallery_cams):
        dist_mat = 1 - np.dot(query_feats, gallery_feats.T)
        num_query = len(query_labels)
        all_AP, all_cmc = [], np.zeros(50)

        for i in range(num_query):
            q_label, q_cam = query_labels[i], query_cams[i]
            valid_mask = ~((gallery_labels == q_label) & (gallery_cams == q_cam))
            if valid_mask.sum() == 0:
                continue

            distances = dist_mat[i][valid_mask]
            g_labels = gallery_labels[valid_mask]
            indices = np.argsort(distances)
            matches = (g_labels[indices] == q_label).astype(np.int32)

            cmc = matches.cumsum()
            cmc[cmc > 1] = 1
            all_cmc[:len(cmc)] += cmc[:50]

            num_rel = matches.sum()
            if num_rel > 0:
                precision = matches.cumsum() / (np.arange(len(matches)) + 1)
                all_AP.append((precision * matches).sum() / num_rel)

        cmc = all_cmc / num_query * 100
        return {
            'Rank-1': cmc[0],
            'Rank-5': cmc[4],
            'Rank-10': cmc[9],
            'mAP': np.mean(all_AP) * 100
        }

    def evaluate():
        print('Evaluating...')
        query_feats, query_labels, query_cams = extract_features(model, query_loader)
        gallery_feats, gallery_labels, gallery_cams = extract_features(model, gallery_loader)
        return compute_metrics(query_feats, gallery_feats, query_labels, gallery_labels, query_cams, gallery_cams)

    # ================== ОБУЧЕНИЕ ==================
    def train_one_epoch(epoch):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')

        for batch_idx, batch in enumerate(pbar):
            # Warmup
            if epoch < WARMUP_EPOCHS:
                progress = (epoch * len(train_loader) + batch_idx) / (WARMUP_EPOCHS * len(train_loader))
                for pg in optimizer.param_groups:
                    pg['lr'] = LR * progress

            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast():
                features, bn_features, logits = model(imgs)
                loss = criterion_ce(logits, labels) + criterion_triplet(features, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        return total_loss / len(train_loader)

    # ================== ОСНОВНОЙ ЦИКЛ ==================
    print('\n' + '='*60)
    print(f'TRAINING ViT-{VIT_SIZE} + BNNeck | {EPOCHS} EPOCHS')
    print('='*60)

    best_mAP = 0
    history = {'loss': [], 'mAP': [], 'rank1': [], 'rank5': [], 'rank10': []}

    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss = train_one_epoch(epoch)
        history['loss'].append(avg_loss)

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Time: {epoch_time:.1f}s')

        # Оценка каждые 10 эпох и на последней
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            results = evaluate()
            history['mAP'].append(results['mAP'])
            history['rank1'].append(results['Rank-1'])
            history['rank5'].append(results['Rank-5'])
            history['rank10'].append(results['Rank-10'])

            print(f"  Rank-1: {results['Rank-1']:.2f}% | Rank-5: {results['Rank-5']:.2f}% | "
                  f"Rank-10: {results['Rank-10']:.2f}% | mAP: {results['mAP']:.2f}%")

            if results['mAP'] > best_mAP:
                best_mAP = results['mAP']
                torch.save(model.state_dict(), f'{SAVE_DIR}/best_model.pth')
                print(f'  *** New best mAP: {best_mAP:.2f}% - Model saved! ***')

    print('\n' + '='*60)
    print(f'Training finished! Best mAP: {best_mAP:.2f}%')
    print('='*60)

    # ================== ФИНАЛЬНАЯ ОЦЕНКА ==================
    print('\nЗагрузка лучшей модели для финальной оценки...')
    model.load_state_dict(torch.load(f'{SAVE_DIR}/best_model.pth'))
    final_results = evaluate()

    print('\n' + '='*50)
    print(f'ViT-{VIT_SIZE.upper()} FINAL RESULTS')
    print('='*50)
    print(f"Rank-1:  {final_results['Rank-1']:.2f}%")
    print(f"Rank-5:  {final_results['Rank-5']:.2f}%")
    print(f"Rank-10: {final_results['Rank-10']:.2f}%")
    print(f"mAP:     {final_results['mAP']:.2f}%")
    print('='*50)

    # ================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==================
    result_data = {
        'model': f'ViT-{VIT_SIZE} + BNNeck',
        'dataset': 'VeRi-776',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'num_classes': NUM_CLASSES,
        'final_results': {k: float(v) for k, v in final_results.items()},
        'best_mAP': float(best_mAP),
        'history': history
    }

    result_path = f'{SAVE_DIR}/result.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f'\nРезультаты сохранены в: {result_path}')
    print(f'Лучшая модель сохранена в: {SAVE_DIR}/best_model.pth')
    print('\nГотово!')


if __name__ == '__main__':
    main()
