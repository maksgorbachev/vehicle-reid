"""
Training Script for Vehicle ReID
=================================
Основной скрипт обучения модели реидентификации транспортных средств.

Использование:
    python train.py --data_dir /path/to/VeRi-776 --epochs 120

Для демо-режима (без реальных данных):
    python train.py --demo --epochs 10
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.amp import GradScaler, autocast
import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vehicle_reid.model import build_model
from vehicle_reid.vit_model import build_vit_model
from vehicle_reid.losses import CombinedLoss
from vehicle_reid.dataset import VeRiDataset, DemoVehicleDataset, PKSampler
from vehicle_reid.transforms import get_train_transforms, get_test_transforms
from vehicle_reid.evaluation import ReIDEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Vehicle ReID Training')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/VeRi-776',
                        help='Path to dataset')
    parser.add_argument('--demo', action='store_true',
                        help='Use demo dataset for testing')

    # Model
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'vit'],
                        help='Model architecture: resnet50, resnet101, or vit')
    parser.add_argument('--vit_size', type=str, default='small',
                        choices=['tiny', 'small', 'base'],
                        help='ViT model size (for vit model type)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--last_stride', type=int, default=1,
                        help='Last stride of backbone (1 or 2, for ResNet)')

    # Training
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (P*K for PK-sampling)')
    parser.add_argument('--p', type=int, default=16,
                        help='Number of identities per batch')
    parser.add_argument('--k', type=int, default=4,
                        help='Number of instances per identity')
    parser.add_argument('--lr', type=float, default=3.5e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')

    # Loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Triplet loss margin')
    parser.add_argument('--label_smooth', type=float, default=0.1,
                        help='Label smoothing epsilon')

    # Augmentation
    parser.add_argument('--aug_level', type=str, default='standard',
                        choices=['minimal', 'standard', 'strong'],
                        help='Augmentation level')

    # Misc
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluation frequency (epochs)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


class Trainer:
    """Класс для обучения модели Vehicle ReID."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Создаём директорию для чекпоинтов
        os.makedirs(args.save_dir, exist_ok=True)

        # Загружаем данные
        self._setup_data()

        # Создаём модель
        self._setup_model()

        # Настраиваем обучение
        self._setup_training()

        # Логгирование
        self.train_losses = []
        self.eval_results = []

    def _setup_data(self):
        """Настройка датасетов и загрузчиков."""
        args = self.args

        train_transform = get_train_transforms(
            img_size=(256, 128),
            augmentation_level=args.aug_level
        )
        test_transform = get_test_transforms(img_size=(256, 128))

        if args.demo:
            print("Using DEMO dataset for testing...")
            self.train_dataset = DemoVehicleDataset(
                num_ids=200, images_per_id=10,
                transform=train_transform, split='train'
            )
            self.query_dataset = DemoVehicleDataset(
                num_ids=50, images_per_id=2,
                transform=test_transform, split='query'
            )
            self.gallery_dataset = DemoVehicleDataset(
                num_ids=50, images_per_id=8,
                transform=test_transform, split='gallery'
            )
        else:
            print(f"Loading VeRi-776 from {args.data_dir}...")
            self.train_dataset = VeRiDataset(
                args.data_dir, split='train', transform=train_transform
            )
            self.query_dataset = VeRiDataset(
                args.data_dir, split='query', transform=test_transform
            )
            self.gallery_dataset = VeRiDataset(
                args.data_dir, split='gallery', transform=test_transform
            )

        self.num_classes = self.train_dataset.num_classes
        print(f"Training set: {len(self.train_dataset)} images, {self.num_classes} IDs")
        print(f"Query set: {len(self.query_dataset)} images")
        print(f"Gallery set: {len(self.gallery_dataset)} images")

        # PK-семплер для обучения
        self.train_sampler = PKSampler(
            self.train_dataset, p=args.p, k=args.k
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True
        )

        self.query_loader = DataLoader(
            self.query_dataset,
            batch_size=128, shuffle=False,
            num_workers=4, pin_memory=True
        )

        self.gallery_loader = DataLoader(
            self.gallery_dataset,
            batch_size=128, shuffle=False,
            num_workers=4, pin_memory=True
        )

    def _setup_model(self):
        """Создание модели."""
        args = self.args

        if args.model_type in ('resnet50', 'resnet101'):
            label = 'ResNet-50' if args.model_type == 'resnet50' else 'ResNet-101'
            print(f"Building {label} model...")
            self.model = build_model(
                num_classes=self.num_classes,
                arch=args.model_type,
                pretrained=args.pretrained,
                last_stride=args.last_stride
            )
            self.model_name = label
        elif args.model_type == 'vit':
            print(f"Building ViT-{args.vit_size} model...")
            self.model = build_vit_model(
                num_classes=self.num_classes,
                model_size=args.vit_size,
                pretrained=args.pretrained
            )
            self.model_name = f'ViT-{args.vit_size}'
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        self.model = self.model.to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {self.model_name}")
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _setup_training(self):
        """Настройка оптимизации."""
        args = self.args

        # Loss
        self.criterion = CombinedLoss(
            num_classes=self.num_classes,
            margin=args.margin,
            epsilon=args.label_smooth
        )

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Scheduler (cosine annealing после warmup)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-6
        )

        # Mixed precision (init_scale=256 — safer for large backbones like ResNet-101)
        self.scaler = GradScaler('cuda', init_scale=256) if args.fp16 else None

        # Evaluator
        self.evaluator = ReIDEvaluator(max_rank=50, remove_same_camera=True)

    def warmup_lr(self, epoch, batch_idx, num_batches):
        """Линейный warmup learning rate."""
        if epoch < self.args.warmup_epochs:
            progress = (epoch * num_batches + batch_idx) / \
                       (self.args.warmup_epochs * num_batches)
            lr = self.args.lr * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_epoch(self, epoch):
        """Обучение одной эпохи."""
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_triplet = 0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Warmup
            self.warmup_lr(epoch, batch_idx, num_batches)

            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:  # Mixed precision
                with autocast('cuda'):
                    features, bn_features, logits = self.model(images)
                    losses = self.criterion(features, bn_features, logits, labels)
                    loss = losses['loss']

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                features, bn_features, logits = self.model(images)
                losses = self.criterion(features, bn_features, logits, labels)
                loss = losses['loss']

                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_ce += losses['loss_ce'].item()
            total_triplet += losses['loss_triplet'].item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"(CE: {losses['loss_ce'].item():.4f}, "
                      f"Triplet: {losses['loss_triplet'].item():.4f})")

        # Scheduler step после warmup
        if epoch >= self.args.warmup_epochs:
            self.scheduler.step()

        avg_loss = total_loss / num_batches
        avg_ce = total_ce / num_batches
        avg_triplet = total_triplet / num_batches

        return {
            'loss': avg_loss,
            'loss_ce': avg_ce,
            'loss_triplet': avg_triplet
        }

    @torch.no_grad()
    def extract_features(self, loader):
        """Извлечение признаков для оценки."""
        self.model.eval()
        features_list = []
        labels_list = []
        cameras_list = []

        for batch in loader:
            images = batch['image'].to(self.device)
            labels = batch['label']
            cameras = batch['camera_id']

            features = self.model(images)  # L2-normalized
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            cameras_list.append(cameras.numpy())

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        cameras = np.concatenate(cameras_list, axis=0)

        return features, labels, cameras

    def evaluate(self, reranking=False):
        """Оценка на тестовом наборе."""
        print("\nExtracting features...")
        query_feats, query_labels, query_cams = self.extract_features(self.query_loader)
        gallery_feats, gallery_labels, gallery_cams = self.extract_features(self.gallery_loader)

        print(f"Query: {query_feats.shape}, Gallery: {gallery_feats.shape}")

        results = self.evaluator.evaluate(
            query_feats, gallery_feats,
            query_labels, gallery_labels,
            query_cams, gallery_cams,
            reranking=reranking
        )

        return results

    def train(self):
        """Полный цикл обучения."""
        args = self.args
        best_mAP = 0

        print(f"\n{'='*60}")
        print(f" Starting training for {args.epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(args.epochs):
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics)

            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']

            print(f"\nEpoch [{epoch+1}/{args.epochs}] "
                  f"Loss: {train_metrics['loss']:.4f} "
                  f"LR: {lr:.6f} "
                  f"Time: {epoch_time:.1f}s")

            # Evaluation
            if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
                print("\n" + "="*40)
                print(" Evaluation")
                print("="*40)

                results = self.evaluate(reranking=False)
                self.evaluator.print_results(results, "Without Re-ranking")

                results_rr = self.evaluate(reranking=True)
                self.evaluator.print_results(results_rr, "With Re-ranking")

                self.eval_results.append({
                    'epoch': epoch + 1,
                    'no_rr': results,
                    'with_rr': results_rr
                })

                # Save best model
                if results_rr['mAP'] > best_mAP:
                    best_mAP = results_rr['mAP']
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"*** New best mAP: {best_mAP:.2f}% ***")

        # Final save
        self.save_checkpoint(args.epochs - 1, is_best=False)

        print(f"\n{'='*60}")
        print(f" Training completed!")
        print(f" Best mAP: {best_mAP:.2f}%")
        print(f"{'='*60}\n")

        return self.train_losses, self.eval_results

    def save_checkpoint(self, epoch, is_best=False):
        """Сохранение чекпоинта."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'eval_results': self.eval_results,
        }

        if is_best:
            path = os.path.join(self.args.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.args.save_dir, f'checkpoint_epoch{epoch+1}.pth')

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")


def main():
    args = parse_args()

    # Фиксируем seed для воспроизводимости
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    trainer = Trainer(args)
    train_losses, eval_results = trainer.train()

    # Сохраняем историю обучения
    history = {
        'train_losses': train_losses,
        'eval_results': eval_results,
        'args': vars(args)
    }
    torch.save(history, os.path.join(args.save_dir, 'training_history.pth'))


if __name__ == '__main__':
    main()
