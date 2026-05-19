"""
Loss Functions for Vehicle ReID
================================
Реализация функций потерь для обучения модели реидентификации:
- Cross-Entropy с Label Smoothing
- Triplet Loss с hard mining
- Комбинированная функция потерь
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross-Entropy Loss с Label Smoothing.

    Label Smoothing помогает предотвратить переобучение и улучшает
    обобщающую способность модели за счёт "смягчения" меток.

    Формула:
        y_smooth = (1 - epsilon) * one_hot(y) + epsilon / num_classes
        loss = -sum(y_smooth * log(softmax(logits)))
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] - выход классификатора
            targets: [B] - истинные метки классов

        Returns:
            loss: скалярное значение потерь
        """
        log_probs = self.logsoftmax(logits)

        # Создаём smoothed targets
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        targets_smooth = (1 - self.epsilon) * targets_one_hot + \
                         self.epsilon / self.num_classes

        # Cross-entropy с smoothed targets
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss с Hard Mining.

    Для каждого якоря (anchor) выбирает:
    - Hard positive: самый далёкий позитив (того же класса)
    - Hard negative: самый близкий негатив (другого класса)

    Формула:
        L = max(0, margin + d(anchor, hard_pos) - d(anchor, hard_neg))

    Hard mining значительно ускоряет сходимость и улучшает качество
    по сравнению с random/semi-hard mining.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] - эмбеддинги (рекомендуется НЕ L2-нормированные)
            labels: [B] - метки классов

        Returns:
            loss: скалярное значение потерь
        """
        # Вычисляем попарные евклидовы расстояния
        dist_mat = self._euclidean_dist(features, features)

        # Для каждого якоря находим hard positive и hard negative
        loss = self._hard_mining_triplet_loss(dist_mat, labels)
        return loss

    def _euclidean_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Вычисление матрицы попарных евклидовых расстояний."""
        m, n = x.size(0), y.size(0)
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist = xx + yy - 2 * torch.mm(x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # Численная стабильность
        return dist

    def _hard_mining_triplet_loss(
        self, dist_mat: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Hard mining: выбор наиболее сложных триплетов."""
        batch_size = dist_mat.size(0)

        # Маска позитивных пар (тот же класс, но не сам элемент)
        is_pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        is_neg = labels.unsqueeze(0).ne(labels.unsqueeze(1))

        # Hard positive: максимальное расстояние среди позитивов
        # Для каждого anchor выбираем самый далёкий позитив
        dist_ap = dist_mat.clone()
        dist_ap[~is_pos] = 0  # Обнуляем негативы
        dist_ap_max, _ = dist_ap.max(dim=1)  # [B]

        # Hard negative: минимальное расстояние среди негативов
        # Для каждого anchor выбираем самый близкий негатив
        dist_an = dist_mat.clone()
        dist_an[~is_neg] = float('inf')  # "Убираем" позитивы
        dist_an_min, _ = dist_an.min(dim=1)  # [B]

        # Triplet loss с margin
        loss = F.relu(dist_ap_max - dist_an_min + self.margin)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Комбинированная функция потерь для Vehicle ReID.

    L_total = L_CE + lambda_triplet * L_triplet

    Комбинирование CE и Triplet Loss позволяет:
    - CE: учить классификации по ID (supervision signal)
    - Triplet: формировать метрическое пространство для ReID
    """

    def __init__(
        self,
        num_classes: int,
        margin: float = 0.3,
        epsilon: float = 0.1,
        triplet_weight: float = 1.0
    ):
        super().__init__()
        self.ce_loss = CrossEntropyLabelSmooth(num_classes, epsilon)
        self.triplet_loss = TripletLoss(margin)
        self.triplet_weight = triplet_weight

    def forward(
        self,
        features: torch.Tensor,
        bn_features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Args:
            features: [B, D] - признаки до BNNeck (для Triplet)
            bn_features: [B, D] - признаки после BNNeck
            logits: [B, C] - логиты классификатора
            labels: [B] - метки классов

        Returns:
            dict с компонентами потерь и общей суммой
        """
        loss_ce = self.ce_loss(logits, labels)
        loss_triplet = self.triplet_loss(features, labels)
        loss_total = loss_ce + self.triplet_weight * loss_triplet

        return {
            'loss': loss_total,
            'loss_ce': loss_ce,
            'loss_triplet': loss_triplet
        }


# Тест функций потерь
if __name__ == "__main__":
    batch_size = 64
    num_classes = 576
    embedding_dim = 2048

    # Создаём тестовые данные
    features = torch.randn(batch_size, embedding_dim)
    bn_features = torch.randn(batch_size, embedding_dim)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Тестируем CE Loss
    ce_loss = CrossEntropyLabelSmooth(num_classes, epsilon=0.1)
    loss_ce = ce_loss(logits, labels)
    print(f"CE Loss: {loss_ce.item():.4f}")

    # Тестируем Triplet Loss
    triplet_loss = TripletLoss(margin=0.3)
    loss_triplet = triplet_loss(features, labels)
    print(f"Triplet Loss: {loss_triplet.item():.4f}")

    # Тестируем комбинированный loss
    combined_loss = CombinedLoss(num_classes, margin=0.3, epsilon=0.1)
    losses = combined_loss(features, bn_features, logits, labels)
    print(f"Combined Loss: {losses['loss'].item():.4f}")
    print(f"  - CE: {losses['loss_ce'].item():.4f}")
    print(f"  - Triplet: {losses['loss_triplet'].item():.4f}")
