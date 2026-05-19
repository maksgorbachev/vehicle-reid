"""
Vehicle ReID Model: ResNet50/101 + BNNeck
==========================================
Baseline архитектура для реидентификации транспортных средств.
Основана на работе "Bag of Tricks" (Luo et al., 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BNNeck(nn.Module):
    """
    Batch Normalization Neck.
    Разделяет пространство признаков для Classification и Metric Learning.
    - До BN: для Triplet Loss (сохраняет масштаб)
    - После BN + L2-норм: для Cross-Entropy и инференса
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # BatchNorm без bias (важно для правильной нормализации)
        self.bn = nn.BatchNorm1d(in_features)
        self.bn.bias.requires_grad_(False)

        # Классификатор для Cross-Entropy Loss
        self.classifier = nn.Linear(in_features, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.bn.weight, 1.0, 0.02)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, features):
        """
        Args:
            features: [B, C] - признаки до BNNeck
        Returns:
            bn_features: [B, C] - после BN (для классификации)
            logits: [B, num_classes] - логиты классификатора
        """
        bn_features = self.bn(features)
        logits = self.classifier(bn_features)
        return bn_features, logits


_ARCH_CONFIG = {
    'resnet50':  {'builder': models.resnet50,  'weights': 'IMAGENET1K_V1', 'dim': 2048},
    'resnet101': {'builder': models.resnet101, 'weights': 'IMAGENET1K_V1', 'dim': 2048},
}


class VehicleReIDModel(nn.Module):
    """
    Модель для Vehicle Re-identification.

    Архитектура:
    - Backbone: ResNet-50 или ResNet-101 (pretrained на ImageNet)
    - Global Average Pooling
    - BNNeck для разделения целей обучения

    Режимы работы:
    - Training: возвращает (features, bn_features, logits)
    - Inference: возвращает L2-нормированные bn_features
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = 'resnet50',
        pretrained: bool = True,
        last_stride: int = 1
    ):
        super().__init__()
        self.num_classes = num_classes

        if arch not in _ARCH_CONFIG:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(_ARCH_CONFIG)}")

        cfg = _ARCH_CONFIG[arch]
        self.embedding_dim = cfg['dim']

        resnet = cfg['builder'](weights=cfg['weights'] if pretrained else None)

        # Модифицируем последний stride для большего разрешения feature map
        if last_stride == 1:
            resnet.layer4[0].downsample[0].stride = (1, 1)
            resnet.layer4[0].conv2.stride = (1, 1)

        # Убираем FC слой
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # BNNeck
        self.bnneck = BNNeck(self.embedding_dim, num_classes)

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 3, H, W]
            return_features: если True, возвращает промежуточные представления

        Returns (training):
            features [B, D], bn_features [B, D], logits [B, num_classes]
        Returns (inference):
            embeddings [B, D] — L2-нормированные
        """
        # Backbone
        feat_map = self.backbone(x)  # [B, 2048, H/16, W/16] или [B, 2048, H/32, W/32]

        # Global pooling
        features = self.gap(feat_map).flatten(1)  # [B, 2048]

        # BNNeck
        bn_features, logits = self.bnneck(features)

        if self.training or return_features:
            return features, bn_features, logits
        else:
            # Инференс: L2-нормированные эмбеддинги
            embeddings = F.normalize(bn_features, p=2, dim=1)
            return embeddings

    def extract_features(self, x):
        """Извлечение L2-нормированных эмбеддингов для инференса."""
        self.eval()
        with torch.no_grad():
            return self(x)


def build_model(
    num_classes: int,
    arch: str = 'resnet50',
    pretrained: bool = True,
    last_stride: int = 1
):
    """Фабричная функция. arch: 'resnet50' | 'resnet101'."""
    return VehicleReIDModel(
        num_classes=num_classes,
        arch=arch,
        pretrained=pretrained,
        last_stride=last_stride
    )


if __name__ == "__main__":
    for arch in ['resnet50', 'resnet101']:
        model = build_model(num_classes=576, arch=arch, pretrained=False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{arch}: {n_params:,} params, embedding_dim={model.embedding_dim}")

        x = torch.randn(2, 3, 256, 128)
        model.train()
        features, bn_features, logits = model(x)
        print(f"  train: features={features.shape}, logits={logits.shape}")

        model.eval()
        emb = model(x)
        print(f"  eval:  embeddings={emb.shape}, norm={emb.norm(dim=1).tolist()}")
