"""
Vehicle ReID Model: ViT (Vision Transformer) + BNNeck
=====================================================
Transformer-based архитектура для реидентификации транспортных средств.
Использует pretrained ViT из библиотеки timm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("Установи timm: pip install timm")


class BNNeck(nn.Module):
    """
    Batch Normalization Neck (идентичен ResNet версии).
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_features, num_classes, bias=False)

        nn.init.normal_(self.bn.weight, 1.0, 0.02)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, features):
        bn_features = self.bn(features)
        logits = self.classifier(bn_features)
        return bn_features, logits


class ViTReIDModel(nn.Module):
    """
    Vision Transformer для Vehicle Re-identification.

    Архитектура:
    - Backbone: ViT-Small/16 (pretrained на ImageNet-21k)
    - CLS token как глобальное представление
    - BNNeck для разделения целей обучения

    Преимущества ViT перед CNN:
    - Глобальное внимание с первого слоя
    - Лучше моделирует дальние зависимости
    - Меньше inductive bias

    Недостатки:
    - Требует больше данных для обучения
    - Медленнее на маленьких изображениях
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = 'vit_small_patch16_224',
        pretrained: bool = True,
        embedding_dim: int = 384,  # ViT-Small: 384, ViT-Base: 768
        img_size: int = 224,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Загружаем ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Убираем классификатор
            img_size=img_size,
        )

        # Получаем реальную размерность эмбеддинга
        self.embedding_dim = self.backbone.embed_dim

        # BNNeck
        self.bnneck = BNNeck(self.embedding_dim, num_classes)

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: [B, 3, H, W] - входные изображения (256x128 будут ресайзнуты внутри)
            return_features: если True, возвращает все промежуточные представления

        Returns:
            Training mode:
                features: [B, embed_dim] - до BNNeck (для Triplet Loss)
                bn_features: [B, embed_dim] - после BNNeck (для CE Loss)
                logits: [B, num_classes] - логиты классификатора
            Inference mode:
                embeddings: [B, embed_dim] - L2-нормированные эмбеддинги
        """
        # ViT backbone (CLS token)
        features = self.backbone(x)  # [B, embed_dim]

        # BNNeck
        bn_features, logits = self.bnneck(features)

        if self.training or return_features:
            return features, bn_features, logits
        else:
            embeddings = F.normalize(bn_features, p=2, dim=1)
            return embeddings

    def extract_features(self, x):
        """Извлечение L2-нормированных эмбеддингов для инференса."""
        self.eval()
        with torch.no_grad():
            return self(x)


def build_vit_model(
    num_classes: int,
    model_size: str = 'small',  # 'tiny', 'small', 'base'
    pretrained: bool = True,
    img_size: int = 224,
):
    """
    Фабричная функция для создания ViT модели.

    Args:
        num_classes: количество классов (идентичностей)
        model_size: размер модели
            - 'tiny': ViT-Ti/16, embed_dim=192, ~5M params (для слабых GPU)
            - 'small': ViT-S/16, embed_dim=384, ~22M params (рекомендуется)
            - 'base': ViT-B/16, embed_dim=768, ~86M params (нужно много VRAM)
        pretrained: использовать pretrained веса
    """
    model_configs = {
        'tiny': ('vit_tiny_patch16_224', 192),
        'small': ('vit_small_patch16_224', 384),
        'base': ('vit_base_patch16_224', 768),
    }

    if model_size not in model_configs:
        raise ValueError(f"model_size должен быть одним из: {list(model_configs.keys())}")

    model_name, embed_dim = model_configs[model_size]

    return ViTReIDModel(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        embedding_dim=embed_dim,
        img_size=img_size,
    )


# Тест модели
if __name__ == "__main__":
    # Проверяем что timm установлен
    print(f"timm version: {timm.__version__}")

    # Тестируем разные размеры
    for size in ['tiny', 'small']:
        print(f"\n=== ViT-{size.upper()} ===")
        model = build_vit_model(num_classes=576, model_size=size, pretrained=True)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")
        print(f"Embedding dim: {model.embedding_dim}")

        # Test forward pass
        x = torch.randn(2, 3, 256, 128)

        model.train()
        features, bn_features, logits = model(x)
        print(f"Training - features: {features.shape}, logits: {logits.shape}")

        model.eval()
        embeddings = model(x)
        print(f"Inference - embeddings: {embeddings.shape}")

        # Проверка памяти
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(x)
            print(f"VRAM (inference, batch=2): {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
