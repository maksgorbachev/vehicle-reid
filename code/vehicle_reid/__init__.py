"""
Vehicle ReID Pipeline
=====================
Пайплайн для реидентификации транспортных средств.

Модули:
- model: ResNet50 + BNNeck архитектура
- losses: Cross-Entropy с Label Smoothing, Triplet Loss
- dataset: Загрузчики VeRi-776, VehicleID, PK-семплер
- transforms: Аугментации (Random Erasing, ColorJitter)
- reranking: K-reciprocal re-ranking
- evaluation: Метрики CMC и mAP
"""

from .model import VehicleReIDModel, build_model
from .losses import CrossEntropyLabelSmooth, TripletLoss, CombinedLoss
from .dataset import VeRiDataset, VehicleIDDataset, PKSampler, DemoVehicleDataset
from .transforms import build_transforms, get_train_transforms, get_test_transforms
from .evaluation import evaluate_reid, ReIDEvaluator

__version__ = "1.0.0"
__all__ = [
    "VehicleReIDModel",
    "build_model",
    "CrossEntropyLabelSmooth",
    "TripletLoss",
    "CombinedLoss",
    "VeRiDataset",
    "VehicleIDDataset",
    "PKSampler",
    "DemoVehicleDataset",
    "build_transforms",
    "get_train_transforms",
    "get_test_transforms",
    "evaluate_reid",
    "ReIDEvaluator",
]
