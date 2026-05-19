"""
Data Transforms and Augmentations for Vehicle ReID
===================================================
Аугментации, повышающие устойчивость модели к:
- Окклюзиям (Random Erasing)
- Вариациям освещения (ColorJitter)
- Межкамерному доменному сдвигу
"""

import math
import random
from typing import Tuple

import torch
from torchvision import transforms
from PIL import Image
import numpy as np


class RandomErasing:
    """
    Random Erasing Data Augmentation.

    Случайно стирает прямоугольную область на изображении,
    заполняя её случайными значениями или средним по ImageNet.

    Эффект: повышает устойчивость к окклюзиям и частичному
    перекрытию объекта на изображении.

    Reference: Zhong et al., "Random Erasing Data Augmentation", AAAI 2020
    """

    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    ):
        """
        Args:
            probability: вероятность применения
            sl: минимальная доля площади стираемой области
            sh: максимальная доля площади стираемой области
            r1: минимальное соотношение сторон
            mean: значения для заполнения (ImageNet mean)
        """
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: [C, H, W] тензор изображения
        Returns:
            img: тензор с возможно стёртой областью
        """
        if random.random() > self.probability:
            return img

        for _ in range(100):  # Максимум 100 попыток
            area = img.size(1) * img.size(2)

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)

                # Заполняем средними значениями по каналам
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                return img

        return img


class RandomPatch:
    """
    Random Patch Augmentation.

    Заменяет случайный патч на изображении патчем из другого
    места того же изображения. Симулирует частичные перекрытия.
    """

    def __init__(
        self,
        probability: float = 0.5,
        patch_size: float = 0.1
    ):
        self.probability = probability
        self.patch_size = patch_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.probability:
            return img

        c, h, w = img.shape
        patch_h = int(h * self.patch_size)
        patch_w = int(w * self.patch_size)

        # Источник патча
        src_x = random.randint(0, h - patch_h)
        src_y = random.randint(0, w - patch_w)

        # Место назначения
        dst_x = random.randint(0, h - patch_h)
        dst_y = random.randint(0, w - patch_w)

        # Копируем патч
        img[:, dst_x:dst_x + patch_h, dst_y:dst_y + patch_w] = \
            img[:, src_x:src_x + patch_h, src_y:src_y + patch_w].clone()

        return img


def build_transforms(
    img_size: Tuple[int, int] = (256, 128),
    is_train: bool = True,
    random_erasing_prob: float = 0.5,
    color_jitter: bool = True,
    auto_augment: bool = False
):
    """
    Построение пайплайна трансформаций.

    Args:
        img_size: (height, width) целевой размер
        is_train: режим обучения или инференса
        random_erasing_prob: вероятность Random Erasing
        color_jitter: применять ли цветовые искажения
        auto_augment: использовать AutoAugment

    Returns:
        transforms.Compose pipeline
    """

    # Нормализация по статистикам ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_train:
        transform_list = [
            # Resize с небольшим запасом для crop
            transforms.Resize((img_size[0] + 32, img_size[1] + 16)),
            # Random crop до целевого размера
            transforms.RandomCrop(img_size),
            # Горизонтальное отражение
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        # Цветовые искажения для устойчивости к освещению
        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.15,
                    saturation=0.1,
                    hue=0.05
                )
            )

        # AutoAugment (опционально)
        if auto_augment:
            transform_list.append(
                transforms.AutoAugment(
                    transforms.AutoAugmentPolicy.IMAGENET
                )
            )

        # Преобразование в тензор и нормализация
        transform_list.extend([
            transforms.ToTensor(),
            normalize,
        ])

        # Random Erasing (после ToTensor)
        if random_erasing_prob > 0:
            transform_list.append(
                RandomErasing(probability=random_erasing_prob)
            )

    else:
        # Inference transforms - без аугментаций
        transform_list = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ]

    return transforms.Compose(transform_list)


def get_train_transforms(
    img_size: Tuple[int, int] = (256, 128),
    augmentation_level: str = 'standard'
):
    """
    Получение трансформаций для обучения.

    Args:
        img_size: размер изображения
        augmentation_level: 'minimal', 'standard', 'strong'
    """
    if augmentation_level == 'minimal':
        return build_transforms(
            img_size=img_size,
            is_train=True,
            random_erasing_prob=0.0,
            color_jitter=False
        )
    elif augmentation_level == 'standard':
        return build_transforms(
            img_size=img_size,
            is_train=True,
            random_erasing_prob=0.5,
            color_jitter=True
        )
    else:  # strong
        return build_transforms(
            img_size=img_size,
            is_train=True,
            random_erasing_prob=0.7,
            color_jitter=True,
            auto_augment=True
        )


def get_test_transforms(img_size: Tuple[int, int] = (256, 128)):
    """Получение трансформаций для тестирования."""
    return build_transforms(img_size=img_size, is_train=False)


# Визуализация аугментаций
def visualize_augmentations(img_path: str, num_samples: int = 5):
    """
    Визуализация эффекта аугментаций на изображении.
    Полезно для отладки и презентаций.
    """
    import matplotlib.pyplot as plt

    img = Image.open(img_path).convert('RGB')
    transform = get_train_transforms(augmentation_level='standard')

    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))

    # Оригинал
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Аугментированные версии
    for i in range(num_samples):
        aug_img = transform(img)
        # Денормализация для визуализации
        aug_img = aug_img.permute(1, 2, 0).numpy()
        aug_img = aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        aug_img = np.clip(aug_img, 0, 1)

        axes[i + 1].imshow(aug_img)
        axes[i + 1].set_title(f'Aug {i + 1}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150)
    plt.close()


# Тест трансформаций
if __name__ == "__main__":
    # Тестируем на случайном изображении
    img = Image.fromarray(
        np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    )

    train_transform = get_train_transforms(augmentation_level='standard')
    test_transform = get_test_transforms()

    train_tensor = train_transform(img)
    test_tensor = test_transform(img)

    print(f"Train transform output: {train_tensor.shape}")
    print(f"Test transform output: {test_tensor.shape}")

    # Проверка Random Erasing
    re = RandomErasing(probability=1.0)
    test_img = torch.randn(3, 256, 128)
    erased = re(test_img.clone())
    diff = (test_img != erased).float().mean()
    print(f"Random Erasing affected {diff * 100:.1f}% of pixels")
