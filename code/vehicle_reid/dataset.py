"""
Dataset and Samplers for Vehicle ReID
======================================
Загрузчики данных для VeRi-776 и VehicleID с PK-семплингом.
"""

import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import numpy as np


class VeRiDataset(Dataset):
    """
    Датасет VeRi-776 для реидентификации транспортных средств.

    Структура VeRi-776:
    - image_train/ - обучающие изображения
    - image_query/ - запросы для тестирования
    - image_test/  - галерея для тестирования

    Формат имени файла: {vehicle_id}_{camera_id}_{frame}.jpg
    Пример: 0001_c001_00016450_0.jpg
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None
    ):
        """
        Args:
            root: путь к корневой папке VeRi-776
            split: 'train', 'query', или 'gallery'
            transform: трансформации изображений
        """
        self.root = root
        self.split = split
        self.transform = transform

        # Определяем папку в зависимости от split
        if split == 'train':
            self.img_dir = os.path.join(root, 'image_train')
        elif split == 'query':
            self.img_dir = os.path.join(root, 'image_query')
        else:  # gallery
            self.img_dir = os.path.join(root, 'image_test')

        # Загружаем данные
        self.data = self._load_data()

        # Для обучения создаём маппинг ID -> индекс
        if split == 'train':
            self._create_label_mapping()

    def _load_data(self) -> List[Tuple[str, int, int]]:
        """Загрузка списка (путь, vehicle_id, camera_id)."""
        data = []
        pattern = re.compile(r'(\d+)_c(\d+)_')

        img_names = os.listdir(self.img_dir)
        for img_name in sorted(img_names):
            if not img_name.endswith('.jpg'):
                continue
            match = pattern.match(img_name)
            if match:
                vehicle_id = int(match.group(1))
                camera_id = int(match.group(2))
                img_path = os.path.join(self.img_dir, img_name)
                data.append((img_path, vehicle_id, camera_id))

        return data

    def _create_label_mapping(self):
        """Создание маппинга оригинальных ID в последовательные индексы."""
        unique_ids = sorted(set([d[1] for d in self.data]))
        self.id_to_label = {vid: idx for idx, vid in enumerate(unique_ids)}
        self.label_to_id = {idx: vid for vid, idx in self.id_to_label.items()}
        self.num_classes = len(unique_ids)

        # Группировка по идентичностям для PK-семплинга
        self.id_to_indices = defaultdict(list)
        for idx, (_, vid, _) in enumerate(self.data):
            label = self.id_to_label[vid]
            self.id_to_indices[label].append(idx)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        img_path, vehicle_id, camera_id = self.data[idx]

        # Загружаем изображение
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Для обучения используем label, для теста - оригинальный ID
        if self.split == 'train':
            label = self.id_to_label[vehicle_id]
        else:
            label = vehicle_id

        return {
            'image': img,
            'label': label,
            'camera_id': camera_id,
            'path': img_path
        }


class VehicleIDDataset(Dataset):
    """
    Датасет PKU VehicleID.

    Структура:
    - image/ - все изображения
    - train_test_split/
        - train_list.txt
        - test_list_{small|medium|large}.txt

    Формат train_list.txt: {img_name} {vehicle_id}
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        test_size: str = 'small',
        transform=None
    ):
        self.root = root
        self.split = split
        self.test_size = test_size
        self.transform = transform

        self.img_dir = os.path.join(root, 'image')

        # Загружаем списки
        if split == 'train':
            list_file = os.path.join(root, 'train_test_split', 'train_list.txt')
        else:
            list_file = os.path.join(
                root, 'train_test_split', f'test_list_{test_size}.txt'
            )

        self.data = self._load_list(list_file)

        if split == 'train':
            self._create_label_mapping()

    def _load_list(self, list_file: str) -> List[Tuple[str, int]]:
        data = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0] + '.jpg'
                    vehicle_id = int(parts[1])
                    img_path = os.path.join(self.img_dir, img_name)
                    data.append((img_path, vehicle_id))
        return data

    def _create_label_mapping(self):
        unique_ids = sorted(set([d[1] for d in self.data]))
        self.id_to_label = {vid: idx for idx, vid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)

        self.id_to_indices = defaultdict(list)
        for idx, (_, vid) in enumerate(self.data):
            label = self.id_to_label[vid]
            self.id_to_indices[label].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, vehicle_id = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.split == 'train':
            label = self.id_to_label[vehicle_id]
        else:
            label = vehicle_id

        return {
            'image': img,
            'label': label,
            'camera_id': 0,  # VehicleID не имеет camera_id
            'path': img_path
        }


class PKSampler(Sampler):
    """
    PK-Sampler для метрического обучения.

    Каждый батч содержит P идентичностей и K изображений каждой.
    Размер батча = P * K.

    Преимущества PK-семплинга:
    - Гарантирует наличие позитивных пар в каждом батче
    - Позволяет эффективный hard mining внутри батча
    - Балансирует представленность идентичностей
    """

    def __init__(
        self,
        dataset: Dataset,
        p: int = 16,
        k: int = 4
    ):
        """
        Args:
            dataset: датасет с атрибутом id_to_indices
            p: количество идентичностей в батче
            k: количество изображений каждой идентичности
        """
        self.dataset = dataset
        self.p = p
        self.k = k

        self.id_to_indices = dataset.id_to_indices
        self.num_identities = len(self.id_to_indices)

        # Фильтруем идентичности с >= k изображений
        self.valid_ids = [
            pid for pid, indices in self.id_to_indices.items()
            if len(indices) >= k
        ]

        if len(self.valid_ids) < p:
            raise ValueError(
                f"Not enough identities with >= {k} images. "
                f"Found {len(self.valid_ids)}, need {p}."
            )

        # Количество батчей за эпоху
        self.num_batches = len(self.valid_ids) // p

    def __iter__(self):
        # Перемешиваем идентичности
        shuffled_ids = np.random.permutation(self.valid_ids)

        for batch_idx in range(self.num_batches):
            batch_ids = shuffled_ids[batch_idx * self.p:(batch_idx + 1) * self.p]
            batch_indices = []

            for pid in batch_ids:
                indices = self.id_to_indices[pid]
                # Выбираем K изображений (с повторением, если нужно)
                if len(indices) >= self.k:
                    selected = np.random.choice(indices, self.k, replace=False)
                else:
                    selected = np.random.choice(indices, self.k, replace=True)
                batch_indices.extend(selected.tolist())

            yield batch_indices

    def __len__(self):
        return self.num_batches


class RandomIdentitySampler(Sampler):
    """
    Альтернативный семплер: выбирает случайные идентичности
    и фиксированное количество изображений каждой.
    """

    def __init__(self, dataset, num_instances=4):
        self.dataset = dataset
        self.num_instances = num_instances
        self.id_to_indices = dataset.id_to_indices
        self.pids = list(self.id_to_indices.keys())

    def __iter__(self):
        indices = []
        pids = np.random.permutation(self.pids)

        for pid in pids:
            idxs = self.id_to_indices[pid]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, self.num_instances, replace=True)
            else:
                idxs = np.random.choice(idxs, self.num_instances, replace=False)
            indices.extend(idxs)

        return iter(indices)

    def __len__(self):
        return len(self.pids) * self.num_instances


# Demo dataset для тестирования без реальных данных
class DemoVehicleDataset(Dataset):
    """
    Демонстрационный датасет для тестирования пайплайна.
    Генерирует синтетические данные.
    """

    def __init__(
        self,
        num_ids: int = 100,
        images_per_id: int = 10,
        num_cameras: int = 5,
        transform=None,
        split: str = 'train'
    ):
        self.num_ids = num_ids
        self.images_per_id = images_per_id
        self.num_cameras = num_cameras
        self.transform = transform
        self.split = split

        # Генерируем данные
        self.data = []
        for vid in range(num_ids):
            for img_idx in range(images_per_id):
                cam_id = img_idx % num_cameras
                self.data.append((vid, cam_id))

        self.num_classes = num_ids
        self.id_to_label = {i: i for i in range(num_ids)}

        # Группировка для PK-семплинга
        self.id_to_indices = defaultdict(list)
        for idx, (vid, _) in enumerate(self.data):
            self.id_to_indices[vid].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid, cam_id = self.data[idx]

        # Генерируем случайное изображение (для демо)
        # В реальности здесь загрузка с диска
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        )

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': vid,
            'camera_id': cam_id,
            'path': f'demo_{vid}_{cam_id}_{idx}.jpg'
        }


# Тест датасетов
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Тест демо-датасета
    demo_dataset = DemoVehicleDataset(
        num_ids=50, images_per_id=8, transform=transform
    )
    print(f"Demo dataset: {len(demo_dataset)} images, {demo_dataset.num_classes} IDs")

    # Тест PK-семплера
    sampler = PKSampler(demo_dataset, p=8, k=4)
    print(f"PK Sampler: {len(sampler)} batches per epoch")

    # Тест одного батча
    for batch_indices in sampler:
        print(f"Batch size: {len(batch_indices)}")
        labels = [demo_dataset.data[i][0] for i in batch_indices]
        unique_labels = set(labels)
        print(f"Unique IDs in batch: {len(unique_labels)}")
        break
