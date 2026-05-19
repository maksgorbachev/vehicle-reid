"""
K-Reciprocal Re-ranking for Vehicle ReID
=========================================
Реализация алгоритма re-ranking для улучшения результатов поиска.

Reference: Zhong et al., "Re-ranking Person Re-identification with
k-reciprocal Encoding", CVPR 2017

Основная идея:
- Если A входит в k-NN для B, И B входит в k-NN для A,
  то пара (A, B) считается более надёжной
- Строится новая метрика на основе Jaccard-расстояния
  по k-reciprocal множествам
"""

import numpy as np
from typing import Optional


def compute_euclidean_distance(
    query_features: np.ndarray,
    gallery_features: np.ndarray
) -> np.ndarray:
    """
    Вычисление матрицы евклидовых расстояний.

    Args:
        query_features: [Q, D] признаки запросов
        gallery_features: [G, D] признаки галереи

    Returns:
        dist_mat: [Q, G] матрица расстояний
    """
    # ||q - g||^2 = ||q||^2 + ||g||^2 - 2 * q^T * g
    q_sq = np.sum(query_features ** 2, axis=1, keepdims=True)  # [Q, 1]
    g_sq = np.sum(gallery_features ** 2, axis=1, keepdims=True)  # [G, 1]

    dist = q_sq + g_sq.T - 2 * np.dot(query_features, gallery_features.T)
    dist = np.clip(dist, 0, None)  # Численная стабильность
    dist = np.sqrt(dist)

    return dist


def compute_cosine_distance(
    query_features: np.ndarray,
    gallery_features: np.ndarray
) -> np.ndarray:
    """
    Вычисление матрицы косинусных расстояний.

    Для L2-нормированных векторов: d_cos = 1 - cos_sim = 1 - q^T * g
    """
    # Нормализация
    query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-12)
    gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-12)

    similarity = np.dot(query_norm, gallery_norm.T)
    dist = 1 - similarity

    return dist


def k_reciprocal_neigh(
    initial_rank: np.ndarray,
    i: int,
    k1: int
) -> np.ndarray:
    """
    Вычисление k-reciprocal соседей для элемента i.

    k-reciprocal neighbors: те элементы j, для которых:
    - j входит в top-k1 соседей для i
    - i входит в top-k1 соседей для j

    Args:
        initial_rank: [N, N] матрица индексов отсортированных соседей
        i: индекс элемента
        k1: размер окрестности

    Returns:
        k_reciprocal: индексы k-reciprocal соседей
    """
    # Top-k1 соседи для i
    forward_k_neigh = initial_rank[i, :k1 + 1]

    # Проверяем взаимность
    backward_k_neigh = initial_rank[forward_k_neigh, :k1 + 1]

    # i должен быть среди k1 соседей каждого из forward соседей
    reciprocal_mask = np.any(backward_k_neigh == i, axis=1)
    k_reciprocal = forward_k_neigh[reciprocal_mask]

    return k_reciprocal


def re_ranking(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3
) -> np.ndarray:
    """
    K-reciprocal re-ranking.

    Алгоритм:
    1. Вычисляем начальные расстояния
    2. Для каждого элемента находим k-reciprocal соседей
    3. Расширяем множество соседей (1/2 * k1 reciprocal expansion)
    4. Строим Jaccard-расстояние на основе расширенных множеств
    5. Комбинируем с исходным расстоянием

    Args:
        query_features: [Q, D] признаки запросов
        gallery_features: [G, D] признаки галереи
        k1: размер окрестности для k-reciprocal
        k2: размер для локального расширения
        lambda_value: вес исходного расстояния в комбинации

    Returns:
        final_dist: [Q, G] улучшенная матрица расстояний
    """
    query_num = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    all_num = query_num + gallery_num

    # Объединяем все признаки для построения графа соседства
    all_features = np.concatenate([query_features, gallery_features], axis=0)

    # Вычисляем матрицу расстояний для всех пар
    original_dist = compute_euclidean_distance(all_features, all_features)

    # Сортируем по расстояниям для получения рангов
    initial_rank = np.argsort(original_dist, axis=1)

    # Инициализируем матрицу для Jaccard-расстояний
    V = np.zeros((all_num, all_num), dtype=np.float32)

    print(f"Re-ranking: computing k-reciprocal sets for {all_num} samples...")

    for i in range(all_num):
        # Получаем k-reciprocal соседей
        k_reciprocal = k_reciprocal_neigh(initial_rank, i, k1)

        # Расширение: добавляем 1/2 * k1 reciprocal соседей соседей
        k_reciprocal_expansion = k_reciprocal.copy()

        for j in k_reciprocal:
            candidate = k_reciprocal_neigh(initial_rank, j, int(np.round(k1 / 2)))
            # Проверяем, что больше половины candidate уже в k_reciprocal
            if len(np.intersect1d(candidate, k_reciprocal)) > 2/3 * len(candidate):
                k_reciprocal_expansion = np.union1d(k_reciprocal_expansion, candidate)

        # Вычисляем веса на основе расстояний (Gaussian kernel)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion])
        V[i, k_reciprocal_expansion] = weight / np.sum(weight)

    # Контекстуальное сглаживание (local query expansion)
    if k2 > 0:
        V_qe = np.zeros_like(V)
        for i in range(all_num):
            V_qe[i] = np.mean(V[initial_rank[i, :k2 + 1]], axis=0)
        V = V_qe

    # Jaccard distance: d_J(i,j) = 1 - |V_i ∩ V_j| / |V_i ∪ V_j|
    # Аппроксимация через min/max
    jaccard_dist = np.zeros((query_num, gallery_num), dtype=np.float32)

    for i in range(query_num):
        # V[i] - это распределение для query i
        # V[query_num + j] - это распределение для gallery j
        v_i = V[i]
        for j in range(gallery_num):
            v_j = V[query_num + j]
            # Jaccard через min-max
            intersection = np.minimum(v_i, v_j).sum()
            union = np.maximum(v_i, v_j).sum()
            if union > 0:
                jaccard_dist[i, j] = 1 - intersection / union
            else:
                jaccard_dist[i, j] = 1.0

    # Комбинируем с исходным расстоянием
    original_dist_qg = original_dist[:query_num, query_num:]
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist_qg

    return final_dist


def re_ranking_fast(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3
) -> np.ndarray:
    """
    Упрощённая и более быстрая версия re-ranking.

    Использует только взаимность k-NN без полного Jaccard.
    Подходит для больших галерей.
    """
    # Начальные расстояния
    dist = compute_euclidean_distance(query_features, gallery_features)

    # Объединённые признаки
    all_features = np.concatenate([query_features, gallery_features], axis=0)
    all_dist = compute_euclidean_distance(all_features, all_features)

    query_num = len(query_features)
    gallery_num = len(gallery_features)

    # Ранги
    initial_rank = np.argsort(all_dist, axis=1)

    # Бонус за взаимное соседство
    bonus = np.zeros_like(dist)

    for i in range(query_num):
        # k-NN запроса в галерее
        q_neighbors = initial_rank[i, 1:k1 + 1]  # Пропускаем себя
        q_neighbors = q_neighbors[q_neighbors >= query_num] - query_num  # Только галерея

        for j in q_neighbors:
            # k-NN элемента галереи
            g_neighbors = initial_rank[query_num + j, 1:k1 + 1]
            # Проверяем взаимность
            if i in g_neighbors:
                bonus[i, j] -= 0.1  # Уменьшаем расстояние

    # Local Query Expansion: усредняем с соседями
    if k2 > 0:
        top_k2 = np.argsort(dist, axis=1)[:, :k2]
        dist_expanded = np.zeros_like(dist)
        for i in range(query_num):
            neighbor_dists = dist[i, top_k2[i]]
            weights = np.exp(-neighbor_dists)
            weights = weights / weights.sum()
            for k, j in enumerate(top_k2[i]):
                dist_expanded[i] += weights[k] * dist[i]
        dist = (dist + dist_expanded) / 2

    final_dist = dist + bonus

    return final_dist


# Тест re-ranking
if __name__ == "__main__":
    # Синтетические данные
    np.random.seed(42)

    query_features = np.random.randn(10, 256).astype(np.float32)
    gallery_features = np.random.randn(100, 256).astype(np.float32)

    # Нормализуем
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)

    # Тест базовых расстояний
    euclidean_dist = compute_euclidean_distance(query_features, gallery_features)
    cosine_dist = compute_cosine_distance(query_features, gallery_features)

    print(f"Euclidean distance shape: {euclidean_dist.shape}")
    print(f"Euclidean distance range: [{euclidean_dist.min():.3f}, {euclidean_dist.max():.3f}]")
    print(f"Cosine distance range: [{cosine_dist.min():.3f}, {cosine_dist.max():.3f}]")

    # Тест быстрого re-ranking
    print("\nTesting fast re-ranking...")
    reranked_dist = re_ranking_fast(
        query_features, gallery_features,
        k1=10, k2=3, lambda_value=0.3
    )
    print(f"Re-ranked distance shape: {reranked_dist.shape}")
    print(f"Re-ranked distance range: [{reranked_dist.min():.3f}, {reranked_dist.max():.3f}]")

    # Сравнение рангов
    original_ranks = np.argsort(euclidean_dist, axis=1)
    reranked_ranks = np.argsort(reranked_dist, axis=1)

    rank_changes = (original_ranks[:, 0] != reranked_ranks[:, 0]).sum()
    print(f"\nTop-1 rank changes after re-ranking: {rank_changes}/{len(query_features)}")
