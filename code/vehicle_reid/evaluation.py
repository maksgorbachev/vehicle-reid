"""
Evaluation Metrics for Vehicle ReID
====================================
Реализация метрик CMC (Cumulative Matching Characteristic)
и mAP (mean Average Precision) для оценки качества реидентификации.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


def compute_ap(
    index: np.ndarray,
    good_index: np.ndarray,
    junk_index: Optional[np.ndarray] = None
) -> float:
    """
    Вычисление Average Precision для одного запроса.

    AP = (1/|R|) * sum_{k: r_k is relevant} P@k

    Args:
        index: отсортированные индексы галереи по расстоянию
        good_index: индексы релевантных элементов (тот же ID)
        junk_index: индексы для исключения (same camera в VeRi-776)

    Returns:
        ap: Average Precision для данного запроса
    """
    if junk_index is None:
        junk_index = np.array([], dtype=np.int64)

    # Создаём маски
    num_gallery = len(index)
    mask = np.isin(index, good_index)  # True для релевантных

    # Исключаем junk
    junk_mask = np.isin(index, junk_index)
    mask[junk_mask] = False

    # Также исключаем из рассмотрения
    valid_mask = ~junk_mask

    # Позиции релевантных среди валидных
    valid_indices = np.where(valid_mask)[0]
    relevant_positions = np.where(mask[valid_mask])[0]

    if len(relevant_positions) == 0:
        return 0.0

    # Вычисляем precision at each recall point
    num_relevant = len(relevant_positions)
    ap = 0.0

    for i, pos in enumerate(relevant_positions):
        # Precision@pos = (i+1) / (pos+1)
        precision_at_k = (i + 1) / (pos + 1)
        ap += precision_at_k

    ap = ap / num_relevant
    return ap


def compute_cmc(
    index: np.ndarray,
    good_index: np.ndarray,
    junk_index: Optional[np.ndarray] = None,
    max_rank: int = 50
) -> np.ndarray:
    """
    Вычисление CMC (Cumulative Matching Characteristic) для одного запроса.

    CMC@k = 1 если хотя бы один релевантный элемент в top-k, иначе 0

    Args:
        index: отсортированные индексы галереи
        good_index: индексы релевантных элементов
        junk_index: индексы для исключения
        max_rank: максимальный ранг для вычисления

    Returns:
        cmc: [max_rank] массив 0/1 для каждого ранга
    """
    if junk_index is None:
        junk_index = np.array([], dtype=np.int64)

    cmc = np.zeros(max_rank, dtype=np.float32)

    # Фильтруем junk
    valid_mask = ~np.isin(index, junk_index)
    filtered_index = index[valid_mask]

    # Находим первое совпадение
    for i, idx in enumerate(filtered_index[:max_rank]):
        if idx in good_index:
            cmc[i:] = 1
            break

    return cmc


def evaluate_reid(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_cameras: Optional[np.ndarray] = None,
    gallery_cameras: Optional[np.ndarray] = None,
    max_rank: int = 50,
    remove_same_camera: bool = True
) -> Dict[str, float]:
    """
    Полная оценка качества ReID.

    Args:
        query_features: [Q, D] признаки запросов
        gallery_features: [G, D] признаки галереи
        query_labels: [Q] метки запросов
        gallery_labels: [G] метки галереи
        query_cameras: [Q] ID камер запросов
        gallery_cameras: [G] ID камер галереи
        max_rank: максимальный ранг для CMC
        remove_same_camera: исключать ли same-camera matches (для VeRi-776)

    Returns:
        dict с метриками: mAP, Rank-1, Rank-5, Rank-10, etc.
    """
    num_query = len(query_labels)

    # Вычисляем матрицу расстояний
    # Для L2-нормированных: ||q-g||^2 = 2 - 2*q^T*g
    # Используем косинусное расстояние
    similarity = np.dot(query_features, gallery_features.T)
    dist_mat = 1 - similarity  # Косинусное расстояние

    # Сортируем по расстоянию (меньше = лучше)
    indices = np.argsort(dist_mat, axis=1)

    all_ap = []
    all_cmc = np.zeros((num_query, max_rank), dtype=np.float32)

    for q_idx in range(num_query):
        q_label = query_labels[q_idx]
        q_cam = query_cameras[q_idx] if query_cameras is not None else -1

        # Находим релевантные элементы (тот же ID)
        good_index = np.where(gallery_labels == q_label)[0]

        # Находим junk (same camera)
        if remove_same_camera and gallery_cameras is not None:
            junk_index = np.where(
                (gallery_labels == q_label) & (gallery_cameras == q_cam)
            )[0]
        else:
            junk_index = np.array([], dtype=np.int64)

        # AP и CMC для этого запроса
        ap = compute_ap(indices[q_idx], good_index, junk_index)
        cmc = compute_cmc(indices[q_idx], good_index, junk_index, max_rank)

        all_ap.append(ap)
        all_cmc[q_idx] = cmc

    # Агрегируем метрики
    mAP = np.mean(all_ap) * 100
    cmc_scores = np.mean(all_cmc, axis=0) * 100

    results = {
        'mAP': mAP,
        'Rank-1': cmc_scores[0],
        'Rank-5': cmc_scores[4] if max_rank >= 5 else 0,
        'Rank-10': cmc_scores[9] if max_rank >= 10 else 0,
        'Rank-20': cmc_scores[19] if max_rank >= 20 else 0,
    }

    return results


def evaluate_with_reranking(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_cameras: Optional[np.ndarray] = None,
    gallery_cameras: Optional[np.ndarray] = None,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    max_rank: int = 50,
    remove_same_camera: bool = True
) -> Dict[str, float]:
    """
    Оценка с применением k-reciprocal re-ranking.
    """
    from .reranking import re_ranking

    # Применяем re-ranking
    dist_mat = re_ranking(
        query_features, gallery_features,
        k1=k1, k2=k2, lambda_value=lambda_value
    )

    num_query = len(query_labels)
    indices = np.argsort(dist_mat, axis=1)

    all_ap = []
    all_cmc = np.zeros((num_query, max_rank), dtype=np.float32)

    for q_idx in range(num_query):
        q_label = query_labels[q_idx]
        q_cam = query_cameras[q_idx] if query_cameras is not None else -1

        good_index = np.where(gallery_labels == q_label)[0]

        if remove_same_camera and gallery_cameras is not None:
            junk_index = np.where(
                (gallery_labels == q_label) & (gallery_cameras == q_cam)
            )[0]
        else:
            junk_index = np.array([], dtype=np.int64)

        ap = compute_ap(indices[q_idx], good_index, junk_index)
        cmc = compute_cmc(indices[q_idx], good_index, junk_index, max_rank)

        all_ap.append(ap)
        all_cmc[q_idx] = cmc

    mAP = np.mean(all_ap) * 100
    cmc_scores = np.mean(all_cmc, axis=0) * 100

    results = {
        'mAP': mAP,
        'Rank-1': cmc_scores[0],
        'Rank-5': cmc_scores[4] if max_rank >= 5 else 0,
        'Rank-10': cmc_scores[9] if max_rank >= 10 else 0,
        'Rank-20': cmc_scores[19] if max_rank >= 20 else 0,
    }

    return results


def compute_mahalanobis_dist_matrix(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    train_features: np.ndarray,
    reg: float = 1e-5
) -> np.ndarray:
    """
    Матрица расстояний Махаланобиса между запросами и галереей.

    d(x, y) = sqrt((x-y)^T Σ^{-1} (x-y))

    Ковариационная матрица Σ оценивается по тренировочным эмбеддингам.
    reg — регуляризация для устойчивости обращения матрицы.

    Args:
        query_features:   [Q, D]
        gallery_features: [G, D]
        train_features:   [N, D] — для оценки Σ
        reg: ridge-регуляризация

    Returns:
        dist_mat: [Q, G]
    """
    D = train_features.shape[1]

    # Оценка ковариации + ridge-регуляризация
    cov = np.cov(train_features, rowvar=False)  # [D, D]
    cov += reg * np.eye(D)
    cov_inv = np.linalg.inv(cov)  # [D, D]

    # Векторизованное вычисление: d(q,g)^2 = (q-g)^T Σ^{-1} (q-g)
    # Для каждой пары (q, g): diff = q - g
    # Но Q*G*D матрица не влезет в память при Q=1678, G=11579, D=2048
    # Считаем батчами по запросам
    Q = query_features.shape[0]
    G = gallery_features.shape[0]
    dist_mat = np.zeros((Q, G), dtype=np.float32)

    for i in range(Q):
        diff = gallery_features - query_features[i]  # [G, D]
        # diff @ cov_inv @ diff.T diagonal
        tmp = diff @ cov_inv  # [G, D]
        dist_mat[i] = np.einsum('gd,gd->g', tmp, diff)  # [G]

    dist_mat = np.sqrt(np.maximum(dist_mat, 0))
    return dist_mat


def evaluate_with_mahalanobis(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    train_features: np.ndarray,
    query_cameras: Optional[np.ndarray] = None,
    gallery_cameras: Optional[np.ndarray] = None,
    reg: float = 1e-5,
    max_rank: int = 50,
    remove_same_camera: bool = True
) -> Dict[str, float]:
    """
    Оценка ReID с расстоянием Махаланобиса вместо L2/косинусного.

    Args:
        train_features: [N, D] эмбеддинги тренировочной галереи для оценки Σ
        reg: регуляризация ковариационной матрицы
    """
    dist_mat = compute_mahalanobis_dist_matrix(
        query_features, gallery_features, train_features, reg=reg
    )

    num_query = len(query_labels)
    indices = np.argsort(dist_mat, axis=1)

    all_ap = []
    all_cmc = np.zeros((num_query, max_rank), dtype=np.float32)

    for q_idx in range(num_query):
        q_label = query_labels[q_idx]
        q_cam = query_cameras[q_idx] if query_cameras is not None else -1

        good_index = np.where(gallery_labels == q_label)[0]

        if remove_same_camera and gallery_cameras is not None:
            junk_index = np.where(
                (gallery_labels == q_label) & (gallery_cameras == q_cam)
            )[0]
        else:
            junk_index = np.array([], dtype=np.int64)

        ap = compute_ap(indices[q_idx], good_index, junk_index)
        cmc = compute_cmc(indices[q_idx], good_index, junk_index, max_rank)

        all_ap.append(ap)
        all_cmc[q_idx] = cmc

    mAP = np.mean(all_ap) * 100
    cmc_scores = np.mean(all_cmc, axis=0) * 100

    return {
        'mAP': mAP,
        'Rank-1': cmc_scores[0],
        'Rank-5': cmc_scores[4] if max_rank >= 5 else 0,
        'Rank-10': cmc_scores[9] if max_rank >= 10 else 0,
        'Rank-20': cmc_scores[19] if max_rank >= 20 else 0,
    }


class ReIDEvaluator:
    """
    Класс для удобной оценки моделей ReID.
    """

    def __init__(
        self,
        max_rank: int = 50,
        remove_same_camera: bool = True
    ):
        self.max_rank = max_rank
        self.remove_same_camera = remove_same_camera

    def evaluate(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        query_labels: np.ndarray,
        gallery_labels: np.ndarray,
        query_cameras: Optional[np.ndarray] = None,
        gallery_cameras: Optional[np.ndarray] = None,
        reranking: bool = False,
        mahalanobis: bool = False,
        train_features: Optional[np.ndarray] = None,
        mahal_reg: float = 1e-5,
        **rerank_kwargs
    ) -> Dict[str, float]:
        """
        Оценка с опциональным re-ranking или расстоянием Махаланобиса.

        Args:
            mahalanobis: использовать расстояние Махаланобиса вместо L2
            train_features: [N, D] — обязателен при mahalanobis=True
            mahal_reg: ridge-регуляризация ковариационной матрицы
        """
        if mahalanobis:
            if train_features is None:
                raise ValueError("train_features required for mahalanobis=True")
            return evaluate_with_mahalanobis(
                query_features, gallery_features,
                query_labels, gallery_labels,
                train_features=train_features,
                query_cameras=query_cameras,
                gallery_cameras=gallery_cameras,
                reg=mahal_reg,
                max_rank=self.max_rank,
                remove_same_camera=self.remove_same_camera
            )
        elif reranking:
            return evaluate_with_reranking(
                query_features, gallery_features,
                query_labels, gallery_labels,
                query_cameras, gallery_cameras,
                max_rank=self.max_rank,
                remove_same_camera=self.remove_same_camera,
                **rerank_kwargs
            )
        else:
            return evaluate_reid(
                query_features, gallery_features,
                query_labels, gallery_labels,
                query_cameras, gallery_cameras,
                max_rank=self.max_rank,
                remove_same_camera=self.remove_same_camera
            )

    def print_results(self, results: Dict[str, float], title: str = "Results"):
        """Красивый вывод результатов."""
        print(f"\n{'='*50}")
        print(f" {title}")
        print(f"{'='*50}")
        print(f" mAP:     {results['mAP']:.2f}%")
        print(f" Rank-1:  {results['Rank-1']:.2f}%")
        print(f" Rank-5:  {results['Rank-5']:.2f}%")
        print(f" Rank-10: {results['Rank-10']:.2f}%")
        print(f"{'='*50}\n")


# Тест метрик
if __name__ == "__main__":
    np.random.seed(42)

    # Симулируем данные
    num_query = 50
    num_gallery = 200
    num_ids = 30
    feature_dim = 256

    # Генерируем признаки (кластеризованные по ID)
    query_labels = np.random.randint(0, num_ids, num_query)
    gallery_labels = np.random.randint(0, num_ids, num_gallery)
    query_cameras = np.random.randint(0, 5, num_query)
    gallery_cameras = np.random.randint(0, 5, num_gallery)

    # Создаём признаки с кластерной структурой
    id_centers = np.random.randn(num_ids, feature_dim)

    query_features = id_centers[query_labels] + np.random.randn(num_query, feature_dim) * 0.3
    gallery_features = id_centers[gallery_labels] + np.random.randn(num_gallery, feature_dim) * 0.3

    # L2-нормализация
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)

    # Оценка
    evaluator = ReIDEvaluator(max_rank=50, remove_same_camera=True)

    # Без re-ranking
    results = evaluator.evaluate(
        query_features, gallery_features,
        query_labels, gallery_labels,
        query_cameras, gallery_cameras,
        reranking=False
    )
    evaluator.print_results(results, "Without Re-ranking")

    # С re-ranking
    results_rr = evaluator.evaluate(
        query_features, gallery_features,
        query_labels, gallery_labels,
        query_cameras, gallery_cameras,
        reranking=True,
        k1=20, k2=6, lambda_value=0.3
    )
    evaluator.print_results(results_rr, "With Re-ranking")

    # Улучшение
    print(f"mAP improvement: +{results_rr['mAP'] - results['mAP']:.2f}%")
    print(f"Rank-1 improvement: +{results_rr['Rank-1'] - results['Rank-1']:.2f}%")
