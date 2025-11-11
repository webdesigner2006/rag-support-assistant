from __future__ import annotations

import hashlib
import time
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.preprocessing import minmax_scale


def now_ms() -> int:
    return int(time.time() * 1000)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Cosine vectors must have same shape")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_list(xs: List[float]) -> List[float]:
    if not xs:
        return []
    if max(xs) == min(xs):
        return [0.5 for _ in xs]
    return list(minmax_scale(xs))


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def rrf_merge(
    semantic_ids: List[str],
    keyword_ids: List[str],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion. Given two ranked lists of IDs (best to worst),
    return fused ranking with scores.
    """
    ranks: dict[str, float] = {}
    for rank, doc_id in enumerate(semantic_ids):
        ranks[doc_id] = ranks.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(keyword_ids):
        ranks[doc_id] = ranks.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(ranks.items(), key=lambda x: x[1], reverse=True)
