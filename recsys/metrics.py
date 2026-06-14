"""Ranking metrics for binary relevance. Each takes `rels`: 0/1 array in *ranked* order (best first)."""
from __future__ import annotations

import numpy as np


def precision_at_k(rels, k: int) -> float:
    rels = np.asarray(rels)[:k]
    return float(rels.sum()) / k if k > 0 else 0.0


def recall_at_k(rels, k: int, n_relevant: int) -> float:
    if n_relevant <= 0:
        return 0.0
    return float(np.asarray(rels)[:k].sum()) / n_relevant


def average_precision_at_k(rels, k: int, n_relevant: int) -> float:
    """AP@k = (1 / min(k, R)) * sum_i rel_i * Precision@i, over the top k."""
    rels = np.asarray(rels)[:k]
    if n_relevant <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            hits += 1
            score += hits / i
    return score / min(k, n_relevant)


def dcg_at_k(rels, k: int) -> float:
    rels = np.asarray(rels, dtype=float)[:k]
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float((rels * discounts).sum())


def ndcg_at_k(rels, k: int, n_relevant: int) -> float:
    ideal_hits = min(k, n_relevant)
    if ideal_hits <= 0:
        return 0.0
    idcg = dcg_at_k(np.ones(ideal_hits), k)
    return dcg_at_k(rels, k) / idcg if idcg > 0 else 0.0


def evaluate_ranking(rels, n_relevant: int, ks=(5, 10)) -> dict:
    """All metrics at each k for one ranked list -> flat dict like {'P@5': .., 'nDCG@10': ..}."""
    out = {}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(rels, k)
        out[f"R@{k}"] = recall_at_k(rels, k, n_relevant)
        out[f"MAP@{k}"] = average_precision_at_k(rels, k, n_relevant)
        out[f"nDCG@{k}"] = ndcg_at_k(rels, k, n_relevant)
    return out
