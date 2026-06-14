"""Hand-computed checks for the ranking metrics."""
import math

import pytest

from recsys.metrics import (
    average_precision_at_k,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_perfect_ranking():
    rels = [1, 1, 1, 0, 0]
    assert precision_at_k(rels, 3) == pytest.approx(1.0)
    assert recall_at_k(rels, 3, n_relevant=3) == pytest.approx(1.0)
    assert average_precision_at_k(rels, 3, n_relevant=3) == pytest.approx(1.0)
    assert ndcg_at_k(rels, 3, n_relevant=3) == pytest.approx(1.0)


def test_precision_recall():
    rels = [0, 1, 0, 1]
    assert precision_at_k(rels, 4) == pytest.approx(0.5)
    assert recall_at_k(rels, 4, n_relevant=2) == pytest.approx(1.0)
    assert recall_at_k(rels, 2, n_relevant=4) == pytest.approx(0.25)


def test_average_precision():
    # AP@4 for [0,1,0,1], R=2: (P@2 + P@4)/2 = (0.5 + 0.5)/2 = 0.5
    assert average_precision_at_k([0, 1, 0, 1], 4, n_relevant=2) == pytest.approx(0.5)


def test_dcg_and_ndcg():
    rels = [0, 1, 0, 1]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)          # positions 2 and 4
    idcg = 1 / math.log2(2) + 1 / math.log2(3)          # ideal: 2 hits up front
    assert dcg_at_k(rels, 4) == pytest.approx(dcg)
    assert ndcg_at_k(rels, 4, n_relevant=2) == pytest.approx(dcg / idcg)


def test_empty_relevance_is_zero():
    rels = [0, 0, 0]
    assert recall_at_k(rels, 3, n_relevant=0) == 0.0
    assert average_precision_at_k(rels, 3, n_relevant=0) == 0.0
    assert ndcg_at_k(rels, 3, n_relevant=0) == 0.0
