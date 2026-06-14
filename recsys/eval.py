"""Reproducible, seeded evaluation of the mood-matching recommender.

Two protocols, both with **random** and **popularity** baselines for context:

  1. leave_one_out_eval  — each labeled song is a query; relevance = shares >= `min_shared` moods
                           with the query. Measures whether cosine over the mood space retrieves
                           mood-consistent songs. (Honest caveat: cosine and the relevance signal
                           both derive from the mood vectors, so this measures internal retrieval
                           *consistency* + the lift over baselines — not human-judged quality.)
  2. text_query_eval     — free-text queries -> (mock BERT) mood vector -> ranking; relevance =
                           song has >= 1 of the query's target moods. An end-to-end sanity check.

Plus label_space_stats() for honest data-quality reporting (support, coverage, co-occurrence).
"""
from __future__ import annotations

import numpy as np

from .config import MOODS, SEED
from .data import coverage_stats, load_song_database, mood_matrix
from .metrics import evaluate_ranking
from .recommend import cosine_scores, text_to_mood

DEFAULT_QUERIES = [
    "When you got rejected by a girl",
    "When you want to go to sleep",
    "I need to get motivated to exercise",
    "rainy day study session",
    "i just want to dance at a party",
    "feeling nostalgic about old summers",
    "songs for a romantic dinner date",
    "angry high-energy gym session",
]


def _mean(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}


def leave_one_out_eval(df=None, ks=(5, 10), min_shared: int = 1, seed: int = SEED) -> dict:
    df = load_song_database() if df is None else df
    B = mood_matrix(df)
    n = B.shape[0]

    N = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    sims = N @ N.T
    np.fill_diagonal(sims, -np.inf)

    R = (B @ B.T) >= min_shared
    np.fill_diagonal(R, False)
    n_rel = R.sum(axis=1)

    pop = B @ B.mean(axis=0)                       # query-independent popularity score
    pop_order = np.argsort(-pop, kind="stable")
    rng = np.random.default_rng(seed)

    cos, popb, rnd = [], [], []
    for i in range(n):
        if n_rel[i] == 0:
            continue
        order = np.argsort(-sims[i], kind="stable")
        order = order[order != i]
        cos.append(evaluate_ranking(R[i, order], int(n_rel[i]), ks))
        po = pop_order[pop_order != i]
        popb.append(evaluate_ranking(R[i, po], int(n_rel[i]), ks))
        perm = rng.permutation(n)
        perm = perm[perm != i]
        rnd.append(evaluate_ranking(R[i, perm], int(n_rel[i]), ks))

    return {"protocol": "leave_one_out", "min_shared": min_shared, "n_queries": len(cos),
            "cosine": _mean(cos), "popularity": _mean(popb), "random": _mean(rnd)}


def genre_retrieval_eval(df=None, ks=(5, 10), seed: int = SEED) -> dict:
    """Cross-signal check: rank by mood cosine, but relevance = **same Melon genre** (an
    independent label the system never sees). Tests whether mood similarity recovers genre.
    Unlike the mood-overlap protocols this is *not* circular, so it's the most meaningful number.
    """
    df = load_song_database() if df is None else df
    B = mood_matrix(df)
    n = B.shape[0]
    genres = df["Genre"].astype(str).to_numpy()

    N = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    sims = N @ N.T
    np.fill_diagonal(sims, -np.inf)
    pop_order = np.argsort(-(B @ B.mean(axis=0)), kind="stable")
    rng = np.random.default_rng(seed)

    cos, popb, rnd = [], [], []
    for i in range(n):
        rel = genres == genres[i]
        rel[i] = False
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        order = np.argsort(-sims[i], kind="stable")
        order = order[order != i]
        cos.append(evaluate_ranking(rel[order], n_rel, ks))
        po = pop_order[pop_order != i]
        popb.append(evaluate_ranking(rel[po], n_rel, ks))
        perm = rng.permutation(n)
        perm = perm[perm != i]
        rnd.append(evaluate_ranking(rel[perm], n_rel, ks))

    return {"protocol": "genre_retrieval", "n_queries": len(cos),
            "cosine": _mean(cos), "popularity": _mean(popb), "random": _mean(rnd)}


def text_query_eval(df=None, queries=None, ks=(5, 10), seed: int = SEED) -> dict:
    df = load_song_database() if df is None else df
    queries = DEFAULT_QUERIES if queries is None else queries
    B = mood_matrix(df)
    n = B.shape[0]
    pop_order = np.argsort(-(B @ B.mean(axis=0)), kind="stable")
    rng = np.random.default_rng(seed)

    cos, popb, rnd = [], [], []
    for q in queries:
        uv = text_to_mood(q)
        target = uv > 0
        rel = B[:, target].sum(axis=1) > 0
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        order = np.argsort(-cosine_scores(uv, B), kind="stable")
        cos.append(evaluate_ranking(rel[order], n_rel, ks))
        popb.append(evaluate_ranking(rel[pop_order], n_rel, ks))
        rnd.append(evaluate_ranking(rel[rng.permutation(n)], n_rel, ks))

    return {"protocol": "text_query", "n_queries": len(cos),
            "cosine": _mean(cos), "popularity": _mean(popb), "random": _mean(rnd)}


def label_space_stats(csv_path=None) -> dict:
    cov = coverage_stats(csv_path)
    df = load_song_database(csv_path)
    B = mood_matrix(df)
    support = {m: int(B[:, i].sum()) for i, m in enumerate(MOODS)}
    per_song = B.sum(axis=1)
    co = B.T @ B
    pairs = [(MOODS[i], MOODS[j], int(co[i, j]))
             for i in range(len(MOODS)) for j in range(i + 1, len(MOODS))]
    pairs.sort(key=lambda x: -x[2])
    return {"coverage": cov,
            "avg_moods_per_song": round(float(per_song.mean()), 2),
            "max_moods_per_song": int(per_song.max()),
            "support": dict(sorted(support.items(), key=lambda x: -x[1])),
            "top_cooccurrences": pairs[:5]}
