"""The recommendation core: text -> mood vector -> cosine ranking.

The text->mood encoder here is a transparent keyword stand-in for the project's fine-tuned BERT
(see notebooks/05). The cosine ranking is the real system step, shared by the demo and the evaluator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MOODS

# Keyword -> mood triggers (stand-in for BERT). Substring match against the lowercased query.
MOOD_TRIGGERS: dict[str, list[str]] = {
    "Happy":       ["happy", "joy", "fun", "celebrat", "party", "cheer", "smile", "good mood"],
    "Sad":         ["sad", "cry", "rejected", "reject", "breakup", "broke up", "heartbreak",
                    "dumped", "tears", "depress", "down", "blue"],
    "Energized":   ["energy", "energetic", "pumped", "pump", "workout", "exercise", "gym",
                    "run", "hype", "dance", "party", "active"],
    "Relaxed":     ["relax", "chill", "calm", "rest", "unwind", "sleep", "lazy", "lofi", "lo-fi"],
    "Nostalgic":   ["nostalg", "memory", "memories", "throwback", "old days", "childhood", "miss"],
    "Romantic":    ["love", "romantic", "romance", "date", "crush", "valentine", "kiss",
                    "girlfriend", "boyfriend"],
    "Angry":       ["angry", "mad", "rage", "furious", "hate", "pissed"],
    "Focused":     ["focus", "study", "studying", "work", "concentrat", "productive", "deep work"],
    "Inspired":    ["inspir", "motivat", "achieve", "hope", "dream big", "uplift", "encourage"],
    "Melancholic": ["melanchol", "gloomy", "rainy", "wistful", "bittersweet", "rejected", "breakup"],
    "Peaceful":    ["peace", "serene", "quiet", "meditat", "calm", "sleep", "zen"],
    "Anxious":     ["anxious", "nervous", "worried", "stress", "panic", "overwhelm"],
    "Confident":   ["confident", "strong", "boss", "win", "power", "motivat", "fearless"],
    "Dreamy":      ["dream", "sleep", "float", "ethereal", "starry", "night drive"],
    "Lonely":      ["lonely", "alone", "rejected", "empty", "solo", "miss", "by myself"],
}


def text_to_mood(query: str) -> np.ndarray:
    """Mock BERT: keyword match -> 15-d mood vector. Uniform if nothing matches (never all-zero)."""
    q = query.lower()
    vec = np.zeros(len(MOODS), dtype=float)
    for i, mood in enumerate(MOODS):
        for trig in MOOD_TRIGGERS[mood]:
            if trig in q:
                vec[i] += 1.0
    if vec.sum() == 0:
        vec[:] = 1.0
    return vec


def cosine_scores(user_vec: np.ndarray, song_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of `user_vec` against every row of `song_matrix`."""
    eps = 1e-9
    u = user_vec / (np.linalg.norm(user_vec) + eps)
    s = song_matrix / (np.linalg.norm(song_matrix, axis=1, keepdims=True) + eps)
    return s @ u


def cosine_topk(user_vec: np.ndarray, song_matrix: np.ndarray, k: int):
    """Return (indices, scores) of the top-k most similar songs."""
    scores = cosine_scores(user_vec, song_matrix)
    order = np.argsort(-scores, kind="stable")[:k]
    return order, scores[order]


def recommend(query: str, songs: pd.DataFrame, matrix: np.ndarray, k: int = 5) -> list[dict]:
    """High-level: free text -> list of {rank, title, artist, score} dicts."""
    idx, scores = cosine_topk(text_to_mood(query), matrix, k)
    out = []
    for rank, (i, sc) in enumerate(zip(idx, scores), 1):
        row = songs.iloc[int(i)]
        out.append({"rank": rank, "title": row["Title"], "artist": row["Artist"],
                    "score": float(sc)})
    return out
