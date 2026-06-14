#!/usr/bin/env python3
"""
recommend_demo.py — a runnable example of the FINAL recommendation step.

This demonstrates the heart of the system (mood-vector → cosine similarity → Top-K playlist)
using the REAL song mood database in data/mood_labels/song_mood_15features.csv, with NO heavy
weights, no GPU, and no audio required.

The only thing mocked is the text -> mood step (which the real project does with a fine-tuned
BERT, see notebooks/05_bert_text_to_mood.ipynb). Here it is replaced by a tiny, transparent
keyword encoder so the demo is fully self-contained and reproducible.

Usage:
    python scripts/recommend_demo.py                       # runs three example queries
    python scripts/recommend_demo.py --query "rainy day study session" --k 5
    python scripts/recommend_demo.py -q "i just want to dance" -k 8

Dependencies: pandas, numpy  (already in requirements.txt)
"""
from __future__ import annotations

import argparse
import html
import os

import numpy as np
import pandas as pd

# The 15-mood space (order matters — it defines the vector layout).
MOODS = [
    "Happy", "Sad", "Energized", "Relaxed", "Nostalgic", "Romantic", "Angry",
    "Focused", "Inspired", "Melancholic", "Peaceful", "Anxious", "Confident",
    "Dreamy", "Lonely",
]

# Keyword -> mood "encoder": a stand-in for the BERT text->mood model.
# Each mood lists trigger substrings; a query that contains a trigger activates that mood.
MOOD_TRIGGERS: dict[str, list[str]] = {
    "Happy":       ["happy", "joy", "fun", "celebrat", "party", "cheer", "smile", "good mood"],
    "Sad":         ["sad", "cry", "rejected", "reject", "breakup", "broke up", "heartbreak",
                    "dumped", "tears", "depress", "down", "blue"],
    "Energized":   ["energy", "energetic", "pumped", "pump", "workout", "exercise", "gym",
                    "run", "hype", "dance", "party", "active"],
    "Relaxed":     ["relax", "chill", "calm", "rest", "unwind", "sleep", "lazy", "lofi", "lo-fi"],
    "Nostalgic":   ["nostalg", "memory", "memories", "throwback", "old days", "childhood", "miss"],
    "Romantic":    ["love", "romantic", "romance", "date", "crush", "valentine", "kiss", "girlfriend",
                    "boyfriend"],
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
    """Mock of the BERT text->mood model: keyword match -> 15-dim mood vector.

    Returns a unit-ish vector over MOODS. If nothing matches, returns a uniform vector
    (so the demo still produces a result and never divides by zero).
    """
    q = query.lower()
    vec = np.zeros(len(MOODS), dtype=float)
    for i, mood in enumerate(MOODS):
        for trig in MOOD_TRIGGERS[mood]:
            if trig in q:
                vec[i] += 1.0
    if vec.sum() == 0:
        vec[:] = 1.0  # no keyword hit -> neutral
    return vec


def load_song_database(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the real song->mood table; keep only songs that have >=1 mood labeled."""
    df = pd.read_csv(csv_path, index_col=0)
    df[MOODS] = df[MOODS].apply(pd.to_numeric, errors="coerce")
    labeled = df[df[MOODS].fillna(0).sum(axis=1) > 0].copy()
    labeled[MOODS] = labeled[MOODS].fillna(0)
    matrix = labeled[MOODS].to_numpy(dtype=float)
    return labeled.reset_index(drop=True), matrix


def cosine_topk(user_vec: np.ndarray, song_matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity between the user mood vector and every song; return top-k (idx, score)."""
    eps = 1e-9
    u = user_vec / (np.linalg.norm(user_vec) + eps)
    s = song_matrix / (np.linalg.norm(song_matrix, axis=1, keepdims=True) + eps)
    scores = s @ u
    order = np.argsort(-scores)
    return order[:k], scores[order[:k]]


def recommend(query: str, songs: pd.DataFrame, matrix: np.ndarray, k: int = 5) -> None:
    user_vec = text_to_mood(query)
    active = [MOODS[i] for i in np.argsort(-user_vec) if user_vec[i] > 0][:5]
    idx, scores = cosine_topk(user_vec, matrix, k)

    print(f'\n🎧  "{query}"')
    print(f"    detected mood: {', '.join(active) if active else '(neutral)'}")
    print(f"    {'-' * 56}")
    for rank, (i, sc) in enumerate(zip(idx, scores), 1):
        row = songs.iloc[i]
        title = html.unescape(str(row["Title"]))
        artist = html.unescape(str(row["Artist"]))
        print(f"    {rank}. {title} — {artist}   (sim {sc:.2f})")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "..", "data", "mood_labels", "song_mood_15features.csv")

    parser = argparse.ArgumentParser(description="Mood-based playlist recommendation demo.")
    parser.add_argument("-q", "--query", help="A free-text mood/situation.")
    parser.add_argument("-k", "--k", type=int, default=5, help="Number of songs to return.")
    args = parser.parse_args()

    songs, matrix = load_song_database(csv_path)
    print(f"Loaded {len(songs)} mood-labeled songs from the database.")

    if args.query:
        recommend(args.query, songs, matrix, k=args.k)
    else:
        for q in [
            "When you got rejected by a girl",
            "When you want to go to sleep",
            "I need to get motivated to exercise",
        ]:
            recommend(q, songs, matrix, k=args.k)

    print(
        "\nℹ️  text→mood here is a keyword mock; the real project uses fine-tuned BERT "
        "(notebooks/05).\n   The similarity + ranking logic shown is the same as the full system."
    )


if __name__ == "__main__":
    main()
