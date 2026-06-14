#!/usr/bin/env python3
"""Runnable example of the FINAL recommendation step: free text → mood → Top-K playlist.

Uses the REAL song mood database (data/mood_labels/song_mood_15features.csv) — no weights, no GPU,
no audio. The only mocked piece is text→mood (the real project uses fine-tuned BERT, notebook 05);
the cosine ranking is the real system step. Logic lives in the `recsys` package.

    python scripts/recommend_demo.py
    python scripts/recommend_demo.py -q "rainy day study session" -k 8
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from recsys import MOODS, load_song_database, mood_matrix, seed_everything  # noqa: E402
from recsys.recommend import cosine_topk, text_to_mood  # noqa: E402


def show(query: str, df, matrix, k: int) -> None:
    uv = text_to_mood(query)
    active = [MOODS[i] for i in np.argsort(-uv) if uv[i] > 0][:5]
    idx, scores = cosine_topk(uv, matrix, k)
    print(f'\n🎧  "{query}"')
    print(f"    detected mood: {', '.join(active) if active else '(neutral)'}")
    print("    " + "-" * 56)
    for rank, (i, sc) in enumerate(zip(idx, scores), 1):
        row = df.iloc[int(i)]
        print(f"    {rank}. {row['Title']} — {row['Artist']}   (sim {sc:.2f})")


def main() -> None:
    seed_everything()
    parser = argparse.ArgumentParser(description="Mood-based playlist recommendation demo.")
    parser.add_argument("-q", "--query", help="A free-text mood/situation.")
    parser.add_argument("-k", "--k", type=int, default=5, help="Number of songs to return.")
    args = parser.parse_args()

    df = load_song_database()
    matrix = mood_matrix(df)
    print(f"Loaded {len(df)} mood-labeled songs from the database.")

    queries = [args.query] if args.query else [
        "When you got rejected by a girl",
        "When you want to go to sleep",
        "I need to get motivated to exercise",
    ]
    for q in queries:
        show(q, df, matrix, k=args.k)

    print("\nℹ️  text→mood here is a keyword mock; the real project uses fine-tuned BERT "
          "(notebooks/05).\n   The similarity + ranking logic shown is the same as the full system.")


if __name__ == "__main__":
    main()
