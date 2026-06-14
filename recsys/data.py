"""Load and clean the song -> mood database that ships with the repo."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MOODS, SONG_MOOD_CSV


def load_song_database(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Return the songs that have >=1 mood labeled, with the 15 mood columns as clean 0/1 ints.

    Titles/artists are HTML-unescaped (the crawl left entities like ``&amp;`` in the data).
    """
    path = Path(csv_path) if csv_path is not None else SONG_MOOD_CSV
    df = pd.read_csv(path, index_col=0)
    df[MOODS] = df[MOODS].apply(pd.to_numeric, errors="coerce")
    labeled = df[df[MOODS].fillna(0).sum(axis=1) > 0].copy()
    labeled[MOODS] = labeled[MOODS].fillna(0).clip(0, 1).astype(int)
    for col in ("Title", "Artist"):
        if col in labeled.columns:
            labeled[col] = labeled[col].map(lambda s: html.unescape(str(s)))
    return labeled.reset_index(drop=True)


def mood_matrix(df: pd.DataFrame) -> np.ndarray:
    """The (n_songs, 15) binary mood matrix as float."""
    return df[MOODS].to_numpy(dtype=float)


def coverage_stats(csv_path: str | Path | None = None) -> dict:
    """Label coverage over the FULL table (labeled + unlabeled), for honest reporting."""
    path = Path(csv_path) if csv_path is not None else SONG_MOOD_CSV
    df = pd.read_csv(path, index_col=0)
    df[MOODS] = df[MOODS].apply(pd.to_numeric, errors="coerce")
    total = len(df)
    labeled_mask = df[MOODS].fillna(0).sum(axis=1) > 0
    n_labeled = int(labeled_mask.sum())
    return {
        "total_songs": total,
        "labeled": n_labeled,
        "unlabeled": total - n_labeled,
        "unlabeled_pct": round(100 * (total - n_labeled) / total, 1),
    }
