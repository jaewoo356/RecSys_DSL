"""Central configuration — paths resolve from the repo root (no hardcoded Colab paths)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SONG_MOOD_CSV = DATA_DIR / "mood_labels" / "song_mood_15features.csv"

# The 15-mood space (order defines the vector layout).
MOODS: list[str] = [
    "Happy", "Sad", "Energized", "Relaxed", "Nostalgic", "Romantic", "Angry",
    "Focused", "Inspired", "Melancholic", "Peaceful", "Anxious", "Confident",
    "Dreamy", "Lonely",
]

SEED = 42
