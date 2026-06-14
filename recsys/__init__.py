"""recsys — the mood-based recommendation core (text → mood → cosine ranking) + evaluation."""
from .config import MOODS, SEED, SONG_MOOD_CSV
from .data import coverage_stats, load_song_database, mood_matrix
from .eval import (
    genre_retrieval_eval,
    label_space_stats,
    leave_one_out_eval,
    text_query_eval,
)
from .recommend import cosine_scores, cosine_topk, recommend, text_to_mood
from .seed import seed_everything

__all__ = [
    "MOODS", "SEED", "SONG_MOOD_CSV",
    "load_song_database", "mood_matrix", "coverage_stats",
    "text_to_mood", "cosine_scores", "cosine_topk", "recommend",
    "leave_one_out_eval", "text_query_eval", "genre_retrieval_eval", "label_space_stats",
    "seed_everything",
]
