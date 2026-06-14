"""Sanity checks for the data loader and the recommendation core."""
import numpy as np

from recsys import MOODS, load_song_database, mood_matrix
from recsys.recommend import cosine_topk, text_to_mood


def test_database_loads_clean():
    df = load_song_database()
    assert len(df) > 1000
    sub = df[MOODS].to_numpy()
    assert sub.shape[1] == 15
    assert not np.isnan(sub).any()                 # no NaNs after cleaning
    assert set(np.unique(sub)).issubset({0, 1})    # strictly binary


def test_text_to_mood_shape_and_keywords():
    v = text_to_mood("when you got rejected by a girl")
    assert v.shape == (15,)
    assert v.sum() > 0
    active = {MOODS[i] for i in range(15) if v[i] > 0}
    assert {"Sad", "Lonely"} & active              # 'rejected' should fire sad/lonely


def test_no_keyword_is_neutral_not_zero():
    v = text_to_mood("xyzzy qwerty")
    assert v.sum() > 0                             # never all-zero (avoids div-by-zero)


def test_cosine_topk_ranks_self_first():
    # a tiny matrix where one row exactly matches the query
    m = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    idx, scores = cosine_topk(np.array([1, 0, 0], dtype=float), m, k=3)
    assert idx[0] == 0                             # exact match ranks first
    assert scores[0] >= scores[1] >= scores[2]     # sorted descending
