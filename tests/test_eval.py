"""The evaluation runs, is reproducible, and beats the random baseline."""
from recsys import seed_everything
from recsys.eval import (
    genre_retrieval_eval,
    label_space_stats,
    leave_one_out_eval,
)


def test_leave_one_out_beats_random_and_is_deterministic():
    seed_everything()
    a = leave_one_out_eval(min_shared=2)
    b = leave_one_out_eval(min_shared=2)
    assert a == b                                          # seeded → identical
    assert a["cosine"]["MAP@10"] > a["random"]["MAP@10"]   # system beats random
    assert a["n_queries"] > 500


def test_genre_eval_is_near_chance():
    # honest expectation: mood similarity barely recovers genre (it's a different signal)
    res = genre_retrieval_eval()
    assert res["cosine"]["P@5"] < 0.4                      # nowhere near "good"
    assert res["n_queries"] > 1000


def test_label_space_reports_coverage_gap():
    ls = label_space_stats()
    assert ls["coverage"]["unlabeled"] > 0
    assert 0 < ls["coverage"]["unlabeled_pct"] < 100
    assert len(ls["support"]) == 15
