#!/usr/bin/env python3
"""Reproducible evaluation of the mood-matching recommender (seeded, no GPU/weights).

    python scripts/evaluate.py

Reports ranking metrics (Precision/Recall/MAP/nDCG @k) for the cosine recommender against
**random** and **popularity** baselines, plus label-space statistics. See recsys/eval.py for the
protocols and their (honest) caveats.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from recsys import load_song_database, seed_everything  # noqa: E402
from recsys.eval import (  # noqa: E402
    genre_retrieval_eval,
    label_space_stats,
    leave_one_out_eval,
    text_query_eval,
)

KS = (5, 10)
COLS = [f"{m}@{k}" for k in KS for m in ("P", "R", "MAP", "nDCG")]


def print_block(title: str, res: dict) -> None:
    print(f"\n{title}")
    print("  " + "-" * (18 + 9 * len(COLS)))
    print("  {:<12}".format("method") + "".join(f"{c:>9}" for c in COLS))
    for method in ("cosine", "popularity", "random"):
        row = res[method]
        cells = "".join(f"{row.get(c, 0):>9.3f}" for c in COLS)
        tag = "  ← system" if method == "cosine" else ""
        print("  {:<12}".format(method) + cells + tag)


def main() -> None:
    seed_everything()
    df = load_song_database()
    print(f"Loaded {len(df)} mood-labeled songs.")

    ls = label_space_stats()
    cov = ls["coverage"]
    print("\n== Label space ==")
    print(f"  coverage : {cov['labeled']}/{cov['total_songs']} labeled "
          f"({cov['unlabeled']} unlabeled = {cov['unlabeled_pct']}%)")
    print(f"  density  : {ls['avg_moods_per_song']} moods/song on average (max {ls['max_moods_per_song']})")
    top = list(ls["support"].items())
    print(f"  top moods: " + ", ".join(f"{m}({n})" for m, n in top[:5]))
    print(f"  rarest   : " + ", ".join(f"{m}({n})" for m, n in top[-3:]))
    print(f"  co-occur : " + ", ".join(f"{a}+{b}({n})" for a, b, n in ls["top_cooccurrences"][:3]))

    for ms in (1, 2):
        res = leave_one_out_eval(df, ks=KS, min_shared=ms)
        print_block(f"== Leave-one-out retrieval | relevance = shares >= {ms} mood "
                    f"| n={res['n_queries']} ==", res)

    res = genre_retrieval_eval(df, ks=KS)
    print_block(f"== Genre retrieval (cross-signal, non-circular) | relevance = same Melon genre "
                f"| n={res['n_queries']} ==", res)

    res = text_query_eval(df, ks=KS)
    print_block(f"== End-to-end text queries | n={res['n_queries']} ==", res)
    print("\n(seeded; re-running gives identical numbers. Baselines: popularity = mood-frequency "
          "ranking, random = shuffled.)")


if __name__ == "__main__":
    main()
