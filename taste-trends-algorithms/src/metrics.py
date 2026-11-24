"""Evaluation metrics (pure NumPy/pandas so they can be unit-tested exactly).

Accuracy
  * RMSE            - typical size of the error in predicted stars.
Ranking (top-k lists, k = 10)
  * Precision@k     - of the k movies we recommended, what fraction did the user
                      actually like (rated >= 4 in the hidden test set)?
  * Recall@k        - of all the movies the user liked in the test set, what
                      fraction appeared in our k recommendations?
  * HitRate@k       - fraction of users with at least one liked movie in the list.
  * NDCG@k          - like precision, but a hit at position 1 counts more than at 10.
Beyond accuracy
  * Catalogue coverage  - what share of all movies is ever recommended to anyone.
  * Popularity bias     - average popularity of recommended movies (ARP), share of
                          recommendations from the "Head" tier, Gini of exposure.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def rmse(predictions: Iterable[float], actual: Iterable[float]) -> float:
    p, a = np.asarray(list(predictions), dtype=float), np.asarray(list(actual), dtype=float)
    if len(p) == 0 or len(p) != len(a):
        raise ValueError("predictions and actual must be non-empty and the same length")
    return float(np.sqrt(np.mean((p - a) ** 2)))


def precision_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """Hits in the top-k divided by k (the list is judged on k slots even if shorter)."""
    top = list(recommended)[:k]
    return sum(1 for item in top if item in relevant) / k


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = list(recommended)[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    top = list(recommended)[:k]
    dcg = sum(1.0 / np.log2(pos + 2) for pos, item in enumerate(top) if item in relevant)
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def ranking_metrics(recs: Mapping[int, Iterable[int]], relevant: Mapping[int, set[int]], k: int) -> dict[str, float]:
    """Average the per-user ranking metrics over users that have >= 1 relevant item."""
    users = [u for u in recs if relevant.get(u)]
    if not users:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "hit_rate_at_k": 0.0, "users": 0}
    prec, rec, ndcg, hit = [], [], [], []
    for u in users:
        r = list(recs[u])
        rel = relevant[u]
        prec.append(precision_at_k(r, rel, k))
        rec.append(recall_at_k(r, rel, k))
        ndcg.append(ndcg_at_k(r, rel, k))
        hit.append(1.0 if any(i in rel for i in r[:k]) else 0.0)
    return {"precision_at_k": float(np.mean(prec)), "recall_at_k": float(np.mean(rec)),
            "ndcg_at_k": float(np.mean(ndcg)), "hit_rate_at_k": float(np.mean(hit)), "users": len(users)}


def catalogue_coverage(recs: Mapping[int, Iterable[int]], catalogue_size: int) -> float:
    """Distinct recommended items / items available for recommendation."""
    if catalogue_size <= 0:
        raise ValueError("catalogue_size must be positive")
    distinct = set()
    for items in recs.values():
        distinct.update(int(i) for i in items)
    return len(distinct) / catalogue_size


def gini(values: Iterable[float]) -> float:
    """Gini coefficient: 0 = perfectly equal, 1 = everything concentrated on one item."""
    x = np.sort(np.asarray(list(values), dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def exposure_counts(recs: Mapping[int, Iterable[int]], catalogue: Iterable[int] | None = None) -> pd.Series:
    """How many times each item appears across all recommendation lists.

    If ``catalogue`` is given, never-recommended items are included with 0 so the
    Gini reflects inequality over the *whole* catalogue.
    """
    counts: dict[int, int] = {}
    for items in recs.values():
        for i in items:
            counts[int(i)] = counts.get(int(i), 0) + 1
    series = pd.Series(counts, dtype=float)
    if catalogue is not None:
        series = series.reindex(list(int(i) for i in catalogue), fill_value=0.0)
    return series


def popularity_bias(recs: Mapping[int, Iterable[int]], item_popularity: Mapping[int, float],
                    item_tier: Mapping[int, str], catalogue: Iterable[int]) -> dict[str, float]:
    """Popularity-bias summary of a set of recommendation lists.

    * ``arp``        - Average Recommendation Popularity: mean number of training
                       ratings of the recommended movies.
    * ``head_share`` / ``mid_share`` / ``tail_share`` - share of all recommended
                       slots that come from each popularity tier.
    * ``gini_exposure`` - Gini coefficient of how often each catalogue item is shown.
    """
    flat = [int(i) for items in recs.values() for i in items]
    if not flat:
        return {"arp": 0.0, "median_rec_popularity": 0.0, "head_share": 0.0, "mid_share": 0.0,
                "tail_share": 0.0, "gini_exposure": 0.0}
    pops = np.array([item_popularity.get(i, 0.0) for i in flat], dtype=float)
    tiers = pd.Series([item_tier.get(i, "Tail") for i in flat])
    share = tiers.value_counts(normalize=True)
    return {
        "arp": float(pops.mean()),
        "median_rec_popularity": float(np.median(pops)),
        "head_share": float(share.get("Head", 0.0)),
        "mid_share": float(share.get("Mid", 0.0)),
        "tail_share": float(share.get("Tail", 0.0)),
        "gini_exposure": gini(exposure_counts(recs, catalogue).values),
    }


def assign_tiers(item_ids: Iterable[int], counts: Iterable[float],
                 head: float = 0.5, mid: float = 0.8) -> dict[int, str]:
    """Head / Mid / Tail by cumulative share of ratings (same rule as the Spark SQL).

    Items are sorted by popularity; an item is *Head* if the more popular items
    before it hold < 50% of all ratings, *Mid* if they hold < 80%, else *Tail*.
    """
    ids = np.asarray(list(item_ids))
    c = np.asarray(list(counts), dtype=float)
    order = np.lexsort((ids, -c))
    ids, c = ids[order], c[order]
    before = (np.cumsum(c) - c) / c.sum()
    labels = np.where(before < head, "Head", np.where(before < mid, "Mid", "Tail"))
    return {int(i): str(t) for i, t in zip(ids, labels)}


def lorenz_points(recs: Mapping[int, Iterable[int]], points: int = 100) -> pd.DataFrame:
    """Cumulative share of recommendation slots taken by the most-exposed items."""
    counts = np.sort(exposure_counts(recs).values)[::-1]
    if counts.sum() == 0:
        return pd.DataFrame({"item_share": [0.0], "exposure_share": [0.0]})
    cum = np.cumsum(counts) / counts.sum()
    idx = np.unique(np.linspace(0, len(counts) - 1, min(points, len(counts))).astype(int))
    return pd.DataFrame({"item_share": np.r_[0.0, (idx + 1) / len(counts)], "exposure_share": np.r_[0.0, cum[idx]]})
