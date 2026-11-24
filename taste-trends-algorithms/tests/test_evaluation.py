"""Tests for MODULE F (evaluation calculations). Expected values are worked out by hand."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import metrics as M


def test_rmse_hand_example():
    # errors 1, -1, 2 -> mean square = 2 -> sqrt(2)
    assert M.rmse([4, 2, 5], [3, 3, 3]) == pytest.approx(math.sqrt(2))
    with pytest.raises(ValueError):
        M.rmse([], [])


def test_precision_recall_ndcg_single_user():
    recs = [5, 1, 9, 2, 8, 7, 6, 3, 4, 0]
    relevant = {1, 2, 42}
    assert M.precision_at_k(recs, relevant, 10) == pytest.approx(0.2)        # 2 hits / 10
    assert M.recall_at_k(recs, relevant, 10) == pytest.approx(2 / 3)         # 2 of 3 liked movies found
    dcg = 1 / math.log2(3) + 1 / math.log2(5)                                # hits at positions 2 and 4
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert M.ndcg_at_k(recs, relevant, 10) == pytest.approx(dcg / idcg)


def test_precision_counts_short_lists_against_k():
    assert M.precision_at_k([1], {1}, 10) == pytest.approx(0.1)


def test_ranking_metrics_averages_over_users_with_relevant_items():
    recs = {1: [10, 11], 2: [20, 21], 3: [30, 31]}
    relevant = {1: {10}, 2: {99}, 3: set()}                                  # user 3 ignored
    out = M.ranking_metrics(recs, relevant, 2)
    assert out["users"] == 2
    assert out["precision_at_k"] == pytest.approx((0.5 + 0.0) / 2)
    assert out["recall_at_k"] == pytest.approx((1.0 + 0.0) / 2)
    assert out["hit_rate_at_k"] == pytest.approx(0.5)


def test_catalogue_coverage():
    recs = {1: [1, 2, 3], 2: [2, 3, 4]}
    assert M.catalogue_coverage(recs, 10) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        M.catalogue_coverage(recs, 0)


def test_gini_extremes():
    assert M.gini([5, 5, 5, 5]) == pytest.approx(0.0)
    assert M.gini([0, 0, 0, 10]) == pytest.approx(0.75)                      # (n-1)/n for one non-zero
    assert M.gini([]) == 0.0


def test_assign_tiers_cumulative_rule():
    # counts 50, 30, 10, 5, 5 (total 100). Share held by more popular items before each:
    # 0 -> Head, 0.50 -> Mid, 0.80 -> Tail, 0.90 -> Tail, 0.95 -> Tail
    tiers = M.assign_tiers([1, 2, 3, 4, 5], [50, 30, 10, 5, 5])
    assert tiers == {1: "Head", 2: "Mid", 3: "Tail", 4: "Tail", 5: "Tail"}


def test_popularity_bias_summary():
    pop = {1: 100.0, 2: 10.0, 3: 1.0}
    tiers = {1: "Head", 2: "Mid", 3: "Tail"}
    recs = {7: [1, 2], 8: [1, 3]}
    out = M.popularity_bias(recs, pop, tiers, catalogue=[1, 2, 3, 4])
    assert out["arp"] == pytest.approx((100 + 10 + 100 + 1) / 4)
    assert out["head_share"] == pytest.approx(0.5)
    assert out["mid_share"] == pytest.approx(0.25)
    assert out["tail_share"] == pytest.approx(0.25)
    # exposure counts over catalogue [2, 1, 1, 0]
    assert out["gini_exposure"] == pytest.approx(M.gini([2, 1, 1, 0]))


def test_lorenz_points_monotonic():
    pts = M.lorenz_points({1: [1, 2], 2: [1, 3], 3: [1, 4]})
    assert pts["exposure_share"].is_monotonic_increasing
    assert pts["exposure_share"].iloc[-1] == pytest.approx(1.0)
    assert pts["item_share"].iloc[-1] == pytest.approx(1.0)


def test_results_markdown_reads_real_files_only(tmp_path):
    from src.io_utils import write_json
    from src.report import build_results_markdown

    empty = build_results_markdown(tmp_path, "dev")
    assert "has not been run yet" in empty
    write_json({"rmse": {"als": 0.81, "bias_baseline": 0.86, "global_mean": 1.04}, "test_rows": 100,
                "test_rows_scored": 98, "test_rows_cold_start_dropped": 2, "eval_users": 5,
                "relevance_threshold": 4.0, "user_profile_arp": 10.0, "popularity_lift": {"ALS": 1.5}},
               tmp_path / "metrics.json")
    md = build_results_markdown(tmp_path, "dev")
    assert "0.8100" in md and "98" in md


def test_spark_baselines_on_tiny_data(spark):
    from src.evaluation import baseline_rmses, warm_test

    train = spark.createDataFrame([(1, 1, 4.0), (1, 2, 2.0), (2, 1, 5.0), (2, 2, 3.0)],
                                  "userId INT, movieId INT, rating FLOAT")
    test = spark.createDataFrame([(1, 1, 4.0), (3, 1, 3.0), (1, 9, 3.0)], "userId INT, movieId INT, rating FLOAT")
    warm = warm_test(train, test)
    assert warm.count() == 1                                   # user 3 and movie 9 are cold-start
    out = baseline_rmses(train, warm, damping=0.0)
    # mu = 3.5; global-mean RMSE on the single row (4.0) = 0.5
    assert out["global_mean"] == pytest.approx(0.5)
    # b_i(movie 1) = ((4-3.5)+(5-3.5))/2 = 1.0 ; b_u(user 1) = ((4-3.5-1)+(2-3.5+1))/2 = -0.5
    # prediction = 3.5 - 0.5 + 1.0 = 4.0 -> error 0
    assert out["bias_baseline"] == pytest.approx(0.0, abs=1e-6)
    assert isinstance(pd.DataFrame([out]), pd.DataFrame) and np.isfinite(out["mu"])


def test_update_readme_inserts_between_markers(tmp_path):
    from src.io_utils import write_json
    from src.report import update_readme

    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nSTALE-TEXT\n<!-- RESULTS:END -->\noutro\n", encoding="utf-8")
    assert not update_readme(readme, tmp_path / "none", "dev")          # nothing to insert yet
    res = tmp_path / "res"
    write_json({"rmse": {"als": 0.7777, "bias_baseline": 0.86, "global_mean": 1.04}, "test_rows": 10,
                "test_rows_scored": 10, "test_rows_cold_start_dropped": 0, "eval_users": 3,
                "relevance_threshold": 4.0, "user_profile_arp": 1.0, "popularity_lift": {}}, res / "metrics.json")
    assert update_readme(readme, res, "dev")
    text = readme.read_text(encoding="utf-8")
    assert "0.7777" in text and "STALE-TEXT" not in text and text.startswith("intro") and text.endswith("outro\n")
