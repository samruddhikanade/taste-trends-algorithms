"""Builds a SYNTHETIC set of pipeline output files for dashboard tests.

These numbers come from random toy data (not MovieLens) and exist only so the
dashboard can be exercised in tests. They are written to a temporary folder
and are never shown as project results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import metrics as M
from src.config import CONFIG_DIR
from src.io_utils import write_csv, write_json, write_table
from src.recommender_core import popularity_ranking, recommend_popular, recommend_random, train_als_numpy

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Animation", "Horror"]


def build(tmp: Path, mode: str = "dev") -> Path:
    """Write synthetic outputs under ``tmp`` and return a config file pointing there."""
    rng = np.random.default_rng(0)
    cfg = yaml.safe_load((CONFIG_DIR / "config.example.yaml").read_text(encoding="utf-8"))
    for key in ("raw_dir", "processed_dir", "models_dir", "results_dir", "figures_dir", "spark_tmp_dir"):
        cfg["paths"][key] = str(tmp / key)
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    res = tmp / "results_dir" / mode
    eda = res / "eda"
    proc = tmp / "processed_dir" / mode

    n_users, n_items = 80, 120
    item_ids = np.arange(1, n_items + 1)
    popularity = rng.zipf(1.6, n_items).clip(1, 400)
    rows = []
    for u in range(1, n_users + 1):
        k = rng.integers(8, 30)
        items = rng.choice(item_ids, size=k, replace=False, p=popularity / popularity.sum())
        rows += [(u, int(i), float(rng.integers(1, 11) / 2)) for i in items]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    is_test = rng.random(len(ratings)) < 0.2
    train, test = ratings[~is_test], ratings[is_test]
    model = train_als_numpy(train.userId.values, train.movieId.values, train.rating.values, rank=4, iterations=5)

    counts = ratings.groupby("movieId").agg(num_ratings=("rating", "size"), avg_rating=("rating", "mean"))
    tiers = M.assign_tiers(counts.index, counts["num_ratings"])
    cat = pd.DataFrame({"movieId": item_ids})
    cat["title"] = [f"Synthetic Movie {i} ({1950 + i % 70})" for i in item_ids]
    cat["clean_title"] = [f"Synthetic Movie {i}" for i in item_ids]
    cat["year"] = 1950 + item_ids % 70
    cat["genres"] = ["|".join(rng.choice(GENRES, size=rng.integers(1, 4), replace=False)) for _ in item_ids]
    cat = cat.merge(counts.reset_index(), on="movieId", how="left")
    cat["num_ratings"] = cat["num_ratings"].fillna(0).astype(int)
    cat["weighted_rating"] = cat["avg_rating"]
    cat["popularity_rank"] = cat["num_ratings"].rank(ascending=False, method="first").astype(int)
    cat["popularity_tier"] = cat["movieId"].map(tiers).fillna("Unrated")
    cat["imdbId"] = [f"{i:07d}" for i in item_ids]
    cat["tmdbId"] = item_ids * 10
    cat["top_tags"] = "synthetic, test"
    write_table(cat, proc / "movie_catalogue")

    fac_cols = [f"f{i}" for i in range(model.rank)]
    write_table(pd.concat([pd.DataFrame({"userId": model.user_ids}), pd.DataFrame(model.user_factors, columns=fac_cols)],
                          axis=1), proc / "user_factors")
    write_table(pd.concat([pd.DataFrame({"movieId": model.item_ids}), pd.DataFrame(model.item_factors, columns=fac_cols)],
                          axis=1), proc / "item_factors")
    seen = {int(u): g.movieId.tolist() for u, g in ratings.groupby("userId")}
    rec_rows = []
    for u in model.user_ids:
        top = model.recommend(int(u), 10, exclude=seen.get(int(u), []))
        rec_rows += [(int(u), int(r["rank"]), int(r["movieId"]), float(r["score"])) for _, r in top.iterrows()]
    recs_pdf = pd.DataFrame(rec_rows, columns=["userId", "rank", "movieId", "score"])
    write_table(recs_pdf, proc / "als_recommendations")
    write_table(ratings[ratings.userId <= 20], proc / "demo_user_history")
    write_table(pd.DataFrame({"userId": [], "movieId": []}), proc / "seen_for_short_users")

    # EDA tables
    write_csv(ratings.groupby("rating").size().rename("num_ratings").reset_index()
              .assign(pct=lambda d: 100 * d.num_ratings / d.num_ratings.sum()), eda / "rating_distribution.csv")
    mr = cat.sort_values("num_ratings", ascending=False).head(50)[["movieId", "title", "genres", "year", "num_ratings",
                                                                  "avg_rating"]]
    write_csv(mr, eda / "most_rated.csv")
    write_csv(mr.assign(weighted_rating=mr.avg_rating).sort_values("avg_rating", ascending=False), eda / "top_rated.csv")
    tier_tbl = cat[cat.num_ratings > 0].groupby("popularity_tier").agg(
        num_movies=("movieId", "size"), num_ratings=("num_ratings", "sum"), mean_of_movie_avgs=("avg_rating", "mean"),
        min_ratings_per_movie=("num_ratings", "min"), max_ratings_per_movie=("num_ratings", "max")).reset_index()
    write_csv(tier_tbl, eda / "popularity_tiers.csv")
    c = np.sort(cat.num_ratings.values)[::-1]
    write_csv(pd.DataFrame({"bucket": range(1, len(c) + 1), "movie_share": np.arange(1, len(c) + 1) / len(c),
                            "ratings_share": np.cumsum(c) / c.sum(), "min_ratings_in_bucket": c}),
              eda / "long_tail_curve.csv")
    g = cat.assign(genre=cat.genres.str.split("|")).explode("genre")
    write_csv(g.groupby("genre").agg(num_movies=("movieId", "nunique"), num_ratings=("num_ratings", "sum"),
                                     avg_rating=("avg_rating", "mean"),
                                     avg_ratings_per_movie=("num_ratings", "mean")).reset_index(),
              eda / "genre_stats.csv")
    gd = g.assign(decade=(g.year // 10) * 10).groupby(["decade", "genre"]).size().rename("num_movies").reset_index()
    gd["share"] = gd.num_movies / gd.groupby("decade").num_movies.transform("sum")
    write_csv(gd, eda / "genre_decade_share.csv")
    write_csv(cat.groupby("year").agg(num_movies=("movieId", "size"), num_ratings=("num_ratings", "sum"),
                                      avg_rating=("avg_rating", "mean")).reset_index(), eda / "release_year_stats.csv")
    years = np.arange(1996, 2020)
    write_csv(pd.DataFrame({"rating_year": years, "num_ratings": rng.integers(100, 1000, len(years)),
                            "avg_rating": rng.uniform(3.3, 3.8, len(years)),
                            "active_users": rng.integers(10, 80, len(years)),
                            "movies_rated": rng.integers(10, 100, len(years))}), eda / "ratings_over_time.csv")
    wd = [(d, i + 1, h, int(rng.integers(1, 50))) for i, d in enumerate(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri",
                                                                         "Sat"]) for h in range(24)]
    write_csv(pd.DataFrame(wd, columns=["weekday", "weekday_num", "hour_utc", "num_ratings"]),
              eda / "ratings_by_weekday_hour.csv")
    write_csv(pd.DataFrame({"bucket": ["01-19", "20-49"], "sort_key": [8, 20], "num_users": [60, 20],
                            "num_ratings": [700, 500]}), eda / "user_activity.csv")
    write_csv(pd.DataFrame({"mean_rating_bin": [2.5, 2.75, 3.0, 3.25], "num_users": [10, 20, 30, 20]}),
              eda / "user_mean_rating_hist.csv")
    write_csv(pd.DataFrame({"tag": ["synthetic"], "uses": [5], "movies": [3], "users": [2]}), eda / "top_tags.csv")
    write_json({"total_ratings": len(ratings), "total_users": n_users, "rated_movies": int((cat.num_ratings > 0).sum()),
                "avg_rating": float(ratings.rating.mean()), "std_rating": float(ratings.rating.std()),
                "first_rating": "1996-01-01T00:00:00", "last_rating": "2019-12-31T00:00:00",
                "total_movies": n_items, "total_tags": 5, "matrix_density_pct": 10.0, "matrix_sparsity_pct": 90.0,
                "ratings_per_user_avg": 15.0, "ratings_per_movie_avg": 10.0, "min_ratings_top_rated": 5,
                "user_behaviour": {"mean_ratings_per_user": 15.0, "median_ratings_per_user": 14,
                                   "p90_ratings_per_user": 25, "max_ratings_per_user": 29, "min_ratings_per_user": 8,
                                   "mean_user_avg_rating": 2.9, "mean_user_rating_std": 1.2, "mean_active_days": 0,
                                   "pct_generous_users": 5.0, "pct_harsh_users": 40.0},
                "popularity_tiers": tier_tbl.to_dict(orient="records")}, eda / "summary.json")
    write_json({"files": {"ratings": {"file": "ratings.csv", "rows": len(ratings), "partitions": 1,
                                      "missing_values": {"userId": 0}, "duplicate_rows_on_key": 0,
                                      "schema": ["userId: int (nullable=true)"], "key_columns": ["userId"]}},
                "users": n_users, "rated_movies": n_items, "ratings": len(ratings), "movies_in_catalogue": n_items},
               res / "ingestion_report.json")
    write_json({"ratings": {"input_rows": len(ratings), "removed_missing": 0, "removed_invalid": 0,
                            "removed_duplicates": 0, "output_rows": len(ratings), "removed_unknown_movie": 0},
                "movies": {"rows": n_items, "without_year": 0, "without_genre": 0}, "tags": {"rows": 5},
                "links": {"rows": n_items, "missing_tmdb": 0},
                "partitioning": {"num_partitions": 2, "non_empty_partitions": 2, "min_rows": 1, "max_rows": 2,
                                 "avg_rows": 1.5}}, res / "preprocessing_report.json")

    # Model evaluation computed with the real metric functions on the toy model
    relevant = {int(u): set(g.movieId) for u, g in test[test.rating >= 4].groupby("userId")}
    users = np.array([u for u in relevant if model.has_user(u)])
    train_seen = {int(u): g.movieId.tolist() for u, g in train.groupby("userId")}
    tc = train.groupby("movieId").size()
    ranking = popularity_ranking(tc.index.values, tc.values)
    rec_sets = {"ALS": model.recommend_batch(users, 10, train_seen),
                "Popularity": {int(u): recommend_popular(ranking, 10, train_seen.get(int(u), [])) for u in users},
                "Random": {int(u): recommend_random(tc.index.values, 10, train_seen.get(int(u), []), rng)
                           for u in users}}
    pop, ttiers = tc.astype(float).to_dict(), M.assign_tiers(tc.index, tc.values)
    rows, lor, tier_rows = [], [], []
    for name, recs in rec_sets.items():
        rk = M.ranking_metrics(recs, relevant, 10)
        b = M.popularity_bias(recs, pop, ttiers, tc.index)
        rows.append({"model": name, "precision_at_10": rk["precision_at_k"], "recall_at_10": rk["recall_at_k"],
                     "ndcg_at_10": rk["ndcg_at_k"], "hit_rate_at_10": rk["hit_rate_at_k"],
                     "coverage": M.catalogue_coverage(recs, len(tc)), **b, "users": rk["users"]})
        lor.append(M.lorenz_points(recs).assign(model=name))
        for t in ("Head", "Mid", "Tail"):
            tier_rows.append({"source": f"{name} recommendations", "popularity_tier": t, "share": b[f"{t.lower()}_share"]})
    for t in ("Head", "Mid", "Tail"):
        tier_rows.append({"source": "All training ratings", "popularity_tier": t, "share": 1 / 3})
    comp = pd.DataFrame(rows)
    write_csv(comp, res / "model_comparison.csv")
    write_csv(pd.concat(lor), res / "exposure_lorenz.csv")
    write_csv(pd.DataFrame(tier_rows), res / "tier_exposure.csv")
    write_csv(pd.DataFrame({"model": ["ALS (matrix factorisation)", "Baseline: global mean"], "rmse": [1.0, 1.2]}),
              res / "rmse_comparison.csv")
    write_csv(pd.DataFrame({"rank": [4, 4, 8, 8], "reg_param": [0.1, 0.2, 0.1, 0.2], "max_iter": 5,
                            "val_rmse": [1.0, 1.1, 1.05, 1.2], "seconds": 1.0}), res / "tuning_results.csv")
    write_json({"rank": 4, "reg_param": 0.1, "max_iter": 5, "users_with_factors": n_users,
                "movies_with_factors": len(model.item_ids), "users_with_full_list": n_users,
                "users_needing_live_scoring": 0, "train_seconds": 1.0, "candidate_k": 50}, res / "als_metadata.json")
    write_json({"k": 10, "relevance_threshold": 4.0, "rmse": {"als": 1.0, "bias_baseline": 1.1, "global_mean": 1.2},
                "test_rows": int(len(test)), "test_rows_scored": int(len(test)), "test_rows_cold_start_dropped": 0,
                "eval_users": int(len(users)), "user_profile_arp": 20.0, "catalogue_mean_popularity": 5.0,
                "catalogue_size": int(len(tc)), "models": {r["model"]: r for r in rows},
                "popularity_lift": {r["model"]: r["arp"] / 20.0 for r in rows},
                "all_users": {"users": n_users, "coverage": 0.5, "arp": 10.0, "median_rec_popularity": 5.0,
                              "head_share": 0.4, "mid_share": 0.3, "tail_share": 0.3, "gini_exposure": 0.6}},
               res / "metrics.json")
    write_json({"mode": mode, "timings_seconds": {"total": 1.0}, "spark_version": "synthetic"}, res / "run_info.json")
    return cfg_path
