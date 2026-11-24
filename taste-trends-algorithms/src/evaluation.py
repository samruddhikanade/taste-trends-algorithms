"""MODULE F - Model evaluation.

Every number written here is computed from the trained model's real outputs
on the held-out test set:

  * RMSE of ALS vs two simple baselines (global mean; user+movie bias
    baseline  r̂ = μ + b_u + b_i  from Koren et al. 2009),
  * Precision@10, Recall@10, NDCG@10 and HitRate@10 for ALS vs a
    most-popular baseline vs random recommendations,
  * catalogue coverage and popularity-bias statistics.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import metrics as M
from src.io_utils import write_csv, write_json
from src.recommendation import factors_to_pandas, rmse_of
from src.recommender_core import FactorModel, popularity_ranking, recommend_popular, recommend_random


def warm_test(train: DataFrame, test: DataFrame) -> DataFrame:
    """Test rows whose user AND movie both appear in the training data.

    ALS cannot score unseen users/movies (cold start), and Spark's
    ``coldStartStrategy='drop'`` removes them; we apply the same filter to every
    model so all RMSEs are computed on exactly the same rows.
    """
    return (test.join(train.select("userId").distinct(), "userId", "left_semi")
                .join(train.select("movieId").distinct(), "movieId", "left_semi"))


def baseline_rmses(train: DataFrame, test: DataFrame, damping: float = 10.0) -> dict[str, float]:
    mu = float(train.agg(F.avg("rating")).collect()[0][0])
    item_bias = train.groupBy("movieId").agg(
        (F.sum(F.col("rating") - mu) / (F.count(F.lit(1)) + damping)).alias("b_i"))
    user_bias = (train.join(item_bias, "movieId")
                 .groupBy("userId").agg((F.sum(F.col("rating") - mu - F.col("b_i")) /
                                        (F.count(F.lit(1)) + damping)).alias("b_u")))
    scored = (test.join(item_bias, "movieId", "left").join(user_bias, "userId", "left")
              .withColumn("pred_bias", F.lit(mu) + F.coalesce("b_u", F.lit(0.0)) + F.coalesce("b_i", F.lit(0.0)))
              .withColumn("pred_bias", F.least(F.lit(5.0), F.greatest(F.lit(0.5), F.col("pred_bias")))))
    row = scored.agg(
        F.sqrt(F.avg((F.col("rating") - F.lit(mu)) ** 2)).alias("global_mean"),
        F.sqrt(F.avg((F.col("rating") - F.col("pred_bias")) ** 2)).alias("bias"),
    ).collect()[0]
    return {"global_mean": float(row["global_mean"]), "bias_baseline": float(row["bias"]), "mu": mu}


def evaluate(model: ALSModel, train: DataFrame, test: DataFrame, all_recs_pdf: pd.DataFrame,
             cfg: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    ecfg = cfg["run"]["evaluation"]
    k, thr, seed = int(ecfg["k"]), float(ecfg["relevance_threshold"]), int(cfg["project"]["random_seed"])
    out = Path(cfg["paths"]["results_dir"])
    results: dict[str, Any] = {"k": k, "relevance_threshold": thr}

    # ---------------------------------------------------------------- RMSE
    test_w = warm_test(train, test).cache()
    n_test, n_warm = test.count(), test_w.count()
    als_rmse = rmse_of(model, test_w)
    base = baseline_rmses(train, test_w)
    results["rmse"] = {"als": als_rmse, **{k_: v for k_, v in base.items() if k_ != "mu"}}
    results["test_rows"] = n_test
    results["test_rows_scored"] = n_warm
    results["test_rows_cold_start_dropped"] = n_test - n_warm
    write_csv(pd.DataFrame([
        {"model": "ALS (matrix factorisation)", "rmse": als_rmse},
        {"model": "Baseline: μ + user bias + movie bias", "rmse": base["bias_baseline"]},
        {"model": "Baseline: global mean", "rmse": base["global_mean"]},
    ]), out / "rmse_comparison.csv")
    print(f"  RMSE  ALS={als_rmse:.4f}  bias-baseline={base['bias_baseline']:.4f}  global-mean={base['global_mean']:.4f}")

    # ---------------------------------------------------------------- ranking
    item_pop = train.groupBy("movieId").count().toPandas()
    pop = dict(zip(item_pop["movieId"].astype(int), item_pop["count"].astype(float)))
    tiers = M.assign_tiers(item_pop["movieId"], item_pop["count"])
    catalogue = np.array(sorted(pop))

    users_fac = factors_to_pandas(model.userFactors, "userId")
    items_fac = factors_to_pandas(model.itemFactors, "movieId")
    fcols = [c for c in users_fac.columns if c.startswith("f")]
    fm = FactorModel(users_fac["userId"].values, users_fac[fcols].values,
                     items_fac["movieId"].values, items_fac[fcols].values)

    liked = (test.where(F.col("rating") >= thr)
             .join(train.select("movieId").distinct(), "movieId", "left_semi")
             .join(model.userFactors.select(F.col("id").alias("userId")), "userId", "left_semi"))
    eval_users = (liked.select("userId").distinct().orderBy(F.rand(seed)).limit(int(ecfg["max_eval_users"]))).cache()
    rel_pdf = liked.join(eval_users, "userId", "left_semi").select("userId", "movieId").toPandas()
    seen_pdf = train.join(eval_users, "userId", "left_semi").select("userId", "movieId").toPandas()
    relevant = {int(u): set(g["movieId"].astype(int)) for u, g in rel_pdf.groupby("userId")}
    seen = {int(u): g["movieId"].astype(int).tolist() for u, g in seen_pdf.groupby("userId")}
    users = np.array(sorted(relevant))
    print(f"  ranking evaluation on {len(users):,} sampled test users ...")

    ranking = popularity_ranking(catalogue, np.array([pop[i] for i in catalogue]))
    rng = np.random.default_rng(seed)
    rec_sets = {
        "ALS": fm.recommend_batch(users, k, seen),
        "Popularity": {int(u): recommend_popular(ranking, k, seen.get(int(u), ())) for u in users},
        "Random": {int(u): recommend_random(catalogue, k, seen.get(int(u), ()), rng) for u in users},
    }
    rows, lorenz = [], []
    for name, recs in rec_sets.items():
        rk = M.ranking_metrics(recs, relevant, k)
        bias = M.popularity_bias(recs, pop, tiers, catalogue)
        rows.append({"model": name, "precision_at_10": rk["precision_at_k"], "recall_at_10": rk["recall_at_k"],
                     "ndcg_at_10": rk["ndcg_at_k"], "hit_rate_at_10": rk["hit_rate_at_k"],
                     "coverage": M.catalogue_coverage(recs, len(catalogue)), **bias, "users": rk["users"]})
        lz = M.lorenz_points(recs)
        lz["model"] = name
        lorenz.append(lz)
    comparison = pd.DataFrame(rows)
    write_csv(comparison, out / "model_comparison.csv")
    write_csv(pd.concat(lorenz, ignore_index=True), out / "exposure_lorenz.csv")
    for r in rows:
        print(f"  {r['model']:<10} P@10={r['precision_at_10']:.4f} R@10={r['recall_at_10']:.4f} "
              f"coverage={r['coverage']:.4f} ARP={r['arp']:.0f} head={r['head_share']:.1%}")

    # What users actually consume, for comparison with what is recommended.
    profile_pop = np.array([pop.get(int(i), 0.0) for i in seen_pdf["movieId"]])
    results["user_profile_arp"] = float(profile_pop.mean()) if len(profile_pop) else 0.0
    results["catalogue_mean_popularity"] = float(np.mean(list(pop.values())))
    results["catalogue_size"] = int(len(catalogue))
    results["eval_users"] = int(len(users))
    results["models"] = {r["model"]: {k_: v for k_, v in r.items() if k_ != "model"} for r in rows}
    results["popularity_lift"] = {r["model"]: (r["arp"] / results["user_profile_arp"]
                                               if results["user_profile_arp"] else None) for r in rows}

    # ---------------------------------------------------------------- tier exposure
    tier_rows = []
    cat_tiers = pd.Series(tiers).value_counts(normalize=True)
    rating_tiers = item_pop.assign(t=item_pop["movieId"].map(tiers)).groupby("t")["count"].sum()
    rating_tiers = rating_tiers / rating_tiers.sum()
    profile_tiers = seen_pdf["movieId"].map(tiers).value_counts(normalize=True)
    for source, dist in [("Catalogue (movies)", cat_tiers), ("All training ratings", rating_tiers),
                         ("Eval users' history", profile_tiers)]:
        for t in ("Head", "Mid", "Tail"):
            tier_rows.append({"source": source, "popularity_tier": t, "share": float(dist.get(t, 0.0))})
    for r in rows:
        for t in ("Head", "Mid", "Tail"):
            tier_rows.append({"source": f"{r['model']} recommendations", "popularity_tier": t,
                              "share": r[f"{t.lower()}_share"]})
    write_csv(pd.DataFrame(tier_rows), out / "tier_exposure.csv")

    # ---------------------------------------------------------------- all users (deployed lists)
    if all_recs_pdf is not None and len(all_recs_pdf):
        srt = all_recs_pdf.sort_values(["userId", "rank"])
        uids, starts = np.unique(srt["userId"].to_numpy(), return_index=True)
        all_recs = dict(zip(uids.tolist(), np.split(srt["movieId"].to_numpy(dtype=np.int64), starts[1:])))
        results["all_users"] = {
            "users": len(all_recs),
            "coverage": M.catalogue_coverage(all_recs, len(catalogue)),
            **M.popularity_bias(all_recs, pop, tiers, catalogue),
        }
        print(f"  all-user ALS lists: coverage={results['all_users']['coverage']:.4f} "
              f"head share={results['all_users']['head_share']:.1%}")

    test_w.unpersist()
    eval_users.unpersist()
    results["evaluation_seconds"] = round(time.time() - t0, 1)
    write_json(results, out / "metrics.json")
    return results
