"""MODULE E - Recommendation engine (Spark MLlib ALS).

Implements the matrix-factorisation approach of Koren, Bell & Volinsky (2009):
the sparse user x movie rating matrix R is approximated by P · Qᵀ where every
user and every movie is described by ``rank`` latent factors. Spark's ALS
learns P and Q by Alternating Least Squares, distributing the work over
user/item blocks.

Steps
  1. hyper-parameter search (rank, regParam) on a validation split of TRAIN,
  2. final model trained on the full TRAIN split,
  3. top-10 recommendations for every user, excluding movies already rated,
  4. factors exported to disk so the dashboard can serve recommendations and
     handle cold-start users without starting Spark.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.io_utils import write_csv, write_json, write_table
from src.ingestion import sample_users


def build_als(rank: int, reg_param: float, max_iter: int, seed: int, cfg: dict[str, Any]) -> ALS:
    als_cfg = cfg["als_defaults"]
    blocks = max(2, min(int(cfg["run"]["shuffle_partitions"]), 32))
    return ALS(
        userCol="userId", itemCol="movieId", ratingCol="rating",
        rank=rank, regParam=reg_param, maxIter=max_iter, seed=seed,
        implicitPrefs=bool(als_cfg["implicit_prefs"]), nonnegative=bool(als_cfg["nonnegative"]),
        coldStartStrategy=als_cfg["cold_start_strategy"],   # 'drop' = ignore unseen users/movies when scoring
        numUserBlocks=blocks, numItemBlocks=blocks,
    )


def rmse_of(model: ALSModel, data: DataFrame) -> float:
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    return float(evaluator.evaluate(model.transform(data)))


def tune(train: DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Grid search over rank x regParam; returns the best params and the full grid."""
    run, seed = cfg["run"], int(cfg["project"]["random_seed"])
    defaults = {"rank": int(cfg["als_defaults"]["rank"]), "reg_param": float(cfg["als_defaults"]["reg_param"])}
    tcfg = run["tuning"]
    if not tcfg.get("enabled", True):
        return defaults, pd.DataFrame()

    data = sample_users(train, float(tcfg.get("sample_fraction", 1.0)), seed)
    fit, val = data.randomSplit([1 - cfg["als_defaults"]["validation_fraction"],
                                 cfg["als_defaults"]["validation_fraction"]], seed=seed)
    fit, val = fit.cache(), val.cache()
    rows = []
    for rank, reg in itertools.product(tcfg["ranks"], tcfg["reg_params"]):
        t0 = time.time()
        model = build_als(int(rank), float(reg), int(tcfg["max_iter"]), seed, cfg).fit(fit)
        score = rmse_of(model, val)
        rows.append({"rank": int(rank), "reg_param": float(reg), "max_iter": int(tcfg["max_iter"]),
                     "val_rmse": round(score, 5), "seconds": round(time.time() - t0, 1)})
        print(f"    rank={rank:<3} regParam={reg:<5} -> validation RMSE {score:.4f}  ({rows[-1]['seconds']}s)")
    fit.unpersist()
    val.unpersist()
    grid = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    best = {"rank": int(grid.loc[0, "rank"]), "reg_param": float(grid.loc[0, "reg_param"])}
    return best, grid


def train_final(train: DataFrame, params: dict[str, Any], cfg: dict[str, Any]) -> tuple[ALSModel, float]:
    t0 = time.time()
    model = build_als(params["rank"], params["reg_param"], int(cfg["run"]["als"]["max_iter"]),
                      int(cfg["project"]["random_seed"]), cfg).fit(train)
    # Force materialisation so the timing includes the actual work.
    model.userFactors.count()
    return model, round(time.time() - t0, 1)


def recommend_all_users(model: ALSModel, all_ratings: DataFrame, k: int, candidate_k: int) -> DataFrame:
    """Top-k recommendations for every user, excluding every movie the user has rated.

    Spark scores ``candidate_k`` movies per user (blocked matrix multiplication),
    then a distributed anti-join removes already-rated movies and a window
    function keeps the best ``k`` that remain.
    """
    cands = (
        model.recommendForAllUsers(candidate_k)
        .select("userId", F.posexplode("recommendations").alias("pos", "rec"))
        .select("userId", F.col("rec.movieId").alias("movieId"), F.col("rec.rating").alias("score"))
    )
    unseen = cands.join(all_ratings.select("userId", "movieId"), ["userId", "movieId"], "left_anti")
    w = Window.partitionBy("userId").orderBy(F.col("score").desc(), F.col("movieId"))
    return unseen.withColumn("rank", F.row_number().over(w)).where(F.col("rank") <= k)


def factors_to_pandas(factors: DataFrame, id_name: str) -> pd.DataFrame:
    """Spark factor table (id, features array) -> wide pandas table (id, f0..fN)."""
    pdf = factors.toPandas()
    mat = np.vstack(pdf["features"].values).astype(np.float32) if len(pdf) else np.zeros((0, 0), np.float32)
    wide = pd.DataFrame(mat, columns=[f"f{i}" for i in range(mat.shape[1])])
    wide.insert(0, id_name, pdf["id"].astype("int64").values)
    return wide


def save_artifacts(model: ALSModel, recs: DataFrame, all_ratings: DataFrame, params: dict[str, Any],
                   cfg: dict[str, Any], train_seconds: float) -> dict[str, Any]:
    """Persist factors, recommendations and metadata for the dashboard."""
    proc = Path(cfg["paths"]["processed_dir"])
    k = int(cfg["run"]["evaluation"]["k"])
    users = factors_to_pandas(model.userFactors, "userId")
    items = factors_to_pandas(model.itemFactors, "movieId")
    write_table(users, proc / "user_factors")
    write_table(items, proc / "item_factors")

    recs_pdf = recs.select("userId", "rank", "movieId", F.round("score", 4).alias("score")).toPandas()
    write_table(recs_pdf, proc / "als_recommendations")

    # Very heavy raters can have fewer than k unseen movies among the candidates.
    # Save their rated-movie lists so the dashboard can compute their list exactly.
    counts = recs_pdf.groupby("userId").size()
    short_users = set(users["userId"]) - set(counts[counts >= k].index)
    seen_short = pd.DataFrame(columns=["userId", "movieId"])
    if short_users:
        su = all_ratings.sparkSession.createDataFrame([(int(u),) for u in short_users], "userId INT")
        seen_short = all_ratings.join(F.broadcast(su), "userId").select("userId", "movieId").toPandas()
    write_table(seen_short, proc / "seen_for_short_users")

    # A few hundred "demo" users whose rating history is saved, so the dashboard
    # can show what they watched next to what ALS recommends.
    demo = (all_ratings.select("userId").distinct()
            .orderBy(F.rand(int(cfg["project"]["random_seed"]))).limit(300))
    demo_hist = all_ratings.join(F.broadcast(demo), "userId").select("userId", "movieId", "rating").toPandas()
    write_table(demo_hist, proc / "demo_user_history")

    meta = {
        "rank": params["rank"], "reg_param": params["reg_param"], "max_iter": int(cfg["run"]["als"]["max_iter"]),
        "users_with_factors": int(len(users)), "movies_with_factors": int(len(items)),
        "users_with_full_list": int((counts >= k).sum()), "users_needing_live_scoring": len(short_users),
        "train_seconds": train_seconds, "candidate_k": int(cfg["run"]["als"]["candidate_k"]),
    }
    write_json(meta, Path(cfg["paths"]["models_dir"]) / "als_metadata.json")
    write_json(meta, Path(cfg["paths"]["results_dir"]) / "als_metadata.json")  # copy kept with the results
    return meta


def try_save_spark_model(model: ALSModel, cfg: dict[str, Any]) -> str | None:
    """Also save the native Spark model. On Windows this needs winutils.exe, so
    failure is reported but not fatal - the exported factors are enough."""
    path = Path(cfg["paths"]["models_dir"]) / "als_spark_model"
    try:
        model.write().overwrite().save(str(path))
        return str(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  (native Spark model not saved: {type(exc).__name__}; exported factors are used instead)")
        return None


def save_tuning(grid: pd.DataFrame, cfg: dict[str, Any]) -> None:
    if not grid.empty:
        write_csv(grid, Path(cfg["paths"]["results_dir"]) / "tuning_results.csv")
