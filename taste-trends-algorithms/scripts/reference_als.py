"""Fallback / teaching tool: train the from-scratch NumPy ALS on the small
MovieLens dataset WITHOUT Spark and report real RMSE, Precision@10, Recall@10.

Use it (a) to understand ALS line by line, or (b) to still get genuine results
if Java/Spark refuses to start on your machine. It only supports the dev
dataset (100k ratings); MovieLens 25M must go through Spark.

    python scripts/reference_als.py
    python scripts/reference_als.py --rank 16 --reg 0.1 --iters 10
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src import metrics as M
from src.config import load_config
from src.io_utils import write_json
from src.recommender_core import popularity_ranking, recommend_popular, train_als_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    cfg = load_config("dev")
    path = cfg["paths"]["dataset_dir"] / "ratings.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing - run: python scripts/download_data.py --mode dev")
    ratings = pd.read_csv(path)  # 100k rows: fine for pandas (the 25M file is never loaded this way)
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    is_test = rng.random(len(ratings)) < args.test_fraction
    train, test = ratings[~is_test], ratings[is_test]

    model = train_als_numpy(train.userId.values, train.movieId.values, train.rating.values,
                            rank=args.rank, reg_param=args.reg, iterations=args.iters, verbose=True)
    warm = test[test.userId.map(model.has_user) & test.movieId.map(model.has_item)]
    preds = [model.predict(u, m) for u, m in zip(warm.userId, warm.movieId)]
    rmse = M.rmse(preds, warm.rating)

    k, thr = 10, 4.0
    liked = warm[warm.rating >= thr]
    relevant = {int(u): set(g.movieId.astype(int)) for u, g in liked.groupby("userId")}
    seen = {int(u): g.movieId.astype(int).tolist() for u, g in train.groupby("userId")}
    users = np.array(sorted(relevant))
    als_recs = model.recommend_batch(users, k, seen)
    counts = train.groupby("movieId").size()
    ranking = popularity_ranking(counts.index.values, counts.values)
    pop_recs = {int(u): recommend_popular(ranking, k, seen.get(int(u), ())) for u in users}

    result = {"engine": "numpy reference ALS (not Spark)", "rank": args.rank, "reg_param": args.reg,
              "iterations": args.iters, "rmse": rmse, "test_rows_scored": int(len(warm)),
              "ALS": M.ranking_metrics(als_recs, relevant, k), "Popularity": M.ranking_metrics(pop_recs, relevant, k),
              "ALS_coverage": M.catalogue_coverage(als_recs, len(counts)),
              "Popularity_coverage": M.catalogue_coverage(pop_recs, len(counts))}
    out = write_json(result, cfg["paths"]["results_dir"] / "reference_als_numpy.json")
    print(f"\nRMSE {rmse:.4f} | ALS P@10 {result['ALS']['precision_at_k']:.4f} R@10 {result['ALS']['recall_at_k']:.4f} | "
          f"Popularity P@10 {result['Popularity']['precision_at_k']:.4f}\nSaved to {out}")


if __name__ == "__main__":
    main()
