"""End-to-end orchestration of modules A-F, used by the scripts in ``scripts/``."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.config import ensure_dirs, load_config
from src.io_utils import read_table, write_json


def run(mode: str = "dev", stages: tuple[str, ...] = ("eda", "model"), sample_fraction: float | None = None) -> dict[str, Any]:
    from src import eda, evaluation, ingestion, preprocessing, recommendation
    from src.analytics import partition_report
    from src.report import write_results_markdown
    from src.spark_session import create_spark

    cfg = load_config(mode)
    if sample_fraction is not None:
        cfg["run"]["sample_fraction"] = sample_fraction
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    seed = int(cfg["project"]["random_seed"])
    timings: dict[str, float] = {}
    t_all = time.time()

    print(f"\n### Taste, Trends & Algorithms - mode={mode}, data={cfg['paths']['dataset_dir']}")
    spark = create_spark(cfg)
    print(f"Spark {spark.version} | master={spark.sparkContext.master} | "
          f"default parallelism={spark.sparkContext.defaultParallelism}")
    try:
        # ---------------------------------------------------------- A. ingestion
        t0 = time.time()
        frames = ingestion.load_movielens(spark, cfg["paths"]["dataset_dir"], float(cfg["run"]["sample_fraction"]), seed)
        frames["ratings"] = frames["ratings"].cache()
        report = ingestion.ingestion_report(frames)
        ingestion.print_report(report)
        write_json(report, results_dir / "ingestion_report.json")
        timings["ingestion"] = round(time.time() - t0, 1)

        # ---------------------------------------------------------- B. preprocessing
        t0 = time.time()
        print("\n=== MODULE B: Preprocessing ===")
        clean, prep_stats = preprocessing.preprocess(frames)
        frames["ratings"].unpersist()
        print(f"  ratings in={prep_stats['ratings']['input_rows']:,} out={prep_stats['ratings']['output_rows']:,} "
              f"(missing={prep_stats['ratings']['removed_missing']}, invalid={prep_stats['ratings']['removed_invalid']}, "
              f"duplicates={prep_stats['ratings']['removed_duplicates']}, "
              f"unknown movie={prep_stats['ratings']['removed_unknown_movie']})")
        ml = preprocessing.prepare_ml_data(clean["ratings"], int(cfg["run"]["shuffle_partitions"]))
        prep_stats["partitioning"] = partition_report(ml)
        print(f"  ML table partitioned by userId: {prep_stats['partitioning']}")
        write_json(prep_stats, results_dir / "preprocessing_report.json")
        timings["preprocessing"] = round(time.time() - t0, 1)

        # ---------------------------------------------------------- C/D. EDA + big-data analytics
        if "eda" in stages:
            t0 = time.time()
            print("\n=== MODULES C + D: Spark SQL analytics / EDA ===")
            eda.run_eda(spark, clean, cfg)
            figs = eda.export_figures(cfg)
            print(f"  {len(figs)} interactive charts written to {cfg['paths']['figures_dir']}")
            timings["eda"] = round(time.time() - t0, 1)
        else:
            eda.build_catalogue(spark, clean, cfg)
        clean["ratings"].unpersist()

        # ---------------------------------------------------------- E. ALS
        if "model" in stages:
            t0 = time.time()
            print("\n=== MODULE E: ALS recommendation engine ===")
            train, test = preprocessing.split_ratings(ml, float(cfg["als_defaults"]["test_fraction"]), seed)
            print(f"  train={train.count():,} ratings, test={test.count():,} ratings")
            print("  hyper-parameter search:")
            best, grid = recommendation.tune(train, cfg)
            recommendation.save_tuning(grid, cfg)
            print(f"  best params: {best}")
            model, train_secs = recommendation.train_final(train, best, cfg)
            print(f"  final model trained in {train_secs}s")
            k = int(cfg["run"]["evaluation"]["k"])
            recs = recommendation.recommend_all_users(model, ml, k, int(cfg["run"]["als"]["candidate_k"])).cache()
            meta = recommendation.save_artifacts(model, recs, ml, best, cfg, train_secs)
            recommendation.try_save_spark_model(model, cfg)
            print(f"  saved factors for {meta['users_with_factors']:,} users and "
                  f"{meta['movies_with_factors']:,} movies; top-{k} lists for all users")
            timings["training"] = round(time.time() - t0, 1)

            # ------------------------------------------------------ F. evaluation
            t0 = time.time()
            print("\n=== MODULE F: Evaluation ===")
            recs_pdf = read_table(Path(cfg["paths"]["processed_dir"]) / "als_recommendations")
            evaluation.evaluate(model, train, test, recs_pdf, cfg)
            recs.unpersist()
            timings["evaluation"] = round(time.time() - t0, 1)
    finally:
        timings["total"] = round(time.time() - t_all, 1)
        write_json({"mode": mode, "timings_seconds": timings, "spark_version": spark.version,
                    "sample_fraction": cfg["run"]["sample_fraction"]}, results_dir / "run_info.json")
        spark.stop()

    md = write_results_markdown(cfg)
    from src.config import PROJECT_ROOT
    from src.report import update_readme
    full_done = (Path(load_config("full")["paths"]["results_dir"]) / "metrics.json").exists()
    # The README shows MovieLens 25M results once they exist; a later dev run won't overwrite them.
    if (mode == "full" or not full_done) and update_readme(
            PROJECT_ROOT / "README.md", results_dir, mode, Path(cfg["paths"]["figures_dir"]) / "screenshots"):
        print("README.md results section updated.")
    print(f"\nDone in {timings['total']}s. Results summary: {md}")
    return timings
