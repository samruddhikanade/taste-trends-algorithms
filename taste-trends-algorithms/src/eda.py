"""MODULE C - Exploratory data analysis.

Runs the Spark/Spark SQL aggregations from ``analytics.py``, collects only the
small aggregated results, saves them to ``reports/results/<mode>/eda`` (CSV +
JSON) and exports interactive Plotly charts to ``reports/figures/<mode>``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from src import analytics as A
from src.io_utils import read_json, write_csv, write_json, write_table


def _save(sdf: DataFrame, path: Path) -> pd.DataFrame:
    pdf = sdf.toPandas()
    write_csv(pdf, path)
    return pdf


def run_eda(spark: SparkSession, frames: dict[str, DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    out = Path(cfg["paths"]["results_dir"]) / "eda"
    out.mkdir(parents=True, exist_ok=True)
    min_count = int(cfg["run"]["min_ratings_top_rated"])
    t0 = time.time()

    A.register_views(frames)
    summary = A.global_summary(spark)
    print(f"  global summary: {summary['total_ratings']:,} ratings, {summary['total_users']:,} users")

    _save(A.rating_distribution(spark), out / "rating_distribution.csv")
    movie_stats = A.movie_popularity(spark, min_count).cache()
    movie_stats.count()
    _save(A.most_rated(spark, 50), out / "most_rated.csv")
    _save(A.top_rated(spark, min_count, 50), out / "top_rated.csv")
    tiers = _save(A.popularity_tiers(spark), out / "popularity_tiers.csv")
    _save(A.long_tail_curve(spark), out / "long_tail_curve.csv")
    print("  popularity analysis done")

    _save(A.genre_stats(spark), out / "genre_stats.csv")
    _save(A.genre_decade_share(spark), out / "genre_decade_share.csv")
    _save(A.release_year_stats(spark), out / "release_year_stats.csv")
    print("  genre and release-year analysis done")

    _save(A.ratings_over_time(spark), out / "ratings_over_time.csv")
    _save(A.ratings_by_weekday_hour(spark), out / "ratings_by_weekday_hour.csv")
    activity, mean_hist, behaviour = A.user_behaviour(spark)
    _save(activity, out / "user_activity.csv")
    _save(mean_hist, out / "user_mean_rating_hist.csv")
    _save(A.top_tags(spark), out / "top_tags.csv")
    print("  time trends and user behaviour done")

    write_table(A.movie_catalogue(spark, frames["links"]).toPandas(),
                Path(cfg["paths"]["processed_dir"]) / "movie_catalogue")

    summary["user_behaviour"] = behaviour
    summary["popularity_tiers"] = tiers.to_dict(orient="records")
    summary["min_ratings_top_rated"] = min_count
    summary["eda_seconds"] = round(time.time() - t0, 1)
    write_json(summary, out / "summary.json")
    movie_stats.unpersist()
    return summary


def build_catalogue(spark: SparkSession, frames: dict[str, DataFrame], cfg: dict[str, Any]) -> None:
    """Movie catalogue for the dashboard (titles, stats, tiers, links, tags) when the EDA is skipped."""
    A.register_views(frames)
    A.movie_popularity(spark, int(cfg["run"]["min_ratings_top_rated"]))
    write_table(A.movie_catalogue(spark, frames["links"]).toPandas(),
                Path(cfg["paths"]["processed_dir"]) / "movie_catalogue")


def export_figures(cfg: dict[str, Any]) -> list[Path]:
    """Save the EDA charts as standalone interactive HTML files (needs plotly)."""
    try:
        from src import visuals as V
    except ImportError:
        print("  plotly not installed - skipping figure export")
        return []
    eda = Path(cfg["paths"]["results_dir"]) / "eda"
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not (eda / "summary.json").exists():
        return []

    def load(name: str) -> pd.DataFrame:
        return pd.read_csv(eda / name)

    min_count = read_json(eda / "summary.json")["min_ratings_top_rated"]
    figures = {
        "01_rating_distribution": V.rating_distribution(load("rating_distribution.csv")),
        "02_ratings_over_time": V.ratings_over_time(load("ratings_over_time.csv")),
        "03_release_years": V.release_years(load("release_year_stats.csv")),
        "04_most_rated": V.top_movies_bar(load("most_rated.csv"), "num_ratings", "Most rated movies"),
        "05_top_rated": V.top_movies_bar(load("top_rated.csv"), "avg_rating",
                                         f"Highest rated (at least {min_count:,} ratings)", color=V.MINT),
        "06_long_tail": V.long_tail(load("long_tail_curve.csv")),
        "07_popularity_tiers": V.tier_summary(load("popularity_tiers.csv")),
        "08_genres": V.genre_bars(load("genre_stats.csv")),
        "09_genre_decades": V.genre_decade_heatmap(load("genre_decade_share.csv")),
        "10_user_activity": V.user_activity(load("user_activity.csv")),
        "11_user_mean_rating": V.user_mean_hist(load("user_mean_rating_hist.csv")),
    }
    written = []
    for name, fig in figures.items():
        path = fig_dir / f"{name}.html"
        fig.update_layout(paper_bgcolor=V.BG, plot_bgcolor=V.BG)
        fig.write_html(str(path), include_plotlyjs="directory", full_html=True)
        written.append(path)
    return written

