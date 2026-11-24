"""MODULE B - Data preprocessing.

Cleans the raw Spark DataFrames:
  * removes rows with missing keys or invalid ratings (outside 0.5-5.0 or not
    on the half-star scale) and invalid timestamps,
  * removes duplicate (userId, movieId) ratings, keeping the most recent,
  * removes ratings for movies that do not exist in movies.csv,
  * converts Unix timestamps to proper timestamps, year and month columns,
  * extracts the release year and a clean title from "Title (1995)",
  * splits the pipe-separated genres into an array,
  * prepares a compact, partitioned (userId, movieId, rating) table for ALS.
"""
from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.constants import MAX_RATING, MIN_RATING, NO_GENRE, YEAR_PATTERN


def clean_ratings(ratings: DataFrame, movies: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Return cleaned ratings and a dict describing how many rows each rule removed."""
    stats: dict[str, int] = {"input_rows": ratings.count()}

    not_null = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    stats["removed_missing"] = stats["input_rows"] - not_null.count()

    valid = not_null.where(
        (F.col("rating") >= MIN_RATING) & (F.col("rating") <= MAX_RATING)
        & ((F.col("rating") * 2) == F.floor(F.col("rating") * 2))     # half-star scale
        & (F.col("timestamp") > 0) & (F.col("userId") > 0) & (F.col("movieId") > 0)
    )
    after_valid = valid.count()
    stats["removed_invalid"] = stats["input_rows"] - stats["removed_missing"] - after_valid

    # Keep the latest rating if a user rated the same movie more than once.
    latest = Window.partitionBy("userId", "movieId").orderBy(F.col("timestamp").desc())
    deduped = valid.withColumn("_rn", F.row_number().over(latest)).where("_rn = 1").drop("_rn")
    after_dedup = deduped.count()
    stats["removed_duplicates"] = after_valid - after_dedup

    known = deduped.join(F.broadcast(movies.select("movieId").dropna().distinct()), "movieId", "left_semi")
    with_time = (
        known.withColumn("rating_ts", F.timestamp_seconds(F.col("timestamp")))
        .withColumn("rating_year", F.year("rating_ts"))
        .withColumn("rating_month", F.date_format("rating_ts", "yyyy-MM"))
        .withColumn("rating", F.col("rating").cast("float"))
    )
    stats["output_rows"] = with_time.count()
    stats["removed_unknown_movie"] = after_dedup - stats["output_rows"]
    return with_time, stats


def clean_movies(movies: DataFrame) -> DataFrame:
    """Add ``year``, ``clean_title``, ``genres_array`` and ``num_genres``."""
    year_str = F.regexp_extract(F.col("title"), YEAR_PATTERN, 1)
    genres = F.when(F.col("genres").isNull() | (F.col("genres") == NO_GENRE), F.lit("")).otherwise(F.col("genres"))
    return (
        movies.dropna(subset=["movieId"])
        .dropDuplicates(["movieId"])
        .withColumn("title", F.trim(F.coalesce(F.col("title"), F.lit("Unknown title"))))
        .withColumn("year", F.when(year_str == "", None).otherwise(year_str.cast("int")))
        .withColumn("year", F.when((F.col("year") >= 1870) & (F.col("year") <= 2030), F.col("year")))
        .withColumn("clean_title", F.trim(F.regexp_replace(F.col("title"), YEAR_PATTERN, "")))
        .withColumn("genres", genres)
        .withColumn("genres_array", F.when(F.col("genres") == "", F.array().cast("array<string>"))
                    .otherwise(F.split(F.col("genres"), r"\|")))
        .withColumn("num_genres", F.size("genres_array"))
    )


def clean_tags(tags: DataFrame) -> DataFrame:
    """Lower-case, trim and drop empty tags."""
    return (
        tags.dropna(subset=["userId", "movieId", "tag"])
        .withColumn("tag", F.lower(F.trim(F.col("tag"))))
        .where(F.length("tag") > 0)
        .withColumn("tag_ts", F.timestamp_seconds(F.col("timestamp")))
    )


def clean_links(links: DataFrame) -> DataFrame:
    return links.dropna(subset=["movieId"]).dropDuplicates(["movieId"])


def prepare_ml_data(ratings: DataFrame, num_partitions: int) -> DataFrame:
    """Compact (userId, movieId, rating) table, hash-partitioned by user and cached.

    Partitioning by ``userId`` keeps each user's ratings in one partition, which
    reduces shuffling when ALS builds its user blocks and when we compute
    per-user metrics.
    """
    return (
        ratings.select(F.col("userId").cast("int"), F.col("movieId").cast("int"), F.col("rating").cast("float"))
        .repartition(num_partitions, "userId")
        .cache()
    )


def split_ratings(ml_ratings: DataFrame, test_fraction: float, seed: int) -> tuple[DataFrame, DataFrame]:
    """Random hold-out split of individual ratings (e.g. 80% train / 20% test)."""
    train, test = ml_ratings.randomSplit([1.0 - test_fraction, test_fraction], seed=seed)
    return train.cache(), test.cache()


def preprocess(frames: dict[str, DataFrame]) -> tuple[dict[str, DataFrame], dict[str, Any]]:
    """Run every cleaning step. Returns cleaned frames and a statistics dict."""
    movies = clean_movies(frames["movies"]).cache()
    ratings, rating_stats = clean_ratings(frames["ratings"], movies)
    ratings = ratings.cache()
    tags = clean_tags(frames["tags"])
    links = clean_links(frames["links"])
    stats = {
        "ratings": rating_stats,
        "movies": {"rows": movies.count(), "without_year": movies.where(F.col("year").isNull()).count(),
                   "without_genre": movies.where(F.col("num_genres") == 0).count()},
        "tags": {"rows": tags.count()},
        "links": {"rows": links.count(), "missing_tmdb": links.where(F.col("tmdbId").isNull()).count()},
    }
    return {"ratings": ratings, "movies": movies, "tags": tags, "links": links}, stats
