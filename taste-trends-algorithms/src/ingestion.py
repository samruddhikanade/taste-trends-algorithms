"""MODULE A - Data ingestion.

Loads the four MovieLens CSV files into Spark DataFrames with *explicit*
schemas (faster and safer than ``inferSchema`` on a 25-million-row file),
optionally samples users for development runs, and produces a data-quality
report: schemas, row counts, distinct users/movies, missing values and
duplicates. Every count is a distributed Spark action - nothing is loaded
into pandas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (DoubleType, IntegerType, LongType, StringType,
                               StructField, StructType)

RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", DoubleType(), True),
    StructField("timestamp", LongType(), True),
])
MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True),
])
TAGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("tag", StringType(), True),
    StructField("timestamp", LongType(), True),
])
LINKS_SCHEMA = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("imdbId", StringType(), True),   # string keeps leading zeros (e.g. 0114709)
    StructField("tmdbId", IntegerType(), True),
])

FILES = {
    "ratings": ("ratings.csv", RATINGS_SCHEMA, ["userId", "movieId"]),
    "movies": ("movies.csv", MOVIES_SCHEMA, ["movieId"]),
    "tags": ("tags.csv", TAGS_SCHEMA, ["userId", "movieId", "tag", "timestamp"]),
    "links": ("links.csv", LINKS_SCHEMA, ["movieId"]),
}


def load_csv(spark: SparkSession, path: str | Path, schema: StructType) -> DataFrame:
    """Read a CSV with a header row using a fixed schema.

    ``PERMISSIVE`` mode turns malformed values into nulls instead of failing,
    so they can be counted and removed during preprocessing. ``quote``/``escape``
    are set because MovieLens titles contain commas inside double quotes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the data first: python scripts/download_data.py --mode <dev|full>"
        )
    return (
        spark.read.option("header", "true")
        .option("mode", "PERMISSIVE")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .schema(schema)
        .csv(str(path))
    )


def sample_users(df: DataFrame, fraction: float, seed: int = 42) -> DataFrame:
    """Keep a deterministic ``fraction`` of users (all of their rows).

    Sampling *users* rather than individual ratings keeps each user's history
    complete, which matters for collaborative filtering.
    """
    if fraction >= 1.0:
        return df
    bucket = F.pmod(F.xxhash64(F.col("userId"), F.lit(seed)), F.lit(10_000))
    return df.where(bucket < int(fraction * 10_000))


def load_movielens(spark: SparkSession, dataset_dir: str | Path,
                   sample_fraction: float = 1.0, seed: int = 42) -> dict[str, DataFrame]:
    """Load ratings, movies, tags and links as Spark DataFrames."""
    dataset_dir = Path(dataset_dir)
    frames = {name: load_csv(spark, dataset_dir / fname, schema) for name, (fname, schema, _) in FILES.items()}
    frames["ratings"] = sample_users(frames["ratings"], sample_fraction, seed)
    frames["tags"] = sample_users(frames["tags"], sample_fraction, seed)
    return frames


def schema_as_text(df: DataFrame) -> list[str]:
    return [f"{f.name}: {f.dataType.simpleString()} (nullable={str(f.nullable).lower()})" for f in df.schema.fields]


def profile_dataframe(df: DataFrame, key_columns: list[str]) -> dict[str, Any]:
    """Row count, per-column missing values and duplicate count, in two Spark jobs."""
    exprs = [F.count(F.lit(1)).alias("__rows")]
    exprs += [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
    row = df.agg(*exprs).collect()[0].asDict()
    rows = int(row.pop("__rows"))
    missing = {c: int(v or 0) for c, v in row.items()}
    distinct_keys = df.select(*key_columns).distinct().count()
    return {
        "rows": rows,
        "missing_values": missing,
        "duplicate_rows_on_key": rows - int(distinct_keys),
        "key_columns": key_columns,
    }


def ingestion_report(frames: dict[str, DataFrame]) -> dict[str, Any]:
    """Data-quality summary for every loaded file plus headline counts."""
    report: dict[str, Any] = {"files": {}}
    for name, (fname, _, keys) in FILES.items():
        df = frames[name]
        info = profile_dataframe(df, keys)
        info["file"] = fname
        info["schema"] = schema_as_text(df)
        info["partitions"] = df.rdd.getNumPartitions()
        report["files"][name] = info

    ratings = frames["ratings"]
    headline = ratings.agg(
        F.countDistinct("userId").alias("users"),
        F.countDistinct("movieId").alias("rated_movies"),
        F.count(F.lit(1)).alias("ratings"),
    ).collect()[0]
    report["users"] = int(headline["users"])
    report["rated_movies"] = int(headline["rated_movies"])
    report["ratings"] = int(headline["ratings"])
    report["movies_in_catalogue"] = report["files"]["movies"]["rows"]
    return report


def print_report(report: dict[str, Any]) -> None:
    print("\n=== MODULE A: Data ingestion ===")
    for name, info in report["files"].items():
        print(f"\n{info['file']}  ({info['rows']:,} rows, {info['partitions']} partitions)")
        for line in info["schema"]:
            print(f"   {line}")
        missing = {k: v for k, v in info["missing_values"].items() if v}
        print(f"   missing values : {missing or 'none'}")
        print(f"   duplicates on {info['key_columns']}: {info['duplicate_rows_on_key']:,}")
    print(f"\nUsers: {report['users']:,} | Rated movies: {report['rated_movies']:,} | "
          f"Ratings: {report['ratings']:,} | Movies in catalogue: {report['movies_in_catalogue']:,}")
