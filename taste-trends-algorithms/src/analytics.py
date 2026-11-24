"""MODULE D - Big Data processing with PySpark and Spark SQL.

Every function here takes Spark DataFrames, runs distributed aggregations
(mostly written in Spark SQL so the queries read like the report), and
returns a *small* Spark DataFrame. ``eda.py`` decides what to collect and save.

Temporary views used by the SQL:
    ratings(userId, movieId, rating, timestamp, rating_ts, rating_year, rating_month)
    movies(movieId, title, clean_title, year, genres, genres_array, num_genres)
    tags(userId, movieId, tag, timestamp, tag_ts)
    movie_stats(...)  - created by ``movie_popularity``
"""
from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def register_views(frames: dict[str, DataFrame]) -> None:
    for name in ("ratings", "movies", "tags", "links"):
        if name in frames:
            frames[name].createOrReplaceTempView(name)


def global_summary(spark: SparkSession) -> dict[str, Any]:
    row = spark.sql("""
        SELECT COUNT(*)                         AS total_ratings,
               COUNT(DISTINCT userId)           AS total_users,
               COUNT(DISTINCT movieId)          AS rated_movies,
               ROUND(AVG(rating), 4)            AS avg_rating,
               ROUND(STDDEV(rating), 4)         AS std_rating,
               MIN(rating_ts)                   AS first_rating,
               MAX(rating_ts)                   AS last_rating
        FROM ratings
    """).collect()[0].asDict()
    row["total_movies"] = spark.sql("SELECT COUNT(*) AS n FROM movies").collect()[0]["n"]
    row["total_tags"] = spark.sql("SELECT COUNT(*) AS n FROM tags").collect()[0]["n"]
    cells = row["total_users"] * row["rated_movies"]
    row["matrix_density_pct"] = round(100.0 * row["total_ratings"] / cells, 4) if cells else 0.0
    row["matrix_sparsity_pct"] = round(100.0 - row["matrix_density_pct"], 4)
    row["ratings_per_user_avg"] = round(row["total_ratings"] / max(row["total_users"], 1), 2)
    row["ratings_per_movie_avg"] = round(row["total_ratings"] / max(row["rated_movies"], 1), 2)
    return row


def rating_distribution(spark: SparkSession) -> DataFrame:
    return spark.sql("""
        SELECT rating, COUNT(*) AS num_ratings,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct
        FROM ratings GROUP BY rating ORDER BY rating
    """)


def movie_popularity(spark: SparkSession, min_count: int) -> DataFrame:
    """Per-movie statistics, a Bayesian weighted rating and popularity tiers.

    * ``weighted_rating`` = v/(v+m)*R + m/(v+m)*C, where v = number of ratings,
      R = the movie's mean, C = global mean, m = ``min_count``. It stops a film
      with three 5-star ratings from beating a film with 50,000 ratings at 4.4.
    * ``popularity_tier``: movies are sorted by number of ratings. The most
      popular movies that together collect the first 50% of all ratings are the
      **Head**, the next 30% are **Mid**, and the remaining 20% are the **Tail**.
    """
    stats = spark.sql(f"""
        WITH per_movie AS (
            SELECT movieId, COUNT(*) AS num_ratings, AVG(rating) AS avg_rating,
                   STDDEV(rating) AS std_rating, COUNT(DISTINCT userId) AS num_users,
                   MIN(rating_year) AS first_rated_year, MAX(rating_year) AS last_rated_year
            FROM ratings GROUP BY movieId
        ),
        g AS (SELECT AVG(rating) AS c FROM ratings),
        ranked AS (
            SELECT p.*,
                   (p.num_ratings / (p.num_ratings + {min_count})) * p.avg_rating
                     + ({min_count} / (p.num_ratings + {min_count})) * g.c          AS weighted_rating,
                   ROW_NUMBER() OVER (ORDER BY p.num_ratings DESC, p.movieId)       AS popularity_rank,
                   SUM(p.num_ratings) OVER (ORDER BY p.num_ratings DESC, p.movieId
                                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                     / SUM(p.num_ratings) OVER ()                                   AS cumulative_share,
                   p.num_ratings / SUM(p.num_ratings) OVER ()                       AS ratings_share
            FROM per_movie p CROSS JOIN g
        )
        -- (cumulative_share - ratings_share) = share of ratings held by MORE popular movies
        SELECT r.*, m.title, m.clean_title, m.year, m.genres,
               CASE WHEN r.cumulative_share - r.ratings_share < 0.5 THEN 'Head'
                    WHEN r.cumulative_share - r.ratings_share < 0.8 THEN 'Mid'
                    ELSE 'Tail' END AS popularity_tier
        FROM ranked r JOIN movies m ON r.movieId = m.movieId
    """)
    stats.createOrReplaceTempView("movie_stats")
    return stats


def most_rated(spark: SparkSession, limit: int = 50) -> DataFrame:
    return spark.sql(f"""
        SELECT movieId, title, genres, year, num_ratings, ROUND(avg_rating, 3) AS avg_rating
        FROM movie_stats ORDER BY num_ratings DESC LIMIT {limit}
    """)


def top_rated(spark: SparkSession, min_count: int, limit: int = 50) -> DataFrame:
    return spark.sql(f"""
        SELECT movieId, title, genres, year, num_ratings, ROUND(avg_rating, 3) AS avg_rating,
               ROUND(weighted_rating, 3) AS weighted_rating
        FROM movie_stats WHERE num_ratings >= {min_count}
        ORDER BY avg_rating DESC, num_ratings DESC LIMIT {limit}
    """)


def popularity_tiers(spark: SparkSession) -> DataFrame:
    return spark.sql("""
        SELECT popularity_tier, COUNT(*) AS num_movies, SUM(num_ratings) AS num_ratings,
               ROUND(AVG(avg_rating), 3) AS mean_of_movie_avgs,
               MIN(num_ratings) AS min_ratings_per_movie, MAX(num_ratings) AS max_ratings_per_movie
        FROM movie_stats GROUP BY popularity_tier
        ORDER BY CASE popularity_tier WHEN 'Head' THEN 1 WHEN 'Mid' THEN 2 ELSE 3 END
    """)


def long_tail_curve(spark: SparkSession, points: int = 200) -> DataFrame:
    """Cumulative share of ratings captured by the top x% of movies (Lorenz-style curve)."""
    return spark.sql(f"""
        WITH n AS (SELECT COUNT(*) AS total FROM movie_stats),
        pts AS (
            SELECT s.popularity_rank, s.num_ratings, s.cumulative_share,
                   s.popularity_rank / n.total AS movie_share,
                   NTILE({points}) OVER (ORDER BY s.popularity_rank) AS bucket
            FROM movie_stats s CROSS JOIN n
        )
        SELECT bucket, MAX(movie_share) AS movie_share, MAX(cumulative_share) AS ratings_share,
               MIN(num_ratings) AS min_ratings_in_bucket
        FROM pts GROUP BY bucket ORDER BY bucket
    """)


def genre_stats(spark: SparkSession) -> DataFrame:
    """Genre analysis: explode the genre array so each movie counts once per genre."""
    return spark.sql("""
        WITH movie_genre AS (
            SELECT movieId, genre FROM movies LATERAL VIEW explode(genres_array) g AS genre
        )
        SELECT mg.genre,
               COUNT(DISTINCT mg.movieId)      AS num_movies,
               SUM(COALESCE(s.num_ratings, 0)) AS num_ratings,
               ROUND(SUM(s.avg_rating * s.num_ratings) / SUM(s.num_ratings), 4) AS avg_rating,
               ROUND(AVG(s.num_ratings), 1)    AS avg_ratings_per_movie
        FROM movie_genre mg LEFT JOIN movie_stats s ON mg.movieId = s.movieId
        GROUP BY mg.genre ORDER BY num_ratings DESC
    """)


def genre_decade_share(spark: SparkSession) -> DataFrame:
    """Share of each genre among movies released in each decade."""
    return spark.sql("""
        WITH mg AS (
            SELECT movieId, CAST(FLOOR(year / 10) * 10 AS INT) AS decade, genre
            FROM movies LATERAL VIEW explode(genres_array) g AS genre
            WHERE year IS NOT NULL
        )
        SELECT decade, genre, COUNT(*) AS num_movies,
               ROUND(COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY decade), 4) AS share
        FROM mg GROUP BY decade, genre ORDER BY decade, num_movies DESC
    """)


def release_year_stats(spark: SparkSession) -> DataFrame:
    return spark.sql("""
        SELECT m.year, COUNT(DISTINCT m.movieId) AS num_movies,
               SUM(COALESCE(s.num_ratings, 0)) AS num_ratings,
               ROUND(SUM(s.avg_rating * s.num_ratings) / SUM(s.num_ratings), 4) AS avg_rating
        FROM movies m LEFT JOIN movie_stats s ON m.movieId = s.movieId
        WHERE m.year IS NOT NULL GROUP BY m.year ORDER BY m.year
    """)


def ratings_over_time(spark: SparkSession) -> DataFrame:
    return spark.sql("""
        SELECT rating_year, COUNT(*) AS num_ratings, ROUND(AVG(rating), 4) AS avg_rating,
               COUNT(DISTINCT userId) AS active_users, COUNT(DISTINCT movieId) AS movies_rated
        FROM ratings GROUP BY rating_year ORDER BY rating_year
    """)


def ratings_by_weekday_hour(spark: SparkSession) -> DataFrame:
    return spark.sql("""
        SELECT date_format(rating_ts, 'E') AS weekday, dayofweek(rating_ts) AS weekday_num,
               hour(rating_ts) AS hour_utc, COUNT(*) AS num_ratings
        FROM ratings GROUP BY 1, 2, 3 ORDER BY weekday_num, hour_utc
    """)


def user_behaviour(spark: SparkSession) -> tuple[DataFrame, DataFrame, dict[str, Any]]:
    """User rating behaviour: activity buckets, average-rating histogram, summary."""
    spark.sql("""
        SELECT userId, COUNT(*) AS n, AVG(rating) AS mean_rating, STDDEV(rating) AS std_rating,
               (MAX(timestamp) - MIN(timestamp)) / 86400.0 AS active_days
        FROM ratings GROUP BY userId
    """).createOrReplaceTempView("user_stats")

    activity = spark.sql("""
        SELECT CASE WHEN n < 20 THEN '01-19' WHEN n < 50 THEN '20-49' WHEN n < 100 THEN '50-99'
                    WHEN n < 200 THEN '100-199' WHEN n < 500 THEN '200-499'
                    WHEN n < 1000 THEN '500-999' ELSE '1000+' END AS bucket,
               MIN(n) AS sort_key, COUNT(*) AS num_users, SUM(n) AS num_ratings
        FROM user_stats GROUP BY 1 ORDER BY sort_key
    """)
    mean_hist = spark.sql("""
        SELECT ROUND(FLOOR(mean_rating * 4) / 4, 2) AS mean_rating_bin, COUNT(*) AS num_users
        FROM user_stats GROUP BY 1 ORDER BY 1
    """)
    summary = spark.sql("""
        SELECT ROUND(AVG(n), 2) AS mean_ratings_per_user,
               percentile_approx(n, 0.5) AS median_ratings_per_user,
               percentile_approx(n, 0.9) AS p90_ratings_per_user,
               MAX(n) AS max_ratings_per_user, MIN(n) AS min_ratings_per_user,
               ROUND(AVG(mean_rating), 4) AS mean_user_avg_rating,
               ROUND(AVG(std_rating), 4) AS mean_user_rating_std,
               ROUND(AVG(active_days), 1) AS mean_active_days,
               ROUND(100.0 * AVG(CASE WHEN mean_rating >= 4 THEN 1 ELSE 0 END), 2) AS pct_generous_users,
               ROUND(100.0 * AVG(CASE WHEN mean_rating < 3 THEN 1 ELSE 0 END), 2) AS pct_harsh_users
        FROM user_stats
    """).collect()[0].asDict()
    return activity, mean_hist, summary


def top_tags(spark: SparkSession, limit: int = 40) -> DataFrame:
    return spark.sql(f"""
        SELECT tag, COUNT(*) AS uses, COUNT(DISTINCT movieId) AS movies, COUNT(DISTINCT userId) AS users
        FROM tags GROUP BY tag ORDER BY uses DESC LIMIT {limit}
    """)


def movie_top_tags(spark: SparkSession, per_movie: int = 5) -> DataFrame:
    """The most-used tags for each movie, joined into one string (used for movie info cards)."""
    return spark.sql(f"""
        WITH c AS (SELECT movieId, tag, COUNT(*) AS uses FROM tags GROUP BY movieId, tag),
        r AS (SELECT movieId, tag, uses,
                     ROW_NUMBER() OVER (PARTITION BY movieId ORDER BY uses DESC, tag) AS rn FROM c)
        SELECT movieId,
               concat_ws(', ', transform(sort_array(collect_list(named_struct('rn', rn, 'tag', tag))),
                                         x -> x.tag)) AS top_tags
        FROM r WHERE rn <= {per_movie} GROUP BY movieId
    """)


def movie_catalogue(spark: SparkSession, links: DataFrame) -> DataFrame:
    """Everything the dashboard needs about a movie, one row per movie in movies.csv."""
    movie_top_tags(spark).createOrReplaceTempView("movie_tags")
    links.createOrReplaceTempView("links")
    return spark.sql("""
        SELECT m.movieId, m.title, m.clean_title, m.year, m.genres,
               COALESCE(s.num_ratings, 0) AS num_ratings, ROUND(s.avg_rating, 3) AS avg_rating,
               ROUND(s.weighted_rating, 3) AS weighted_rating, s.popularity_rank,
               COALESCE(s.popularity_tier, 'Unrated') AS popularity_tier,
               l.imdbId, l.tmdbId, t.top_tags
        FROM movies m
        LEFT JOIN movie_stats s ON m.movieId = s.movieId
        LEFT JOIN links l ON m.movieId = l.movieId
        LEFT JOIN movie_tags t ON m.movieId = t.movieId
    """)


def partition_report(df: DataFrame) -> dict[str, Any]:
    """Number of partitions and rows per partition (shows how work is distributed)."""
    sizes = df.withColumn("pid", F.spark_partition_id()).groupBy("pid").count().orderBy("pid").collect()
    counts = [int(r["count"]) for r in sizes]
    return {
        "num_partitions": df.rdd.getNumPartitions(),
        "non_empty_partitions": len(counts),
        "min_rows": min(counts) if counts else 0,
        "max_rows": max(counts) if counts else 0,
        "avg_rows": round(sum(counts) / len(counts), 1) if counts else 0,
    }
