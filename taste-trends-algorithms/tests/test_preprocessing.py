"""Tests for MODULE B (preprocessing)."""
from __future__ import annotations

import re

from src.constants import YEAR_PATTERN


def test_year_pattern_matches_movielens_titles():
    cases = {"Toy Story (1995)": "1995", "Heat (1995) ": "1995", "Se7en (a.k.a. Seven) (1995)": "1995",
             "Movie Without Year": None, "Babylon 5": None, "2001: A Space Odyssey (1968)": "1968"}
    for title, year in cases.items():
        m = re.search(YEAR_PATTERN, title)
        assert (m.group(1) if m else None) == year, title


def test_clean_ratings_applies_every_rule(spark, tiny_movielens):
    from src.ingestion import load_movielens
    from src.preprocessing import clean_movies, clean_ratings

    frames = load_movielens(spark, tiny_movielens)
    movies = clean_movies(frames["movies"])
    clean, stats = clean_ratings(frames["ratings"], movies)
    assert stats == {"input_rows": 13, "removed_missing": 1, "removed_invalid": 2, "removed_duplicates": 1,
                     "output_rows": 8, "removed_unknown_movie": 1}
    rows = {(r["userId"], r["movieId"]): r for r in clean.collect()}
    assert rows[(3, 3)]["rating"] == 2.5                 # the later duplicate is kept
    assert (3, 99) not in rows                           # unknown movie removed
    assert rows[(1, 1)]["rating_year"] == 2000           # 964982703 = 30 Jul 2000 UTC
    assert rows[(1, 1)]["rating_month"] == "2000-07"
    assert dict(clean.dtypes)["rating"] == "float"


def test_clean_movies_parses_year_title_and_genres(spark, tiny_movielens):
    from src.ingestion import load_movielens
    from src.preprocessing import clean_movies

    movies = {r["movieId"]: r for r in clean_movies(load_movielens(spark, tiny_movielens)["movies"]).collect()}
    assert movies[1]["year"] == 1995 and movies[1]["clean_title"] == "Toy Story"
    assert movies[1]["genres_array"] == ["Adventure", "Animation", "Children", "Comedy", "Fantasy"]
    assert movies[4]["year"] is None
    assert movies[4]["genres"] == "" and movies[4]["num_genres"] == 0


def test_clean_tags_normalises_text(spark, tiny_movielens):
    from src.ingestion import load_movielens
    from src.preprocessing import clean_tags

    tags = [r["tag"] for r in clean_tags(load_movielens(spark, tiny_movielens)["tags"]).collect()]
    assert sorted(tags) == ["al pacino", "pixar", "pixar"]


def test_prepare_and_split(spark, tiny_movielens):
    from src.ingestion import load_movielens
    from src.preprocessing import clean_movies, clean_ratings, prepare_ml_data, split_ratings

    frames = load_movielens(spark, tiny_movielens)
    clean, _ = clean_ratings(frames["ratings"], clean_movies(frames["movies"]))
    ml = prepare_ml_data(clean, 3)
    assert ml.columns == ["userId", "movieId", "rating"]
    assert ml.rdd.getNumPartitions() == 3
    train, test = split_ratings(ml, 0.25, seed=1)
    assert train.count() + test.count() == 8
