"""Tests for MODULE A (data ingestion) and configuration."""
from __future__ import annotations

import pytest

from src.config import load_config


def test_config_modes_resolve_paths():
    for mode in ("dev", "full"):
        cfg = load_config(mode)
        assert cfg["mode"] == mode
        assert cfg["paths"]["results_dir"].name == mode
        assert cfg["paths"]["dataset_dir"].name == cfg["dataset"]["folder"]
        assert cfg["run"]["evaluation"]["k"] == 10


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError):
        load_config("huge")


def test_download_detects_missing_and_present_files(tiny_movielens, tmp_path):
    import sys

    from tests.conftest import ROOT
    sys.path.insert(0, str(ROOT / "scripts"))
    from download_data import dataset_ready  # noqa: E402

    assert dataset_ready(tiny_movielens)
    assert not dataset_ready(tmp_path / "does-not-exist")


def test_load_movielens_schemas_and_counts(spark, tiny_movielens):
    from src.ingestion import load_movielens

    frames = load_movielens(spark, tiny_movielens)
    assert set(frames) == {"ratings", "movies", "tags", "links"}
    assert dict(frames["ratings"].dtypes) == {"userId": "int", "movieId": "int", "rating": "double",
                                              "timestamp": "bigint"}
    assert frames["ratings"].count() == 13
    assert frames["movies"].count() == 4
    # quoted title with a comma must stay one field
    titles = {r["title"] for r in frames["movies"].collect()}
    assert "American President, The (1995)" in titles
    # imdbId is read as a string so leading zeros survive
    assert frames["links"].where("movieId = 1").first()["imdbId"] == "0114709"


def test_ingestion_report_finds_missing_and_duplicates(spark, tiny_movielens):
    from src.ingestion import ingestion_report, load_movielens

    report = ingestion_report(load_movielens(spark, tiny_movielens))
    ratings = report["files"]["ratings"]
    assert ratings["rows"] == 13
    assert ratings["missing_values"]["movieId"] == 1
    assert ratings["duplicate_rows_on_key"] == 1           # user 3 rated movie 3 twice
    assert report["users"] == 5
    assert report["files"]["links"]["missing_values"]["tmdbId"] == 1


def test_missing_file_gives_helpful_error(spark, tmp_path):
    from src.ingestion import RATINGS_SCHEMA, load_csv

    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_csv(spark, tmp_path / "nope.csv", RATINGS_SCHEMA)


def test_user_sampling_is_deterministic_and_keeps_whole_users(spark, tiny_movielens):
    from src.ingestion import load_movielens

    a = load_movielens(spark, tiny_movielens, sample_fraction=0.5, seed=7)["ratings"]
    b = load_movielens(spark, tiny_movielens, sample_fraction=0.5, seed=7)["ratings"]
    assert sorted(map(tuple, a.collect())) == sorted(map(tuple, b.collect()))
    full = load_movielens(spark, tiny_movielens)["ratings"]
    kept_users = {r["userId"] for r in a.collect()}
    for u in kept_users:  # every rating of a kept user is kept
        assert a.where(f"userId = {u}").count() == full.where(f"userId = {u}").count()
