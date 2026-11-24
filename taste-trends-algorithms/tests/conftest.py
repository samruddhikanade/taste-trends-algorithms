"""Shared pytest fixtures.

* ``tiny_movielens`` - a hand-made MovieLens-format folder with deliberately
  dirty rows (missing values, invalid ratings, a duplicate, an unknown movie).
* ``spark`` - a single-core local Spark session; tests that need it are
  skipped automatically when PySpark/Java are not available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RATINGS = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,2,5.0,964981247
1,3,3.5,964982224
2,1,4.5,1445714835
2,2,2.0,1445714885
2,4,4.0,1445715000
3,1,5.0,1700000000
3,3,1.0,1700000100
3,3,2.5,1700000500
3,99,4.0,1700000600
4,2,7.0,1700000700
4,,3.0,1700000800
5,4,3.3,1700000900
"""
# rows: 13 data rows
#  - (3,3) duplicated -> keep the later 2.5
#  - movie 99 is not in movies.csv -> removed
#  - rating 7.0 and 3.3 are invalid -> removed
#  - missing movieId -> removed

MOVIES = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,"American President, The (1995)",Comedy|Drama|Romance
3,Heat (1995),Action|Crime|Thriller
4,Some Movie Without Year,(no genres listed)
"""

TAGS = """userId,movieId,tag,timestamp
1,1,  Pixar ,964982703
2,1,pixar,1445714835
2,3,,1445714835
3,3,Al Pacino,1700000000
"""

LINKS = """movieId,imdbId,tmdbId
1,0114709,862
2,0112346,9087
3,0113277,949
4,0000001,
"""


@pytest.fixture
def tiny_movielens(tmp_path: Path) -> Path:
    folder = tmp_path / "ml-tiny"
    folder.mkdir()
    for name, text in (("ratings.csv", RATINGS), ("movies.csv", MOVIES), ("tags.csv", TAGS), ("links.csv", LINKS)):
        (folder / name).write_text(text, encoding="utf-8")
    return folder


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    import os
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    try:
        session = (SparkSession.builder.master("local[1]").appName("tests")
                   .config("spark.sql.shuffle.partitions", "2")
                   .config("spark.ui.enabled", "false")
                   .config("spark.sql.session.timeZone", "UTC")
                   .getOrCreate())
    except Exception as exc:  # Java missing or misconfigured
        pytest.skip(f"Spark could not start: {exc}")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
