"""The ZIP must contain the code and docs but never the raw data, venvs or secrets."""
from __future__ import annotations

import sys

from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from package_zip import included  # noqa: E402


def test_zip_rules():
    assert included("src/recommendation.py")
    assert included("reports/results/full/metrics.json")
    assert included("data/raw/.gitkeep")
    assert not included("data/raw/ml-25m/ratings.csv")
    assert not included("data/processed/full/user_factors.parquet")
    assert not included("models/full/als_metadata.json")
    assert not included(".venv/Lib/site-packages/x.py")
    assert not included("src/__pycache__/a.cpython-311.pyc")
    assert not included("config/config.yaml")
    assert included("config/config.example.yaml")
    assert not included("Samruddhi_Kanade_BDA_Project.zip")
