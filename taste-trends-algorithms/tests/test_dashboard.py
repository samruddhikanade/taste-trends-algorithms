"""Headless dashboard test: renders every page with Streamlit's AppTest on a
synthetic results folder and fails on any exception.

Skipped automatically when Streamlit/Plotly are not installed.
"""
from __future__ import annotations

import pytest

from tests.conftest import ROOT

PAGES = ["Project Overview", "Dataset Analytics", "Movie Popularity Analysis", "Genre Analysis",
         "Personalised Recommendations", "Model Performance", "ALS vs Popularity-Based"]


@pytest.fixture
def synthetic_config(tmp_path, monkeypatch):
    from tests.synthetic_results import build

    cfg = build(tmp_path)
    monkeypatch.setenv("TTA_CONFIG", str(cfg))
    return cfg


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page, synthetic_config):
    pytest.importorskip("plotly")
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_without_results_shows_instructions(tmp_path, monkeypatch):
    pytest.importorskip("plotly")
    testing = pytest.importorskip("streamlit.testing.v1")
    import yaml

    from src.config import CONFIG_DIR
    cfg = yaml.safe_load((CONFIG_DIR / "config.example.yaml").read_text(encoding="utf-8"))
    for key in ("raw_dir", "processed_dir", "models_dir", "results_dir", "figures_dir", "spark_tmp_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("TTA_CONFIG", str(tmp_path / "c.yaml"))
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    for page in PAGES[1:]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
