"""Cached loaders for everything the pipeline wrote to disk.

The dashboard never starts Spark: it reads the small aggregated tables and the
exported ALS factors. Each loader returns ``None`` when a file does not exist
yet, and the page shows instructions instead of invented numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.io_utils import read_json, read_table
from src.recommender_core import FactorModel, load_factor_model, popularity_ranking


@st.cache_data(show_spinner=False)
def paths(mode: str) -> dict[str, str]:
    cfg = load_config(mode)
    return {k: str(v) for k, v in cfg["paths"].items()}


def available_modes() -> list[str]:
    modes = []
    for mode in ("full", "dev"):
        p = paths(mode)
        if (Path(p["results_dir"]) / "eda" / "summary.json").exists() or (Path(p["results_dir"]) / "metrics.json").exists():
            modes.append(mode)
    return modes


@st.cache_data(show_spinner=False)
def json_file(mode: str, name: str) -> dict | None:
    return read_json(Path(paths(mode)["results_dir"]) / name)


@st.cache_data(show_spinner=False)
def eda_table(mode: str, name: str) -> pd.DataFrame | None:
    path = Path(paths(mode)["results_dir"]) / "eda" / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def result_table(mode: str, name: str) -> pd.DataFrame | None:
    path = Path(paths(mode)["results_dir"]) / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Loading movie catalogue ...")
def catalogue(mode: str) -> pd.DataFrame | None:
    df = read_table(Path(paths(mode)["processed_dir"]) / "movie_catalogue")
    if df is None:
        return None
    df["genres"] = df["genres"].fillna("")
    df["popularity_tier"] = df["popularity_tier"].fillna("Unrated")
    df["year_label"] = df["year"].apply(lambda y: "" if pd.isna(y) else str(int(y)))
    return df.set_index("movieId", drop=False)


@st.cache_data(show_spinner="Loading recommendations ...")
def als_recommendations(mode: str) -> pd.DataFrame | None:
    return read_table(Path(paths(mode)["processed_dir"]) / "als_recommendations")


@st.cache_data(show_spinner=False)
def demo_history(mode: str) -> pd.DataFrame | None:
    return read_table(Path(paths(mode)["processed_dir"]) / "demo_user_history")


@st.cache_data(show_spinner=False)
def seen_for_short_users(mode: str) -> pd.DataFrame | None:
    return read_table(Path(paths(mode)["processed_dir"]) / "seen_for_short_users")


@st.cache_resource(show_spinner="Loading ALS factors ...")
def factor_model(mode: str) -> FactorModel | None:
    meta = json_file(mode, "als_metadata.json") or {}
    return load_factor_model(paths(mode)["processed_dir"], reg_param=float(meta.get("reg_param", 0.1)))


@st.cache_data(show_spinner=False)
def popular_ranking(mode: str, min_ratings: int) -> np.ndarray:
    """Movies ordered by Bayesian weighted rating among sufficiently-rated movies,
    used for the cold-start (new user) list."""
    cat = catalogue(mode)
    if cat is None:
        return np.array([], dtype=np.int64)
    eligible = cat[(cat["num_ratings"] >= min_ratings)].sort_values(["weighted_rating", "num_ratings"],
                                                                      ascending=False)
    return eligible["movieId"].to_numpy(dtype=np.int64)


@st.cache_data(show_spinner=False)
def most_popular_ranking(mode: str) -> np.ndarray:
    cat = catalogue(mode)
    if cat is None:
        return np.array([], dtype=np.int64)
    return popularity_ranking(cat["movieId"].to_numpy(), cat["num_ratings"].to_numpy())
