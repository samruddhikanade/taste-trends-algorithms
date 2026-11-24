"""Streamlit dashboard entry point.

    streamlit run app.py

Reads the results written by ``scripts/run_pipeline.py``; it never starts Spark,
so it opens in seconds even for MovieLens 25M.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Taste, Trends & Algorithms", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

from dashboard import data as D  # noqa: E402
from dashboard import views  # noqa: E402
from dashboard.theme import inject_css  # noqa: E402

PAGES = {
    "Project Overview": views.overview,
    "Dataset Analytics": views.dataset,
    "Movie Popularity Analysis": views.popularity,
    "Genre Analysis": views.genres,
    "Personalised Recommendations": views.recommendations,
    "Model Performance": views.performance,
    "ALS vs Popularity-Based": views.comparison,
}


def main() -> None:
    inject_css()
    with st.sidebar:
        st.markdown("<div style='font-size:1.35rem;font-weight:800;letter-spacing:-.02em'>🎬 Taste, Trends "
                    "&amp; Algorithms</div><div style='color:#8A94AD;font-size:.8rem;margin-bottom:14px'>"
                    "Big Data Analytics · FR CRCE</div>", unsafe_allow_html=True)
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
        st.divider()
        modes = D.available_modes()
        if modes:
            mode = st.radio("Results from", modes, horizontal=True,
                            format_func=lambda m: "MovieLens 25M" if m == "full" else "Dev (small)",
                            help="'full' = MovieLens 25M run, 'dev' = ml-latest-small run")
        else:
            mode = "dev"
            st.caption("No results yet - run the pipeline first.")
        st.divider()
        st.caption("Samruddhi Kanade · Roll No. 10251\n\nPySpark · Spark SQL · MLlib ALS · Plotly · Streamlit")
    PAGES[page](mode)


main()
