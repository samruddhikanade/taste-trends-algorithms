"""The seven dashboard sections. Each function renders one page for a mode."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import data as D
from dashboard.theme import (AMBER, BLUE, MINT, MUTED, card, empty_state, fmt_compact, fmt_int, kpi_row,
                             movie_row, page_header, plot)
from src import visuals as V
from src.recommender_core import recommend_popular

RUN = "python scripts/run_pipeline.py --mode {mode}"


def _pct(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{100 * v:.{digits}f}%"


def _imdb(imdb_id) -> str:
    if imdb_id is None or (isinstance(imdb_id, float) and np.isnan(imdb_id)):
        return ""
    return f"https://www.imdb.com/title/tt{str(imdb_id).split('.')[0].zfill(7)}/"


# =============================================================== 1. overview
def overview(mode: str) -> None:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Big Data Analytics · Mini Project</div>
      <h1>Taste, Trends &amp; Algorithms</h1>
      <div class="sub">Big Data-Driven Movie Recommendations and Popularity Bias Analysis</div>
      <div class="meta"><b>Samruddhi Kanade</b> · Roll No. 10251 · Fr. Conceicao Rodrigues College of Engineering
      (FR CRCE) · Academic Year 2026-27</div>
    </div>""", unsafe_allow_html=True)

    summary = D.json_file(mode, "eda/summary.json")
    metrics = D.json_file(mode, "metrics.json")
    if summary:
        kpi_row([
            ("Ratings processed", fmt_compact(summary["total_ratings"]), f"{fmt_int(summary['total_ratings'])} rows", BLUE),
            ("Users", fmt_compact(summary["total_users"]), "unique raters", MINT),
            ("Movies", fmt_compact(summary["total_movies"]), f"{fmt_int(summary['rated_movies'])} rated", AMBER),
            ("Matrix sparsity", f"{summary['matrix_sparsity_pct']:.2f}%", "of user × movie cells are empty", BLUE),
        ])
    if metrics:
        als = metrics["models"]["ALS"]
        kpi_row([
            ("ALS RMSE", f"{metrics['rmse']['als']:.4f}", "error in stars (test set)", MINT),
            ("Precision@10", f"{als['precision_at_10']:.4f}", "share of top-10 the user liked", BLUE),
            ("Catalogue coverage", _pct(als["coverage"]), "of movies ever recommended", AMBER),
            ("Head-tier share", _pct(als["head_share"]), "of ALS recommendation slots", AMBER),
        ])
    if not summary and not metrics:
        empty_state("No pipeline results found yet.", RUN.format(mode=mode))

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Research questions")
        card("1 · Understanding taste at scale",
             "How can user preferences be analysed from millions of ratings with distributed Spark processing?")
        card("2 · Personalised recommendations",
             "Can collaborative filtering (ALS matrix factorisation) recommend unseen movies to each user?")
        card("3 · Popularity bias",
             "Do the algorithm's recommendations favour already-popular movies, and by how much?")
        card("4 · Personal vs popular",
             "How do personalised recommendations compare with a simple 'most popular' list?")
    with right:
        st.markdown("#### Pipeline")
        st.markdown("""
        <div class="flow">
          <div class="step">MovieLens 25M<span>ratings · movies · tags · links</span></div><div class="arrow">→</div>
          <div class="step">Spark ingestion<span>explicit schemas, quality checks</span></div><div class="arrow">→</div>
          <div class="step">Preprocessing<span>clean · dedupe · timestamps</span></div><div class="arrow">→</div>
          <div class="step">Spark SQL analytics<span>genres · popularity · trends</span></div><div class="arrow">→</div>
          <div class="step">ALS (MLlib)<span>tuning · training · top-10</span></div><div class="arrow">→</div>
          <div class="step">Evaluation<span>RMSE · P@10 · R@10 · bias</span></div><div class="arrow">→</div>
          <div class="step">This dashboard<span>Streamlit + Plotly</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("#### Research foundation")
        card("Matrix Factorization Techniques for Recommender Systems",
             "Y. Koren, R. Bell, C. Volinsky · <i>IEEE Computer</i> 42(8), 2009 · DOI 10.1109/MC.2009.263. "
             "Users and movies are mapped to a shared space of latent factors; a rating is predicted by the dot "
             "product of the two vectors, and the factors are learned with Alternating Least Squares. This project "
             "is an independent implementation of these concepts with Spark MLlib - it does not reproduce the "
             "paper's Netflix Prize experiments.")
        run_info = D.json_file(mode, "run_info.json")
        if run_info:
            st.caption(f"Results from a `{mode}` run · Spark {run_info.get('spark_version')} · "
                       f"total pipeline time {run_info['timings_seconds'].get('total', '-')} s")


# =============================================================== 2. dataset analytics
def dataset(mode: str) -> None:
    page_header("Module A · B · C", "Dataset Analytics",
                "Scale, quality and shape of the MovieLens data, computed with distributed Spark aggregations.")
    summary = D.json_file(mode, "eda/summary.json")
    if not summary:
        empty_state("EDA results not found.", RUN.format(mode=mode))
        return
    ub = summary.get("user_behaviour", {})
    kpi_row([
        ("Total ratings", fmt_compact(summary["total_ratings"]), fmt_int(summary["total_ratings"]), BLUE),
        ("Total users", fmt_compact(summary["total_users"]), f"median {fmt_int(ub.get('median_ratings_per_user'))} ratings each", MINT),
        ("Total movies", fmt_compact(summary["total_movies"]), f"{fmt_int(summary['rated_movies'])} with ≥ 1 rating", AMBER),
        ("Average rating", f"{summary['avg_rating']:.2f} ★", f"σ = {summary['std_rating']:.2f}", BLUE),
        ("Tags", fmt_compact(summary["total_tags"]), "free-text user tags", MINT),
    ])

    tab1, tab2, tab3, tab4 = st.tabs(["Ratings", "Trends over time", "User behaviour", "Data quality"])
    with tab1:
        c1, c2 = st.columns(2)
        dist = D.eda_table(mode, "rating_distribution")
        if dist is not None:
            with c1:
                plot(V.rating_distribution(dist))
        hist = D.eda_table(mode, "user_mean_rating_hist")
        if hist is not None:
            with c2:
                plot(V.user_mean_hist(hist))
        if dist is not None:
            top = dist.sort_values("num_ratings", ascending=False).iloc[0]
            half = dist[(dist["rating"] * 2) % 2 == 1]["pct"].sum()
            st.caption(f"Most common rating: **{top['rating']} stars** ({top['pct']:.1f}% of all ratings). "
                       f"Half-star ratings make up {half:.1f}%.")
    with tab2:
        over = D.eda_table(mode, "ratings_over_time")
        if over is not None:
            plot(V.ratings_over_time(over))
            c1, c2 = st.columns(2)
            with c1:
                plot(V.active_users_over_time(over))
            wh = D.eda_table(mode, "ratings_by_weekday_hour")
            if wh is not None:
                with c2:
                    plot(V.weekday_hour_heatmap(wh))
        st.caption(f"Ratings span {str(summary['first_rating'])[:10]} to {str(summary['last_rating'])[:10]}.")
    with tab3:
        c1, c2 = st.columns([1.3, 1])
        act = D.eda_table(mode, "user_activity")
        if act is not None:
            with c1:
                plot(V.user_activity(act))
        with c2:
            st.markdown("#### Rating behaviour")
            rows = [("Mean ratings per user", f"{ub.get('mean_ratings_per_user', 0):,.1f}"),
                    ("Median ratings per user", fmt_int(ub.get("median_ratings_per_user"))),
                    ("90th percentile", fmt_int(ub.get("p90_ratings_per_user"))),
                    ("Most ratings by one user", fmt_int(ub.get("max_ratings_per_user"))),
                    ("Average of user averages", f"{ub.get('mean_user_avg_rating', 0):.3f}"),
                    ("Generous users (avg ≥ 4)", f"{ub.get('pct_generous_users', 0):.1f}%"),
                    ("Harsh users (avg < 3)", f"{ub.get('pct_harsh_users', 0):.1f}%"),
                    ("Mean active span", f"{ub.get('mean_active_days', 0):,.0f} days")]
            st.dataframe(pd.DataFrame(rows, columns=["Measure", "Value"]), hide_index=True, use_container_width=True)
            st.caption("A long right tail: a small group of very active users contributes a large share of ratings.")
    with tab4:
        ing = D.json_file(mode, "ingestion_report.json")
        prep = D.json_file(mode, "preprocessing_report.json")
        if ing:
            rows = []
            for name, info in ing["files"].items():
                rows.append({"File": info["file"], "Rows": info["rows"], "Partitions": info["partitions"],
                             "Missing values": sum(info["missing_values"].values()),
                             "Duplicates on key": info["duplicate_rows_on_key"],
                             "Schema": "; ".join(s.split(" (")[0] for s in info["schema"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if prep:
            r = prep["ratings"]
            kpi_row([
                ("Ratings in", fmt_int(r["input_rows"]), "", BLUE),
                ("Removed: missing", fmt_int(r["removed_missing"]), "", MUTED),
                ("Removed: invalid", fmt_int(r["removed_invalid"]), "outside 0.5-5 or bad time", MUTED),
                ("Removed: duplicates", fmt_int(r["removed_duplicates"]), "same user + movie", MUTED),
                ("Ratings out", fmt_int(r["output_rows"]), "clean", MINT),
            ])
            p = prep.get("partitioning", {})
            st.caption(f"ML table hash-partitioned by userId into {p.get('num_partitions')} partitions "
                       f"({fmt_int(p.get('min_rows'))}-{fmt_int(p.get('max_rows'))} rows each). "
                       f"Movies without a parsable release year: {fmt_int(prep['movies']['without_year'])}; "
                       f"without a genre: {fmt_int(prep['movies']['without_genre'])}.")


# =============================================================== 3. popularity
def popularity(mode: str) -> None:
    page_header("Module D", "Movie Popularity Analysis",
                "Which movies attract attention, how concentrated that attention is, and a searchable movie explorer.")
    summary = D.json_file(mode, "eda/summary.json")
    if not summary:
        empty_state("Popularity results not found.", RUN.format(mode=mode))
        return
    tiers = D.eda_table(mode, "popularity_tiers")
    if tiers is not None:
        t = tiers.set_index("popularity_tier")
        total_movies = tiers["num_movies"].sum()
        items = []
        for name, accent in (("Head", AMBER), ("Mid", BLUE), ("Tail", MINT)):
            if name in t.index:
                items.append((f"{name} tier", f"{100 * t.loc[name, 'num_movies'] / total_movies:.1f}% of movies",
                              f"{fmt_int(t.loc[name, 'num_movies'])} movies · "
                              f"{100 * t.loc[name, 'num_ratings'] / tiers['num_ratings'].sum():.0f}% of ratings", accent))
        kpi_row(items)

    tab1, tab2, tab3 = st.tabs(["Most rated & highest rated", "Long tail", "Movie search"])
    with tab1:
        c1, c2 = st.columns(2)
        mr = D.eda_table(mode, "most_rated")
        tr = D.eda_table(mode, "top_rated")
        n = c1.slider("Movies to show", 10, 50, 15, 5, key="pop_n")
        if mr is not None:
            with c1:
                plot(V.top_movies_bar(mr, "num_ratings", "Most rated movies", n=n))
        if tr is not None:
            with c2:
                c2.write("")
                c2.write("")
                plot(V.top_movies_bar(tr, "avg_rating", f"Highest rated (≥ {fmt_int(summary['min_ratings_top_rated'])} "
                                                        "ratings)", n=n, color=MINT))
        st.caption("A minimum rating count is required for 'highest rated', otherwise movies with a single "
                   "5-star rating would top the list.")
    with tab2:
        lt = D.eda_table(mode, "long_tail_curve")
        c1, c2 = st.columns([1.2, 1])
        if lt is not None:
            with c1:
                plot(V.long_tail(lt))
            share_at = lambda x: float(np.interp(x, lt["movie_share"], lt["ratings_share"]))  # noqa: E731
            st.caption(f"The top 1% of movies receive **{100 * share_at(0.01):.1f}%** of all ratings; the top 10% "
                       f"receive **{100 * share_at(0.10):.1f}%**.")
        if tiers is not None:
            with c2:
                plot(V.tier_summary(tiers))
        cat = D.catalogue(mode)
        if cat is not None:
            rated = cat[cat["num_ratings"] > 0]
            sample = rated if len(rated) <= 6000 else rated.sample(6000, random_state=1)
            plot(V.popularity_vs_quality(sample))
        card("How tiers are defined", "Movies are sorted from most to least rated. The most popular movies that "
             "together collect the first <b>50%</b> of all ratings form the <b>Head</b>; the next 30% of ratings "
             "come from the <b>Mid</b> tier; the remaining 20% are spread over the long <b>Tail</b>.")
    with tab3:
        _movie_search(mode)


def _movie_search(mode: str) -> None:
    cat = D.catalogue(mode)
    if cat is None:
        empty_state("Movie catalogue not found.", RUN.format(mode=mode))
        return
    query = st.text_input("Search for a movie by title", placeholder="e.g. Matrix, Toy Story, Amélie")
    if not query:
        st.caption("Type part of a title. Results are ordered by number of ratings.")
        return
    hits = cat[cat["title"].str.contains(query, case=False, regex=False, na=False)].sort_values(
        "num_ratings", ascending=False)
    if hits.empty:
        st.info("No movie matches that title.")
        return
    options = hits["movieId"].head(50).tolist()
    choice = st.selectbox("Matches", options, format_func=lambda m: f"{cat.loc[m, 'title']}  ·  "
                                                                    f"{fmt_int(cat.loc[m, 'num_ratings'])} ratings")
    m = cat.loc[choice]
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown(f"### {html.escape(str(m['title']))}")
        st.markdown(f'<span class="pill {m["popularity_tier"]}">{m["popularity_tier"]}</span> '
                    f'{html.escape(str(m["genres"]).replace("|", " · "))}', unsafe_allow_html=True)
        kpi_row([("Ratings", fmt_int(m["num_ratings"]), f"popularity rank #{fmt_int(m['popularity_rank'])}", BLUE),
                 ("Average", "-" if pd.isna(m["avg_rating"]) else f"{m['avg_rating']:.2f} ★",
                  "-" if pd.isna(m["weighted_rating"]) else f"weighted {m['weighted_rating']:.2f}", AMBER)])
        if isinstance(m.get("top_tags"), str) and m["top_tags"]:
            st.markdown(f"**Top user tags:** {html.escape(m['top_tags'])}")
        links = []
        if _imdb(m.get("imdbId")):
            links.append(f"[IMDb]({_imdb(m['imdbId'])})")
        if not pd.isna(m.get("tmdbId")):
            links.append(f"[TMDB](https://www.themoviedb.org/movie/{int(m['tmdbId'])})")
        if links:
            st.markdown(" · ".join(links))
    with c2:
        fm = D.factor_model(mode)
        st.markdown("#### Similar movies (ALS latent factors)")
        if fm is None:
            st.caption("Train the model to see similar movies.")
        elif not fm.has_item(int(choice)):
            st.caption("This movie had no ratings in the training split, so ALS has no factors for it (cold start).")
        else:
            sims = fm.similar_items(int(choice), 8)
            rows = "".join(movie_row(int(r["rank"]), cat.loc[int(r["movieId"]), "title"],
                                     cat.loc[int(r["movieId"]), "genres"], float(r["score"]), "cosine",
                                     cat.loc[int(r["movieId"]), "popularity_tier"]) for _, r in sims.iterrows()
                           if int(r["movieId"]) in cat.index)
            st.markdown(rows, unsafe_allow_html=True)


# =============================================================== 4. genres
def genres(mode: str) -> None:
    page_header("Module D", "Genre Analysis",
                "Genres are stored as pipe-separated text; Spark SQL explodes them so each movie counts once per genre.")
    gs = D.eda_table(mode, "genre_stats")
    if gs is None:
        empty_state("Genre results not found.", RUN.format(mode=mode))
        return
    top = gs.sort_values("num_ratings", ascending=False).iloc[0]
    best = gs[gs["num_movies"] >= 50].sort_values("avg_rating", ascending=False)
    worst = gs[gs["num_movies"] >= 50].sort_values("avg_rating")
    kpi_row([
        ("Genres", fmt_int(len(gs)), "distinct labels", BLUE),
        ("Most rated genre", str(top["genre"]), f"{fmt_compact(top['num_ratings'])} ratings", AMBER),
        ("Best rated genre", str(best.iloc[0]["genre"]) if len(best) else "-",
         f"avg {best.iloc[0]['avg_rating']:.2f} (≥ 50 movies)" if len(best) else "", MINT),
        ("Lowest rated genre", str(worst.iloc[0]["genre"]) if len(worst) else "-",
         f"avg {worst.iloc[0]['avg_rating']:.2f}" if len(worst) else "", MUTED),
    ])
    c1, c2 = st.columns([1.1, 1])
    with c1:
        plot(V.genre_bars(gs))
    with c2:
        plot(V.genre_quality_scatter(gs))
    gd = D.eda_table(mode, "genre_decade_share")
    if gd is not None:
        plot(V.genre_decade_heatmap(gd))

    st.markdown("#### Best movies in a genre")
    cat = D.catalogue(mode)
    summary = D.json_file(mode, "eda/summary.json") or {}
    if cat is not None:
        c1, c2 = st.columns([1, 1])
        genre = c1.selectbox("Genre", sorted(gs["genre"].tolist()))
        min_n = c2.number_input("Minimum ratings", min_value=1, value=int(summary.get("min_ratings_top_rated", 50)),
                                step=10)
        sub = cat[cat["genres"].str.split("|").apply(lambda g: genre in g) & (cat["num_ratings"] >= min_n)]
        sub = sub.sort_values("weighted_rating", ascending=False).head(15)
        st.dataframe(sub[["title", "genres", "num_ratings", "avg_rating", "weighted_rating", "popularity_tier"]],
                     hide_index=True, use_container_width=True,
                     column_config={"num_ratings": st.column_config.NumberColumn("Ratings", format="%d"),
                                    "avg_rating": st.column_config.NumberColumn("Average", format="%.2f"),
                                    "weighted_rating": st.column_config.NumberColumn("Weighted", format="%.2f"),
                                    "popularity_tier": "Tier", "title": "Title", "genres": "Genres"})
    with st.expander("Full genre table"):
        st.dataframe(gs, hide_index=True, use_container_width=True)


# =============================================================== 5. recommendations
def recommendations(mode: str) -> None:
    page_header("Module E", "Personalised Recommendations",
                "Pick a user and ALS returns the 10 unseen movies with the highest predicted rating. "
                "Unknown users get a popularity list or can build a taste profile (cold start).")
    cat, recs, fm = D.catalogue(mode), D.als_recommendations(mode), D.factor_model(mode)
    if cat is None or recs is None or fm is None:
        empty_state("The ALS model has not been trained yet.", f"python scripts/train_model.py --mode {mode}")
        return
    history = D.demo_history(mode)
    demo_users = sorted(history["userId"].unique().tolist()) if history is not None else []

    c1, c2, c3 = st.columns([1.1, 1, 1])
    how = c1.radio("Choose a user", ["Pick from demo users", "Type any user ID", "New user (cold start)"],
                   horizontal=False)
    user_id = None
    if how == "Pick from demo users" and demo_users:
        user_id = int(c2.selectbox("Demo user (history available)", demo_users))
    elif how == "Type any user ID":
        user_id = int(c2.number_input("User ID", min_value=1, value=int(fm.user_ids[0]), step=1))
    k = int(c3.slider("Number of recommendations", 5, 20, 10))

    if how == "New user (cold start)":
        _cold_start(mode, cat, fm, k)
        return
    if user_id is None:
        st.info("No demo users were saved - type a user ID instead.")
        return
    if not fm.has_user(user_id):
        st.warning(f"User {user_id} is not in the training data (an unseen user). ALS has no latent vector for "
                   f"them, so we fall back to the most highly rated popular movies. Choose **New user (cold start)** "
                   f"to build a profile instead.")
        _render_list(cat, _popular_fallback(mode, k, ()), "weighted")
        return

    seen = _seen_items(user_id, history, D.seen_for_short_users(mode))
    pre = recs[recs["userId"] == user_id].sort_values("rank")
    if len(pre) >= k and k <= 10:
        top = pre.head(k)[["movieId", "score", "rank"]]
    else:  # need more than the precomputed top-10, or a heavy rater: score live from the factors
        if seen is None:
            st.caption("Showing the stored top-10 (rated-movie list for this user is not stored locally).")
            top = pre[["movieId", "score", "rank"]]
        else:
            top = fm.recommend(user_id, k, exclude=seen)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(f"#### Top {len(top)} for user {user_id}")
        _render_list(cat, top, "predicted")
    with right:
        tiers = cat.reindex(top["movieId"])["popularity_tier"].value_counts(normalize=True)
        kpi_row([("Head-tier share", _pct(tiers.get("Head", 0.0), 0), "of this list", AMBER),
                 ("Tail-tier share", _pct(tiers.get("Tail", 0.0), 0), "of this list", MINT)])
        g = cat.reindex(top["movieId"])["genres"].str.split("|").explode().value_counts().head(8)
        if len(g):
            gdf = g.rename_axis("genre").reset_index(name="count")
            st.markdown("##### Genres in these recommendations")
            st.bar_chart(gdf.set_index("genre"), color=BLUE, height=220)
        if history is not None and user_id in demo_users:
            h = history[history["userId"] == user_id].merge(cat[["movieId", "title", "genres"]].reset_index(drop=True),
                                                             on="movieId", how="left")
            st.markdown(f"##### What user {user_id} rated ({len(h)} movies)")
            st.dataframe(h.sort_values("rating", ascending=False)[["title", "rating", "genres"]].head(25),
                         hide_index=True, use_container_width=True, height=330)

    with st.expander("Compare with the popularity-based list for the same user"):
        pop = _popular_list(mode, k, seen or ())
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ALS (personalised)**")
            _render_list(cat, top, "predicted", compact=True)
        with c2:
            st.markdown("**Most popular (same list for everyone)**")
            _render_list(cat, pop, None, compact=True)
        overlap = len(set(top["movieId"]) & set(pop["movieId"]))
        st.caption(f"Overlap between the two lists: {overlap} of {k} movies."
                   + ("" if seen is not None else " (This user's history is not stored locally, so the popularity "
                                                  "list does not exclude movies they have already rated.)"))


def _seen_items(user_id: int, history: pd.DataFrame | None, short: pd.DataFrame | None) -> list[int] | None:
    for df in (history, short):
        if df is not None and len(df) and (df["userId"] == user_id).any():
            return df.loc[df["userId"] == user_id, "movieId"].astype(int).tolist()
    return None


def _popular_list(mode: str, k: int, exclude) -> pd.DataFrame:
    ids = recommend_popular(D.most_popular_ranking(mode), k, exclude)
    return pd.DataFrame({"movieId": ids, "score": np.nan, "rank": np.arange(1, len(ids) + 1)})


def _popular_fallback(mode: str, k: int, exclude) -> pd.DataFrame:
    summary = D.json_file(mode, "eda/summary.json") or {}
    ranking = D.popular_ranking(mode, int(summary.get("min_ratings_top_rated", 50)))
    ids = recommend_popular(ranking, k, exclude)
    cat = D.catalogue(mode)
    return pd.DataFrame({"movieId": ids, "score": cat.reindex(ids)["weighted_rating"].to_numpy(),
                         "rank": np.arange(1, len(ids) + 1)})


def _render_list(cat: pd.DataFrame, df: pd.DataFrame, score_kind: str | None, compact: bool = False) -> None:
    rows = []
    for _, r in df.iterrows():
        mid = int(r["movieId"])
        if mid not in cat.index:
            continue
        m = cat.loc[mid]
        score = None
        label = ""
        if score_kind == "predicted" and not pd.isna(r["score"]):
            score, label = min(float(r["score"]), 5.0), "predicted ★"
        elif score_kind == "weighted" and not pd.isna(r["score"]):
            score, label = float(r["score"]), "weighted ★"
        extra = "" if compact else (f"{fmt_int(m['num_ratings'])} ratings · avg "
                                    f"{'-' if pd.isna(m['avg_rating']) else format(m['avg_rating'], '.2f')}"
                                    + (f" · <a href='{_imdb(m['imdbId'])}' target='_blank'>IMDb</a>"
                                       if _imdb(m.get("imdbId")) else ""))
        rows.append(movie_row(int(r["rank"]), str(m["title"]), str(m["genres"]), score, label,
                              str(m["popularity_tier"]), extra))
    st.markdown("".join(rows) or "_No movies to show._", unsafe_allow_html=True)


def _cold_start(mode: str, cat: pd.DataFrame, fm, k: int) -> None:
    st.markdown("#### New user: build a taste profile")
    st.caption("Rate a few movies you know. The app solves one ALS step for a new user vector with all movie "
               "factors held fixed (\"fold-in\"), then recommends from it - no retraining needed.")
    pool = cat[cat["movieId"].map(fm.has_item) & (cat["num_ratings"] > 0)].sort_values("num_ratings", ascending=False)
    picks = st.multiselect("Movies you have seen (type to search)", pool["movieId"].head(5000).tolist(),
                           format_func=lambda m: str(cat.loc[m, "title"]), max_selections=15)
    ratings = {}
    if picks:
        cols = st.columns(min(len(picks), 3))
        for i, mid in enumerate(picks):
            ratings[int(mid)] = cols[i % len(cols)].slider(str(cat.loc[mid, "title"])[:48], 0.5, 5.0, 4.0, 0.5,
                                                          key=f"cs_{mid}")
    if len(ratings) < 3:
        st.info("Rate at least 3 movies for a personalised list. Until then, here are popular, highly rated movies:")
        _render_list(cat, _popular_fallback(mode, k, ratings.keys()), "weighted")
        return
    vec = fm.fold_in(ratings)
    top = fm.recommend_vector(vec, k, exclude=ratings.keys())
    c1, c2 = st.columns([1.35, 1])
    with c1:
        _render_list(cat, top, "predicted")
    with c2:
        tiers = cat.reindex(top["movieId"])["popularity_tier"].value_counts(normalize=True)
        kpi_row([("Head-tier share", _pct(tiers.get("Head", 0.0), 0), "of this list", AMBER),
                 ("Movies rated", str(len(ratings)), "used for fold-in", BLUE)])
        plot(V.score_bars(top.assign(title=cat.reindex(top["movieId"])["title"].to_numpy(),
                                     score=np.minimum(top["score"], 5.0))))


# =============================================================== 6. model performance
METRIC_HELP = {
    "RMSE": "Root Mean Squared Error. For every hidden test rating we compare the predicted stars with the real "
            "stars, square the gaps, average them and take the square root. An RMSE of 0.80 means predictions are "
            "typically about 0.8 stars off. Lower is better.",
    "Precision@10": "Out of the 10 movies we recommend, how many did the user actually like (rated ≥ 4 in the "
                    "hidden test set)? 0.10 means 1 hit in 10. Values look small because users have only rated a "
                    "tiny fraction of the catalogue, so many good recommendations simply cannot be confirmed.",
    "Recall@10": "Out of all the movies the user liked in the test set, what fraction appeared in our top 10?",
    "NDCG@10": "Like precision, but a hit at position 1 is worth more than a hit at position 10.",
    "Hit rate@10": "Share of users for whom at least one of the 10 recommendations was a liked test movie.",
    "Catalogue coverage": "Share of all movies that are recommended to at least one user. Low coverage means the "
                          "system keeps showing the same small set of movies.",
    "Popularity bias": "Measured by ARP (average number of ratings of recommended movies), the share of "
                       "recommendation slots taken by Head-tier movies, and the Gini coefficient of exposure "
                       "(0 = every movie shown equally, 1 = one movie shown to everyone).",
}


def performance(mode: str) -> None:
    page_header("Module F", "Model Performance",
                "Real metrics from the trained ALS model on the held-out 20% test split.")
    metrics = D.json_file(mode, "metrics.json")
    if not metrics:
        empty_state("Evaluation results not found.", f"python scripts/train_model.py --mode {mode}")
        return
    meta = D.json_file(mode, "als_metadata.json") or {}
    r, als = metrics["rmse"], metrics["models"]["ALS"]
    improvement = (r["bias_baseline"] - r["als"]) / r["bias_baseline"] if r["bias_baseline"] else 0
    kpi_row([
        ("ALS RMSE", f"{r['als']:.4f}", f"{_pct(improvement)} better than bias baseline" if improvement > 0
         else f"{_pct(-improvement)} worse than bias baseline", MINT),
        ("Precision@10", f"{als['precision_at_10']:.4f}", f"{metrics['eval_users']:,} test users", BLUE),
        ("Recall@10", f"{als['recall_at_10']:.4f}", "", BLUE),
        ("NDCG@10", f"{als['ndcg_at_10']:.4f}", "", BLUE),
        ("Coverage", _pct(als["coverage"]), "sample of users", AMBER),
    ])
    if meta:
        st.caption(f"Best hyper-parameters: rank = **{meta.get('rank')}**, regParam (λ) = **{meta.get('reg_param')}**, "
                   f"maxIter = {meta.get('max_iter')}. Training time {meta.get('train_seconds')} s. Factors learned for "
                   f"{fmt_int(meta.get('users_with_factors'))} users and {fmt_int(meta.get('movies_with_factors'))} movies. "
                   f"{fmt_int(metrics['test_rows_cold_start_dropped'])} of {fmt_int(metrics['test_rows'])} test ratings "
                   "involved a user or movie never seen in training (cold start) and were excluded for every model.")
    c1, c2 = st.columns(2)
    rc = D.result_table(mode, "rmse_comparison")
    if rc is not None:
        with c1:
            plot(V.rmse_comparison(rc))
    tu = D.result_table(mode, "tuning_results")
    if tu is not None and len(tu):
        with c2:
            plot(V.tuning_heatmap(tu))
    comp = D.result_table(mode, "model_comparison")
    if comp is not None:
        plot(V.metric_comparison(comp, ["precision_at_10", "recall_at_10", "ndcg_at_10", "hit_rate_at_10"],
                                 "Top-10 ranking quality"))
    st.markdown("#### What do these metrics mean?")
    cols = st.columns(2)
    for i, (name, text) in enumerate(METRIC_HELP.items()):
        with cols[i % 2]:
            card(name, text)


# =============================================================== 7. ALS vs popularity
def comparison(mode: str) -> None:
    page_header("Research questions 3 & 4", "ALS vs Popularity-Based Recommendations",
                "Does personalisation beat a simple 'most popular' list - and does it inherit popularity bias?")
    metrics = D.json_file(mode, "metrics.json")
    comp = D.result_table(mode, "model_comparison")
    if not metrics or comp is None:
        empty_state("Comparison results not found.", f"python scripts/train_model.py --mode {mode}")
        return
    m = comp.set_index("model")
    als, pop = m.loc["ALS"], m.loc["Popularity"]
    kpi_row([
        ("P@10 · ALS vs Popular", f"{als['precision_at_10']:.3f} / {pop['precision_at_10']:.3f}",
         "higher is better", BLUE),
        ("Coverage · ALS vs Popular", f"{_pct(als['coverage'])} / {_pct(pop['coverage'])}", "of catalogue", MINT),
        ("ARP · ALS vs Popular", f"{fmt_compact(als['arp'])} / {fmt_compact(pop['arp'])}",
         "avg ratings of recommended movies", AMBER),
        ("Head share · ALS vs Popular", f"{_pct(als['head_share'])} / {_pct(pop['head_share'])}",
         "of recommendation slots", AMBER),
    ])

    c1, c2 = st.columns(2)
    with c1:
        plot(V.metric_comparison(comp, ["precision_at_10", "recall_at_10", "ndcg_at_10"], "Accuracy of top-10 lists"))
    with c2:
        plot(V.metric_comparison(comp, ["coverage", "head_share", "tail_share", "gini_exposure"],
                                 "Diversity & popularity bias"))
    te = D.result_table(mode, "tier_exposure")
    lz = D.result_table(mode, "exposure_lorenz")
    c1, c2 = st.columns(2)
    if te is not None:
        with c1:
            plot(V.tier_exposure(te))
    if lz is not None:
        with c2:
            plot(V.exposure_curve(lz))

    st.markdown("#### What the numbers say")
    lift = metrics.get("popularity_lift", {})
    te_idx = te.set_index(["source", "popularity_tier"])["share"] if te is not None else None
    head_ratings = float(te_idx.get(("All training ratings", "Head"), np.nan)) if te_idx is not None else np.nan
    better = "higher" if als["precision_at_10"] > pop["precision_at_10"] else "lower"
    points = [
        f"ALS precision@10 is <b>{als['precision_at_10']:.4f}</b> versus <b>{pop['precision_at_10']:.4f}</b> for the "
        f"popularity list, i.e. {better} - and random recommendations score "
        f"<b>{m.loc['Random', 'precision_at_10']:.4f}</b>." if "Random" in m.index else "",
        f"ALS recommends <b>{_pct(als['coverage'])}</b> of the catalogue across the evaluated users, compared with "
        f"<b>{_pct(pop['coverage'])}</b> for the popularity list, which shows almost the same movies to everyone.",
        f"Head-tier movies hold <b>{_pct(head_ratings)}</b> of all training ratings but take "
        f"<b>{_pct(als['head_share'])}</b> of ALS recommendation slots." if not np.isnan(head_ratings) else "",
        f"Popularity lift (average popularity of recommendations ÷ average popularity of what users actually rated): "
        f"ALS <b>{lift.get('ALS', 0):.2f}×</b>, Popularity <b>{lift.get('Popularity', 0):.2f}×</b>. A value above 1 "
        f"means the recommender pushes more popular movies than users chose on their own." if lift else "",
    ]
    card("Findings (computed from this run)", "<br><br>".join(p for p in points if p))
    if "all_users" in metrics:
        a = metrics["all_users"]
        st.caption(f"Across the stored top-10 lists for all {fmt_int(a['users'])} users, ALS covers "
                   f"{_pct(a['coverage'])} of the catalogue; {_pct(a['head_share'])} of slots are Head-tier and "
                   f"{_pct(a['tail_share'])} Tail-tier (Gini {a['gini_exposure']:.3f}).")
    st.dataframe(comp, hide_index=True, use_container_width=True)
