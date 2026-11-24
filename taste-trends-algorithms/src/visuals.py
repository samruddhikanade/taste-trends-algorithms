"""Plotly chart factory with the project's dark cinematic theme.

Shared by the Streamlit dashboard and by ``eda.export_figures`` (which saves
the same charts as standalone HTML files in ``reports/figures``). Every
function takes a small pandas DataFrame produced by the Spark pipeline.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BG = "#0A0F1E"
BG2 = "#151D31"
BLUE = "#578BEE"
AMBER = "#F2AE37"
MINT = "#42C79D"
WHITE = "#EFF2F8"
MUTED = "#8A94AD"
GRID = "rgba(239,242,248,0.07)"
CORAL = "#E8697D"
SEQUENCE = [BLUE, AMBER, MINT, CORAL, "#A98BF0", "#5BC8E8", "#F07F4F", "#C9D36A"]
TIER_COLORS = {"Head": AMBER, "Mid": BLUE, "Tail": MINT, "Unrated": MUTED}
MODEL_COLORS = {"ALS": BLUE, "Popularity": AMBER, "Random": MUTED}
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"


def style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15, color=WHITE)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=WHITE, size=12),
        colorway=SEQUENCE,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h", y=1.02, x=1, xanchor="right",
                    yanchor="bottom"),
        hoverlabel=dict(bgcolor=BG2, bordercolor=BLUE, font=dict(color=WHITE, family=FONT)),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    return fig


# ---------------------------------------------------------------- dataset analytics
def rating_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="rating", y="num_ratings", custom_data=["pct"], color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="Rating %{x}<br>%{y:,} ratings (%{customdata[0]:.2f}%)<extra></extra>",
                      marker_line_width=0)
    fig.update_xaxes(title="Star rating", dtick=0.5)
    fig.update_yaxes(title="Number of ratings")
    return style(fig, "Rating distribution")


def ratings_over_time(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["rating_year"], y=df["num_ratings"], name="Ratings", marker_color=BLUE, opacity=0.85,
                hovertemplate="%{x}: %{y:,} ratings<extra></extra>")
    fig.add_scatter(x=df["rating_year"], y=df["avg_rating"], name="Average rating", yaxis="y2", mode="lines+markers",
                    line=dict(color=AMBER, width=2.5), marker=dict(size=5),
                    hovertemplate="%{x}: avg %{y:.2f}<extra></extra>")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Average rating",
                                  tickfont=dict(color=AMBER), title_font=dict(color=AMBER)))
    fig.update_xaxes(title="Year the rating was given")
    fig.update_yaxes(title="Ratings")
    return style(fig, "Rating activity and average rating over time")


def active_users_over_time(df: pd.DataFrame) -> go.Figure:
    fig = px.area(df, x="rating_year", y="active_users", color_discrete_sequence=[MINT])
    fig.update_traces(hovertemplate="%{x}: %{y:,} active users<extra></extra>")
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Users who rated at least once")
    return style(fig, "Active users per year")


def release_years(df: pd.DataFrame, min_year: int = 1920) -> go.Figure:
    d = df[df["year"] >= min_year]
    fig = go.Figure()
    fig.add_bar(x=d["year"], y=d["num_movies"], name="Movies released", marker_color=BLUE,
                hovertemplate="%{x}: %{y:,} movies<extra></extra>")
    fig.add_scatter(x=d["year"], y=d["avg_rating"], name="Avg rating", yaxis="y2", mode="lines",
                    line=dict(color=AMBER, width=2), hovertemplate="%{x}: avg %{y:.2f}<extra></extra>")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Average rating",
                                  tickfont=dict(color=AMBER), title_font=dict(color=AMBER)))
    fig.update_xaxes(title="Release year")
    fig.update_yaxes(title="Movies")
    return style(fig, "Movies by release year and their average rating")


def user_activity(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="bucket", y="num_users", color_discrete_sequence=[MINT], custom_data=["num_ratings"])
    fig.update_traces(hovertemplate="%{x} ratings: %{y:,} users<br>%{customdata[0]:,} ratings in total<extra></extra>")
    fig.update_xaxes(title="Ratings given per user", type="category")
    fig.update_yaxes(title="Users")
    return style(fig, "How many movies do users rate?")


def user_mean_hist(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="mean_rating_bin", y="num_users", color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="Avg %{x:.2f}: %{y:,} users<extra></extra>")
    fig.update_xaxes(title="User's average rating (0.25 bins)")
    fig.update_yaxes(title="Users")
    return style(fig, "Generous vs harsh raters")


def weekday_hour_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="weekday", columns="hour_utc", values="num_ratings", aggfunc="sum").fillna(0)
    order = df.drop_duplicates("weekday").sort_values("weekday_num")["weekday"].tolist()
    pivot = pivot.reindex(order)
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                               colorscale=[[0, BG2], [0.5, BLUE], [1, AMBER]],
                               hovertemplate="%{y} %{x}:00 UTC<br>%{z:,} ratings<extra></extra>", showscale=False))
    fig.update_xaxes(title="Hour of day (UTC)", dtick=3)
    return style(fig, "When do people rate movies?", height=320)


# ---------------------------------------------------------------- popularity
def top_movies_bar(df: pd.DataFrame, value: str, title: str, n: int = 20, color: str = AMBER) -> go.Figure:
    d = df.head(n).iloc[::-1]
    fig = px.bar(d, x=value, y="title", orientation="h", color_discrete_sequence=[color],
                 custom_data=["num_ratings", "avg_rating"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{customdata[0]:,} ratings · avg %{customdata[1]:.2f}<extra></extra>")
    fig.update_yaxes(title=None, tickfont=dict(color=WHITE, size=11))
    fig.update_xaxes(title=value.replace("_", " ").capitalize())
    return style(fig, title, height=max(380, 24 * len(d) + 80))


def long_tail(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=df["movie_share"] * 100, y=df["ratings_share"] * 100, mode="lines", fill="tozeroy",
                    line=dict(color=AMBER, width=3), fillcolor="rgba(242,174,55,0.15)", name="Actual",
                    hovertemplate="Top %{x:.1f}% of movies → %{y:.1f}% of ratings<extra></extra>")
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=MUTED, dash="dot"), name="Perfect equality",
                    hoverinfo="skip")
    fig.update_xaxes(title="% of movies (most popular first)", range=[0, 100])
    fig.update_yaxes(title="% of all ratings", range=[0, 101])
    return style(fig, "The long tail: a few movies get most of the attention")


def tier_summary(df: pd.DataFrame) -> go.Figure:
    tiers = df["popularity_tier"].tolist()
    movie_pct = 100 * df["num_movies"] / df["num_movies"].sum()
    rating_pct = 100 * df["num_ratings"] / df["num_ratings"].sum()
    fig = go.Figure()
    fig.add_bar(x=tiers, y=movie_pct, name="% of movies", marker_color=BLUE,
                hovertemplate="%{x}: %{y:.1f}% of movies<extra></extra>")
    fig.add_bar(x=tiers, y=rating_pct, name="% of ratings", marker_color=AMBER,
                hovertemplate="%{x}: %{y:.1f}% of ratings<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Percent")
    return style(fig, "Head / Mid / Tail: share of movies vs share of ratings")


def popularity_vs_quality(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="num_ratings", y="avg_rating", color="popularity_tier", color_discrete_map=TIER_COLORS,
                     hover_name="title", log_x=True, opacity=0.6, custom_data=["num_ratings"])
    fig.update_traces(marker=dict(size=5, line=dict(width=0)),
                      hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]:,} ratings · avg %{y:.2f}<extra></extra>")
    fig.update_xaxes(title="Number of ratings (log scale)")
    fig.update_yaxes(title="Average rating")
    return style(fig, "Popularity vs average rating", height=440)


# ---------------------------------------------------------------- genres
def genre_bars(df: pd.DataFrame) -> go.Figure:
    d = df.sort_values("num_ratings")
    fig = px.bar(d, x="num_ratings", y="genre", orientation="h", color="avg_rating",
                 color_continuous_scale=[[0, BLUE], [0.5, MINT], [1, AMBER]], custom_data=["num_movies", "avg_rating"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} ratings · %{customdata[0]:,} movies"
                                    "<br>avg %{customdata[1]:.2f}<extra></extra>")
    fig.update_layout(coloraxis_colorbar=dict(title="Avg", tickfont=dict(color=MUTED)))
    fig.update_yaxes(title=None, tickfont=dict(color=WHITE))
    fig.update_xaxes(title="Ratings received")
    return style(fig, "Ratings per genre (colour = average rating)", height=520)


def genre_quality_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="num_movies", y="avg_rating", size="num_ratings", text="genre", size_max=48,
                     color_discrete_sequence=[BLUE])
    fig.update_traces(textposition="top center", textfont=dict(color=MUTED, size=10),
                      marker=dict(line=dict(width=1, color=WHITE), opacity=0.75),
                      hovertemplate="<b>%{text}</b><br>%{x:,} movies · avg %{y:.2f}<extra></extra>")
    fig.update_xaxes(title="Movies in genre")
    fig.update_yaxes(title="Rating-weighted average")
    return style(fig, "Genre size vs average rating (bubble = ratings)", height=480)


def genre_decade_heatmap(df: pd.DataFrame, min_decade: int = 1930) -> go.Figure:
    d = df[df["decade"] >= min_decade]
    pivot = d.pivot_table(index="genre", columns="decade", values="share", aggfunc="sum").fillna(0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(z=pivot.values * 100, x=[f"{c}s" for c in pivot.columns], y=pivot.index,
                               colorscale=[[0, BG2], [0.4, BLUE], [1, AMBER]],
                               hovertemplate="%{y}, %{x}: %{z:.1f}% of genre labels<extra></extra>",
                               colorbar=dict(title="%", tickfont=dict(color=MUTED))))
    return style(fig, "Genre mix by release decade", height=560)


# ---------------------------------------------------------------- model evaluation
def metric_comparison(df: pd.DataFrame, metrics: list[str], title: str) -> go.Figure:
    long = df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")
    fig = px.bar(long, x="metric", y="value", color="model", barmode="group", color_discrete_map=MODEL_COLORS,
                 text_auto=".3f")
    fig.update_traces(textfont=dict(color=WHITE, size=11), textposition="outside", cliponaxis=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    return style(fig, title)


def rmse_comparison(df: pd.DataFrame) -> go.Figure:
    d = df.sort_values("rmse", ascending=False)
    colors = [BLUE if "ALS" in m else MUTED for m in d["model"]]
    fig = go.Figure(go.Bar(x=d["rmse"], y=d["model"], orientation="h", marker_color=colors,
                           text=[f"{v:.4f}" for v in d["rmse"]], textposition="outside",
                           hovertemplate="%{y}: RMSE %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="RMSE on the test set (lower is better)")
    fig.update_yaxes(title=None, tickfont=dict(color=WHITE))
    return style(fig, "Rating prediction error", height=300)


def tuning_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="rank", columns="reg_param", values="val_rmse")
    fig = go.Figure(go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
                               colorscale=[[0, MINT], [0.5, BLUE], [1, BG2]], texttemplate="%{z:.4f}",
                               hovertemplate="rank %{y}, λ %{x}: RMSE %{z:.4f}<extra></extra>",
                               colorbar=dict(title="RMSE", tickfont=dict(color=MUTED))))
    fig.update_xaxes(title="regParam (λ)", type="category")
    fig.update_yaxes(title="rank (latent factors)", type="category")
    return style(fig, "Hyper-parameter search: validation RMSE", height=320)


def tier_exposure(df: pd.DataFrame) -> go.Figure:
    """Share of recommendations (and of catalogue/ratings) falling in each popularity tier."""
    fig = px.bar(df, x="source", y="share", color="popularity_tier", color_discrete_map=TIER_COLORS,
                 category_orders={"popularity_tier": ["Head", "Mid", "Tail"]}, text_auto=".1%")
    fig.update_traces(textfont=dict(color=BG, size=11))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Share", tickformat=".0%")
    fig.update_xaxes(title=None)
    return style(fig, "Where do recommendations come from? (popularity tiers)", height=420)


def exposure_curve(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for model, grp in df.groupby("model"):
        fig.add_scatter(x=grp["item_share"] * 100, y=grp["exposure_share"] * 100, mode="lines", name=model,
                        line=dict(color=MODEL_COLORS.get(model, WHITE), width=3),
                        hovertemplate=model + ": top %{x:.1f}% of items get %{y:.1f}% of slots<extra></extra>")
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Equal exposure", line=dict(color=MUTED, dash="dot"),
                    hoverinfo="skip")
    fig.update_xaxes(title="% of recommended items (most exposed first)")
    fig.update_yaxes(title="% of recommendation slots")
    return style(fig, "Exposure concentration (Lorenz curve)", height=400)


def score_bars(df: pd.DataFrame, title: str = "Predicted rating") -> go.Figure:
    d = df.iloc[::-1]
    fig = px.bar(d, x="score", y="title", orientation="h", color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>score %{x:.2f}<extra></extra>",
                      text=[f"{v:.2f}" for v in d["score"]], textposition="outside")
    fig.update_yaxes(title=None, tickfont=dict(color=WHITE, size=11))
    fig.update_xaxes(title=title)
    return style(fig, None, height=40 * len(d) + 60)
