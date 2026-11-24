"""Dark cinematic look-and-feel: CSS, KPI cards, headers and chart helpers."""
from __future__ import annotations

import html

import streamlit as st

from src.visuals import AMBER, BG, BG2, BLUE, MINT, MUTED, WHITE

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}
.stApp {{
  background:
    radial-gradient(1100px 500px at 85% -10%, rgba(87,139,238,0.16), transparent 60%),
    radial-gradient(800px 400px at -10% 110%, rgba(242,174,55,0.08), transparent 60%),
    {BG};
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0D1427 0%, {BG2} 100%);
  border-right: 1px solid rgba(239,242,248,0.06); }}
section[data-testid="stSidebar"] .stRadio label {{ padding: 6px 4px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {WHITE}; }}
a {{ color: {BLUE} !important; }}
hr {{ border-color: rgba(239,242,248,0.08); }}

.hero {{ padding: 34px 38px; border-radius: 22px; margin-bottom: 26px;
  background: linear-gradient(120deg, rgba(87,139,238,0.22) 0%, rgba(21,29,49,0.92) 45%, rgba(242,174,55,0.16) 100%);
  border: 1px solid rgba(239,242,248,0.08); box-shadow: 0 30px 60px -30px rgba(0,0,0,0.7); }}
.hero .eyebrow {{ color: {AMBER}; font-weight: 700; letter-spacing: .22em; font-size: .74rem; text-transform: uppercase; }}
.hero h1 {{ font-size: 2.7rem; font-weight: 800; margin: .35rem 0 .2rem 0; line-height: 1.08; }}
.hero .sub {{ color: #C9D1E4; font-size: 1.08rem; margin-bottom: 1rem; }}
.hero .meta {{ color: {MUTED}; font-size: .86rem; }}
.hero .meta b {{ color: {WHITE}; font-weight: 600; }}

.page-head {{ margin: 0 0 18px 0; }}
.page-head .kicker {{ color: {AMBER}; font-size: .72rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase; }}
.page-head h2 {{ font-size: 1.95rem; font-weight: 800; margin: .15rem 0 .25rem 0; }}
.page-head p {{ color: {MUTED}; margin: 0; max-width: 820px; }}

.kpi {{ background: linear-gradient(160deg, {BG2} 0%, #10172A 100%); border: 1px solid rgba(239,242,248,0.07);
  border-radius: 16px; padding: 16px 18px 14px 18px; height: 100%; position: relative; overflow: hidden; }}
.kpi:before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent); }}
.kpi .label {{ color: {MUTED}; font-size: .74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }}
.kpi .value {{ color: {WHITE}; font-size: 1.75rem; font-weight: 800; margin-top: 4px; letter-spacing: -0.02em; }}
.kpi .note {{ color: {MUTED}; font-size: .78rem; margin-top: 2px; }}

.card {{ background: {BG2}; border: 1px solid rgba(239,242,248,0.07); border-radius: 16px; padding: 18px 20px; margin-bottom: 12px; }}
.card h4 {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
.card p {{ color: #C3CBDD; margin: 0; font-size: .92rem; line-height: 1.5; }}

.movie {{ display: flex; gap: 14px; align-items: flex-start; background: {BG2}; border: 1px solid rgba(239,242,248,0.07);
  border-radius: 14px; padding: 12px 16px; margin-bottom: 8px; }}
.movie .rank {{ font-size: 1.4rem; font-weight: 800; color: {AMBER}; min-width: 30px; text-align: right; }}
.movie .body {{ flex: 1; }}
.movie .title {{ font-weight: 700; color: {WHITE}; font-size: 1rem; }}
.movie .genres {{ color: {MUTED}; font-size: .82rem; margin-top: 2px; }}
.movie .score {{ text-align: right; min-width: 92px; }}
.movie .score .big {{ font-size: 1.3rem; font-weight: 800; color: {BLUE}; }}
.movie .score .small {{ color: {MUTED}; font-size: .72rem; }}
.pill {{ display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .7rem; font-weight: 700;
  margin-right: 6px; border: 1px solid currentColor; }}
.pill.Head {{ color: {AMBER}; }} .pill.Mid {{ color: {BLUE}; }} .pill.Tail {{ color: {MINT}; }} .pill.Unrated {{ color: {MUTED}; }}

.flow {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 6px 0 4px 0; }}
.flow .step {{ background: {BG2}; border: 1px solid rgba(87,139,238,0.35); border-radius: 12px; padding: 10px 14px;
  font-size: .86rem; color: {WHITE}; }}
.flow .step span {{ display: block; color: {MUTED}; font-size: .72rem; }}
.flow .arrow {{ color: {AMBER}; font-weight: 800; }}
.empty {{ border: 1px dashed rgba(239,242,248,0.2); border-radius: 16px; padding: 26px; color: #C3CBDD; background: rgba(21,29,49,0.5); }}
.empty code {{ color: {AMBER}; }}
div[data-testid="stMetricValue"] {{ font-weight: 800; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ background: {BG2}; border-radius: 10px 10px 0 0; padding: 8px 16px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, text: str = "") -> None:
    st.markdown(f'<div class="page-head"><div class="kicker">{html.escape(kicker)}</div>'
                f'<h2>{html.escape(title)}</h2><p>{text}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", accent: str = BLUE) -> str:
    return (f'<div class="kpi" style="--accent:{accent}"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>')


def kpi_row(items: list[tuple[str, str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, note, accent) in zip(cols, items):
        col.markdown(kpi(label, value, note, accent), unsafe_allow_html=True)
    st.write("")


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><h4>{html.escape(title)}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def empty_state(what: str, command: str) -> None:
    st.markdown(f'<div class="empty"><b>{html.escape(what)}</b><br><br>Run this in PowerShell from the project '
                f'folder, then refresh the page:<br><br><code>{html.escape(command)}</code></div>',
                unsafe_allow_html=True)


def plot(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def fmt_int(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "-"


def fmt_compact(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.1f}{unit}".replace(".0" + unit, unit)
    return f"{v:,.0f}"


def movie_row(rank: int, title: str, genres: str, score: float | None, score_label: str, tier: str,
              extra: str = "") -> str:
    score_html = (f'<div class="score"><div class="big">{score:.2f}</div><div class="small">{html.escape(score_label)}'
                  f'</div></div>') if score is not None else ""
    return (f'<div class="movie"><div class="rank">{rank}</div><div class="body"><div class="title">{html.escape(title)}'
            f'</div><div class="genres"><span class="pill {html.escape(tier)}">{html.escape(tier)}</span>'
            f'{html.escape((genres or "").replace("|", " · "))}</div>'
            f'<div class="genres">{extra}</div></div>{score_html}</div>')


__all__ = ["inject_css", "page_header", "kpi", "kpi_row", "card", "empty_state", "plot", "fmt_int", "fmt_compact",
           "movie_row", "AMBER", "BLUE", "MINT", "MUTED", "WHITE", "BG", "BG2"]
