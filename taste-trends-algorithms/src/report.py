"""Writes ``reports/results/<mode>/RESULTS.md`` from the JSON/CSV files that the
pipeline produced. Nothing is typed in by hand: if a file is missing, the
section says so."""
from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import read_json


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, numbers.Integral) and not isinstance(v, bool):
        return f"{int(v):,}"
    if isinstance(v, numbers.Real):
        return f"{float(v):,.{digits}f}"
    return str(v)


def _table(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.to_dict(orient="records"):
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def build_results_markdown(results_dir: str | Path, mode: str) -> str:
    rd = Path(results_dir)
    summary = read_json(rd / "eda" / "summary.json")
    metrics = read_json(rd / "metrics.json")
    run_info = read_json(rd / "run_info.json")
    meta = read_json(rd / "als_metadata.json")
    out = [f"# Results - `{mode}` mode", "",
           "_Generated automatically by `src/report.py` from the pipeline's output files._", ""]

    if run_info:
        out += [f"Spark version: {run_info.get('spark_version')} · sample fraction: {run_info.get('sample_fraction')}", "",
                "| Stage | Seconds |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in run_info.get("timings_seconds", {}).items()] + [""]

    out += ["## Dataset", ""]
    if summary:
        rows = [("Ratings", summary["total_ratings"]), ("Users", summary["total_users"]),
                ("Movies in catalogue", summary["total_movies"]), ("Movies with ≥ 1 rating", summary["rated_movies"]),
                ("Tags", summary["total_tags"]), ("Average rating", summary["avg_rating"]),
                ("Matrix sparsity (%)", summary["matrix_sparsity_pct"]),
                ("First rating", summary["first_rating"]), ("Last rating", summary["last_rating"])]
        out += ["| Measure | Value |", "|---|---|"] + [f"| {a} | {_fmt(b)} |" for a, b in rows] + [""]
    else:
        out += ["EDA has not been run yet.", ""]

    out += ["## Model", ""]
    if meta:
        out += [f"ALS rank = **{meta['rank']}**, regParam = **{meta['reg_param']}**, maxIter = {meta['max_iter']}; "
                f"trained in {meta['train_seconds']} s; factors for {meta['users_with_factors']:,} users and "
                f"{meta['movies_with_factors']:,} movies.", ""]
    tuning = rd / "tuning_results.csv"
    if tuning.exists():
        out += ["### Hyper-parameter search (validation RMSE)", "", _table(pd.read_csv(tuning)), ""]

    if metrics:
        out += ["## Rating prediction (RMSE, lower is better)", "",
                f"Scored {metrics['test_rows_scored']:,} of {metrics['test_rows']:,} test ratings "
                f"({metrics['test_rows_cold_start_dropped']:,} dropped because the user or movie never appeared "
                "in training - cold start).", ""]
        r = metrics["rmse"]
        out += ["| Model | RMSE |", "|---|---|", f"| ALS | {_fmt(r['als'])} |",
                f"| Baseline μ + b_u + b_i | {_fmt(r['bias_baseline'])} |",
                f"| Baseline global mean | {_fmt(r['global_mean'])} |", ""]
        comp = rd / "model_comparison.csv"
        if comp.exists():
            df = pd.read_csv(comp)[["model", "precision_at_10", "recall_at_10", "ndcg_at_10", "hit_rate_at_10",
                                    "coverage", "arp", "head_share", "tail_share", "gini_exposure"]]
            out += [f"## Top-10 recommendation quality ({metrics['eval_users']:,} test users, "
                    f"relevant = rated ≥ {metrics['relevance_threshold']})", "", _table(df), "",
                    f"Average popularity of movies in users' own histories (ARP of profiles): "
                    f"{_fmt(metrics['user_profile_arp'], 1)} ratings. Popularity lift (ARP of recommendations ÷ ARP of "
                    f"profiles): " + ", ".join(f"{m} {_fmt(v, 2)}" for m, v in metrics["popularity_lift"].items()), ""]
        if "all_users" in metrics:
            a = metrics["all_users"]
            out += ["### ALS lists for every user", "",
                    f"{a['users']:,} users · catalogue coverage {_fmt(a['coverage'])} · Head share "
                    f"{a['head_share']:.1%} · Tail share {a['tail_share']:.1%} · Gini {_fmt(a['gini_exposure'])}", ""]
    else:
        out += ["Model evaluation has not been run yet.", ""]
    return "\n".join(out)


def write_results_markdown(cfg: dict[str, Any]) -> Path:
    rd = Path(cfg["paths"]["results_dir"])
    path = rd / "RESULTS.md"
    path.write_text(build_results_markdown(rd, cfg["mode"]), encoding="utf-8")
    return path


RESULTS_START, RESULTS_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
SHOTS_START, SHOTS_END = "<!-- SCREENSHOTS:START -->", "<!-- SCREENSHOTS:END -->"


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        return text
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}\n{end}{tail}"


def update_readme(readme: str | Path, results_dir: str | Path, mode: str,
                  screenshots_dir: str | Path | None = None) -> bool:
    """Copy the generated results (and any captured screenshots) into README.md
    between the marker comments. Returns True if the README changed."""
    readme = Path(readme)
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    new = text
    if (Path(results_dir) / "metrics.json").exists() or (Path(results_dir) / "eda" / "summary.json").exists():
        body = build_results_markdown(results_dir, mode).split("\n", 3)[-1]  # drop the H1 title
        body = body.replace("\n## ", "\n#### ").replace("\n### ", "\n##### ")
        if body.startswith("## "):
            body = "#### " + body[3:]
        new = _replace_block(new, RESULTS_START, RESULTS_END,
                             f"_Auto-inserted from `reports/results/{mode}/` (mode: **{mode}**)._\n\n{body.strip()}")
    if screenshots_dir and Path(screenshots_dir).exists():
        shots = sorted(Path(screenshots_dir).glob("*.png"))
        if shots:
            rel = [p.relative_to(readme.parent).as_posix() for p in shots]
            gallery = "\n\n".join(f"**{p.stem[3:].replace('_', ' ').title()}**\n\n![{p.stem}]({r})"
                                  for p, r in zip(shots, rel))
            new = _replace_block(new, SHOTS_START, SHOTS_END, gallery)
    if new != text:
        readme.write_text(new, encoding="utf-8")
        return True
    return False
