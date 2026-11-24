"""Capture a PNG screenshot of every dashboard section (optional).

1. Start the dashboard in one PowerShell window:   streamlit run app.py
2. In a second window:
       pip install playwright
       python -m playwright install chromium      (or skip this and add --chrome)
       python scripts/capture_screenshots.py --mode full

Screenshots go to reports/figures/<mode>/screenshots/ and are inserted into the
README's "Dashboard screenshots" section. You can also take them by hand
(Win + Shift + S) and save them in that folder with names like 01_overview.png.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import PROJECT_ROOT, load_config
from src.report import update_readme

PAGES = ["Project Overview", "Dataset Analytics", "Movie Popularity Analysis", "Genre Analysis",
         "Personalised Recommendations", "Model Performance", "ALS vs Popularity-Based"]


def wait_until_idle(page, timeout: float = 60.0) -> None:
    """Wait until Streamlit has finished running the script (status widget gone)."""
    deadline = time.time() + timeout
    time.sleep(1.0)
    while time.time() < deadline:
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(2.0)  # let Plotly finish drawing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--mode", choices=("dev", "full"), default="full")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--chrome", action="store_true",
                        help="use your installed Google Chrome instead of Playwright's Chromium")
    args = parser.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install playwright && python -m playwright install chromium")

    out = Path(load_config(args.mode)["paths"]["figures_dir"]) / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome") if args.chrome else p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": 1000}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        wait_until_idle(page)
        label = "MovieLens 25M" if args.mode == "full" else "Dev (small)"
        mode_radio = page.locator("section[data-testid='stSidebar']").get_by_text(label, exact=True)
        if mode_radio.count():
            mode_radio.first.click()
            wait_until_idle(page)
        for i, name in enumerate(PAGES, start=1):
            page.locator("section[data-testid='stSidebar']").get_by_text(name, exact=True).first.click()
            wait_until_idle(page)
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            path = out / f"{i:02d}_{slug}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"saved {path}")
        browser.close()
    if update_readme(PROJECT_ROOT / "README.md", load_config(args.mode)["paths"]["results_dir"], args.mode, out):
        print("README.md screenshot gallery updated.")


if __name__ == "__main__":
    main()
