"""Create Samruddhi_Kanade_BDA_Project.zip (default: on your Desktop) and verify it.

    python scripts/package_zip.py
    python scripts/package_zip.py --out D:\\submissions

Excluded: virtual environments, caches, the raw MovieLens data, processed data,
trained-model binaries, Spark temp files, local config/secrets and .git.
Included: all code, docs, tests, config example, and the generated
reports/results + reports/figures (your real results).
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = _bootstrap.ROOT
ZIP_NAME = "Samruddhi_Kanade_BDA_Project.zip"
TOP = "taste-trends-algorithms"
EXCLUDE_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
                "spark-tmp", "spark-warehouse", "metastore_db", ".idea", ".vscode"}
EXCLUDE_PATTERNS = ["*.pyc", "*.zip", "*.crc", "derby.log", "config/config.yaml", ".env", "*.pem", "*.key",
                    "data/raw/*", "data/processed/*", "models/*"]
KEEP = {"data/raw/.gitkeep", "data/processed/.gitkeep", "models/.gitkeep"}
REQUIRED = ["README.md", "requirements.txt", "app.py", ".gitignore", "config/config.example.yaml",
            "src/ingestion.py", "src/preprocessing.py", "src/eda.py", "src/analytics.py", "src/recommendation.py",
            "src/evaluation.py", "scripts/download_data.py", "scripts/train_model.py", "scripts/run_pipeline.py",
            "docs/architecture.md", "docs/methodology.md", "docs/research_paper_notes.md", "docs/viva_preparation.md"]


def default_desktop() -> Path:
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    for candidate in (home / "Desktop", home / "OneDrive" / "Desktop"):
        if candidate.exists():
            return candidate
    return home


def included(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        return False
    if rel in KEEP:
        return True
    return not any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(parts[-1], pat) for pat in EXCLUDE_PATTERNS)


def collect_files(root: Path = ROOT) -> list[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            rel = Path(dirpath, name).relative_to(root).as_posix()
            if included(rel):
                files.append(rel)
    return sorted(files)


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ZIP_NAME
    files = collect_files()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            zf.write(ROOT / rel, f"{TOP}/{rel}")
    return target


def verify(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"Corrupt member in zip: {bad}")
        names = set(zf.namelist())
    missing = [r for r in REQUIRED if f"{TOP}/{r}" not in names]
    leaked = [n for n in names if "/data/raw/" in n and not n.endswith(".gitkeep")]
    if missing or leaked:
        raise SystemExit(f"Verification failed. missing={missing} raw-data-leaked={leaked[:5]}")
    size = path.stat().st_size / 1e6
    print(f"Verified {path} - {len(names)} files, {size:.2f} MB, all required files present, no raw data.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: Desktop)")
    args = parser.parse_args()
    target = build(args.out or default_desktop())
    verify(target)


if __name__ == "__main__":
    main()
