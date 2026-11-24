"""Download and extract the MovieLens dataset.

    python scripts/download_data.py --mode full    # MovieLens 25M (~250 MB zip, ~1.1 GB extracted)
    python scripts/download_data.py --mode dev     # MovieLens latest-small (~1 MB)

The raw data goes to ``data/raw/<folder>/`` which is git-ignored and is never
included in the project ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import load_config

REQUIRED_FILES = ("ratings.csv", "movies.csv", "tags.csv", "links.csv")


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = done * 100 / total_size
    bar = "#" * int(pct // 2.5)
    sys.stdout.write(f"\r  [{bar:<40}] {pct:5.1f}%  {done / 1e6:7.1f} / {total_size / 1e6:.1f} MB")
    sys.stdout.flush()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_expected_md5(md5_url: str) -> str | None:
    if not md5_url:
        return None
    try:
        with urllib.request.urlopen(md5_url, timeout=30) as resp:
            text = resp.read().decode("utf-8").strip()
        return text.split()[0].lower()
    except Exception as exc:  # network problems should not block the download
        print(f"  ! Could not fetch checksum ({exc}); skipping verification.")
        return None


def dataset_ready(dataset_dir: Path) -> bool:
    return all((dataset_dir / name).exists() for name in REQUIRED_FILES)


def download(mode: str, force: bool = False, keep_zip: bool = False) -> Path:
    cfg = load_config(mode)
    raw_dir: Path = cfg["paths"]["raw_dir"]
    dataset_dir: Path = cfg["paths"]["dataset_dir"]
    url: str = cfg["dataset"]["url"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset_ready(dataset_dir) and not force:
        print(f"Dataset already present in {dataset_dir} (use --force to re-download).")
        return dataset_dir

    zip_path = raw_dir / Path(url).name
    print(f"Downloading {url}\n  -> {zip_path}")
    urllib.request.urlretrieve(url, zip_path, _progress)
    print()

    expected = fetch_expected_md5(cfg["dataset"].get("md5_url", ""))
    if expected:
        actual = md5sum(zip_path)
        if actual != expected:
            raise SystemExit(f"Checksum mismatch: expected {expected}, got {actual}. Delete {zip_path} and retry.")
        print(f"  Checksum OK ({actual})")

    print(f"Extracting to {raw_dir} ...")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    missing = [name for name in REQUIRED_FILES if not (dataset_dir / name).exists()]
    if missing:
        raise SystemExit(f"Extraction finished but these files are missing: {missing}")
    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    for name in REQUIRED_FILES:
        size_mb = (dataset_dir / name).stat().st_size / 1e6
        print(f"  {name:<12} {size_mb:9.1f} MB")
    print("Done.")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("dev", "full"), default="full")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip file")
    args = parser.parse_args()
    download(args.mode, force=args.force, keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
