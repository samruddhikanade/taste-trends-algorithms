"""Configuration loading.

Reads ``config/config.yaml`` (falls back to ``config/config.example.yaml``),
merges the selected mode (``dev`` or ``full``) into a flat, easy-to-use
dictionary and resolves every path relative to the project root so the code
works from any working directory on Windows, Linux or macOS.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
VALID_MODES = ("dev", "full")


def _config_file() -> Path:
    override = os.environ.get("TTA_CONFIG")  # used by the tests to point at a temporary config
    if override:
        return Path(override)
    user_cfg = CONFIG_DIR / "config.yaml"
    return user_cfg if user_cfg.exists() else CONFIG_DIR / "config.example.yaml"


def load_config(mode: str = "dev", config_path: str | Path | None = None) -> dict[str, Any]:
    """Return the merged configuration for ``mode``.

    The returned dict contains the top-level sections plus:
      * ``mode``      - the selected mode name
      * ``run``       - the mode-specific settings
      * ``dataset``   - the dataset entry used by this mode
      * ``paths``     - absolute ``Path`` objects; results are separated per mode
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    path = Path(config_path) if config_path else _config_file()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = copy.deepcopy(raw)
    run = cfg["modes"][mode]
    cfg["mode"] = mode
    cfg["run"] = run
    cfg["dataset"] = cfg["datasets"][run["dataset"]]

    paths = {key: (PROJECT_ROOT / value).resolve() for key, value in cfg["paths"].items()}
    # Keep dev and full outputs apart so a quick dev run never overwrites full results.
    paths["results_dir"] = paths["results_dir"] / mode
    paths["figures_dir"] = paths["figures_dir"] / mode
    paths["processed_dir"] = paths["processed_dir"] / mode
    paths["models_dir"] = paths["models_dir"] / mode
    paths["dataset_dir"] = paths["raw_dir"] / cfg["dataset"]["folder"]
    cfg["paths"] = paths
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create every output directory used by the pipeline."""
    for key in ("processed_dir", "models_dir", "results_dir", "figures_dir", "spark_tmp_dir", "raw_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
