"""Small file helpers shared by the pipeline and the dashboard.

Spark writes through the Hadoop file-system layer, which on Windows needs the
extra ``winutils.exe``/``hadoop.dll`` binaries. To keep the project working on
a plain Windows laptop, Spark does all of the heavy computation and only the
*small, aggregated* results (thousands to a few million rows) are brought to
the driver and saved with pandas. The full ratings table is never converted to
pandas.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_table(df: pd.DataFrame, path_stem: str | Path) -> Path:
    """Save ``df`` as Parquet (preferred) or gzip CSV and return the written path.

    ``path_stem`` is given without an extension.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if _has_pyarrow():
        out = stem.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        old = stem.with_suffix(".csv.gz")
    else:
        out = Path(str(stem) + ".csv.gz")
        df.to_csv(out, index=False, compression="gzip")
        old = stem.with_suffix(".parquet")
    if old.exists():  # avoid a stale copy in the other format being read later
        old.unlink()
    return out


def read_table(path_stem: str | Path) -> pd.DataFrame | None:
    """Read a table written by :func:`write_table`. Returns ``None`` if missing."""
    stem = Path(path_stem)
    parquet = stem.with_suffix(".parquet")
    csv_gz = Path(str(stem) + ".csv.gz")
    if parquet.exists() and _has_pyarrow():
        return pd.read_parquet(parquet)
    if csv_gz.exists():
        return pd.read_csv(csv_gz)
    return None


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    return path


def read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
