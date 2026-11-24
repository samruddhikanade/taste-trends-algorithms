"""Adds the project root to ``sys.path`` so ``import src...`` works when a script
is started as ``python scripts/<name>.py`` from any folder."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
