"""Train and evaluate the ALS model only (modules A, B, E, F - skips the EDA).

    python scripts/train_model.py --mode dev
    python scripts/train_model.py --mode full
    python scripts/train_model.py --mode full --sample 0.25
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("dev", "full"), default="dev")
    parser.add_argument("--sample", type=float, default=None, help="fraction of users to keep (0-1)")
    args = parser.parse_args()
    run(args.mode, stages=("model",), sample_fraction=args.sample)


if __name__ == "__main__":
    main()
