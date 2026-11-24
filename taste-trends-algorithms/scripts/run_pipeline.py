"""Run the complete pipeline: ingestion -> preprocessing -> Spark SQL EDA ->
ALS tuning/training -> recommendations -> evaluation -> RESULTS.md.

    python scripts/run_pipeline.py --mode dev                     # small dataset, a few minutes
    python scripts/run_pipeline.py --mode full                    # MovieLens 25M
    python scripts/run_pipeline.py --mode full --sample 0.2       # 20% of users from 25M
    python scripts/run_pipeline.py --mode dev --skip-eda          # model only
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("dev", "full"), default="dev")
    parser.add_argument("--sample", type=float, default=None,
                        help="fraction of users to keep (0-1); overrides the config")
    parser.add_argument("--skip-eda", action="store_true", help="skip modules C/D")
    parser.add_argument("--skip-model", action="store_true", help="skip modules E/F")
    args = parser.parse_args()
    stages = tuple(s for s, skip in (("eda", args.skip_eda), ("model", args.skip_model)) if not skip)
    run(args.mode, stages=stages, sample_fraction=args.sample)


if __name__ == "__main__":
    main()
