"""Run the second fixed-calendar SIR holdout.

This reuses the first holdout's window, preprocessing, fitted model, score,
bank split, and rank calculation.  It changes only the prior-target exclusion
list and the two recorded seeds.

Run from the release root:

    python experiments/run_second_fixed_calendar_holdout.py --phase freeze
    python experiments/run_second_fixed_calendar_holdout.py --phase evaluate
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse


from experiments import run_fixed_calendar_holdout as first


SELECTION_SEED = 2026090302
NULL_BANK_BASE_SEED = 2026090302001
NULL_BANK_PURPOSE = "second-fixed-calendar-holdout"


def _configure_second_holdout() -> None:
    """Point the shared implementation at the recorded second holdout."""
    first.SELECTION_SEED = SELECTION_SEED
    first.NULL_BANK_BASE_SEED = NULL_BANK_BASE_SEED
    first.NULL_BANK_PURPOSE = NULL_BANK_PURPOSE
    first.EXCLUDED_TARGETS = tuple(sorted(set(first.EXCLUDED_TARGETS) | {"CZE"}))
    first.OUT_ROOT = (
        first.ROOT
        / "cache"
        / "summaries"
        / "fixed_calendar_holdout"
        / "second_holdout"
    )
    first.FREEZE_PATH = (
        first.OUT_ROOT / "second_holdout_freeze.json"
    )
    first.RESULTS_PATH = first.OUT_ROOT / "second_holdout_results.csv"
    first.REFERENCE_DRAWS_PATH = (
        first.OUT_ROOT / "second_holdout_reference_draws.csv"
    )
    first.METADATA_PATH = first.OUT_ROOT / "second_holdout_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "evaluate"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_second_holdout()
    if args.phase == "freeze":
        first.freeze()
    else:
        first.evaluate()


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
