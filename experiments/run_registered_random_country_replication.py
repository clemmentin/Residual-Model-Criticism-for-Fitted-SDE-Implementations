"""One-shot randomized new-country replication of the frozen SDE-native score.

The phases are run in order:

1. ``draw`` records the selected countries from the fixed pool and seed.
2. ``evaluate`` reads that draw and runs one 5,000-path audit per country.

This is a prospective replication of an already-developed endpoint, not a
preregistration of the endpoint's development.  All selected countries and all
results are retained regardless of their rejection decisions.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


from experiments import run_sir_sde_native_country_audit as native


PROTOCOL_ID = "random_country_batch2_frozen"
OUT_ROOT = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_random_country_replication"
    / PROTOCOL_ID
)
DRAW_PATH = OUT_ROOT / "registered_country_draw.json"
STAGING_DIR = OUT_ROOT / "_evaluation_staging"
FINAL_DIR = OUT_ROOT / "evaluation"
POSTRUN_PATH = OUT_ROOT / "postrun_summary.json"

# This is the ISO-sorted eligible pool frozen on 2026-08-12 after removing the
# three countries drawn in batch 1.  No member below has been evaluated in any
# recorded country audit before this draw.
ELIGIBLE_POOL = (
    "ALB",
    "BIH",
    "BLR",
    "CZE",
    "EST",
    "GRC",
    "HRV",
    "IRL",
    "LTU",
    "LVA",
    "MDA",
    "MKD",
    "POL",
    "PRT",
    "ROU",
    "RUS",
    "SRB",
    "SVN",
    "UKR",
)

SELECTION_SEED = 20260813
SELECTION_SIZE = 3
NULL_BANK_BASE_SEED = 20260814
N_BOOTSTRAP = 5000
N_ALIGNMENT_ANCHORS = 8
N_ALIGNMENT_MC = 512


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _expected_draw() -> list[str]:
    rng = np.random.default_rng(SELECTION_SEED)
    return [
        str(country)
        for country in rng.choice(
            np.asarray(ELIGIBLE_POOL),
            size=SELECTION_SIZE,
            replace=False,
        ).tolist()
    ]


def draw() -> None:
    if len(ELIGIBLE_POOL) != len(set(ELIGIBLE_POOL)):
        raise ValueError("Eligible pool contains duplicates.")
    if DRAW_PATH.exists() or STAGING_DIR.exists() or FINAL_DIR.exists():
        raise FileExistsError("Draw or evaluation artifacts already exist; refusing a redraw.")
    countries = _expected_draw()
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "registered_draw_revealed_before_evaluation",
        "eligible_pool_iso_sorted": list(ELIGIBLE_POOL),
        "selection_seed": SELECTION_SEED,
        "sampling": "three countries without replacement; generated order retained",
        "countries_in_generated_evaluation_order": countries,
        "no_substitution_allowed": True,
    }
    _write_json_exclusive(DRAW_PATH, payload)
    print(f"Registered draw: {', '.join(countries)}", flush=True)


def _verify_draw() -> tuple[dict[str, object], list[str]]:
    if not DRAW_PATH.exists():
        raise FileNotFoundError("Registered country draw does not exist.")
    payload = json.loads(DRAW_PATH.read_text(encoding="utf-8"))
    expected = _expected_draw()
    saved = payload.get("countries_in_generated_evaluation_order")
    checks = {
        "countries": (saved, expected),
        "eligible_pool": (payload.get("eligible_pool_iso_sorted"), list(ELIGIBLE_POOL)),
        "selection_seed": (payload.get("selection_seed"), SELECTION_SEED),
        "no_substitution_allowed": (payload.get("no_substitution_allowed"), True),
    }
    failures = [
        f"{name}: saved={left!r}, expected={right!r}"
        for name, (left, right) in checks.items()
        if left != right
    ]
    if failures:
        raise ValueError("Registered draw verification failed: " + "; ".join(failures))
    return payload, expected


def evaluate() -> None:
    _, countries = _verify_draw()
    if FINAL_DIR.exists() or POSTRUN_PATH.exists():
        raise FileExistsError("Final evaluation artifacts already exist; refusing to overwrite them.")
    if STAGING_DIR.exists():
        raise FileExistsError("A staging evaluation exists; inspect it before any retry.")

    native.run_countries(
        tuple(countries),
        STAGING_DIR,
        n_bootstrap=N_BOOTSTRAP,
        seed=NULL_BANK_BASE_SEED,
        n_alignment_anchors=N_ALIGNMENT_ANCHORS,
        n_alignment_mc=N_ALIGNMENT_MC,
    )
    STAGING_DIR.replace(FINAL_DIR)

    postrun = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "one-shot_registered_evaluation_complete",
        "countries_in_registered_order": countries,
        "n_bootstrap": N_BOOTSTRAP,
        "pilot_n": N_BOOTSTRAP // 2,
        "evaluation_n": N_BOOTSTRAP // 2,
        "null_bank_base_seed": NULL_BANK_BASE_SEED,
        "no_interim_run_performed_by_this_protocol": True,
        "result_directory": str(FINAL_DIR.relative_to(ROOT)).replace("\\", "/"),
    }
    _write_json_exclusive(POSTRUN_PATH, postrun)
    print(f"Completed registered evaluation: {FINAL_DIR}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("draw", "evaluate"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "draw":
        draw()
    else:
        evaluate()


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
