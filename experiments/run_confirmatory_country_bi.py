"""
Run a single pre-specified country-held-out fitted-null BI audit.

This is a thin wrapper around run_sir_bi_bootstrap.py for a new validation
country.  It keeps the training pool fixed to the original nine-country
development set and writes artifacts to a separate confirmatory directory.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import pandas as pd


from experiments import run_sir_bi_bootstrap as bi
from experiments.country_bi_scoring import build_country_rows


ROOT = Path(__file__).resolve().parents[1]
BASE_COUNTRIES = ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE", "GBR"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", required=True, help="New validation country ISO code.")
    parser.add_argument("--case", default="recov28_frzgamma", choices=sorted(bi.CASES))
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "cache" / "summaries" / "confirmatory_country_bi",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models/country_cv"),
        help="Directory containing the country-CV model checkpoints.",
    )
    parser.add_argument("--artifact-prefix", default="confirmatory")
    parser.add_argument(
        "--analysis-role",
        default="new-country held-out audit against frozen primary score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val = args.val.upper()
    train_countries = [country for country in BASE_COUNTRIES if country != val]
    updates = {
        **bi.CASES[args.case],
        "VAL_COUNTRY": val,
        "TRAIN_COUNTRIES": train_countries,
        "MODEL_SAVE_PATH": args.model_root / f"neural_sde_sir_val{val}.eqx",
    }

    out_dir = args.out_dir / f"{args.case}_val{val}"
    summary_rows, draws, metadata = bi._run_case(
        case_name=args.case,
        updates=updates,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        out_dir=out_dir,
    )

    summary = pd.DataFrame(summary_rows)
    summary["val_country"] = val
    summary["fold_train_countries"] = ",".join(train_countries)
    draws = draws.copy()
    draws["val_country"] = val
    draws["fold_train_countries"] = ",".join(train_countries)
    metadata["val_country"] = val
    metadata["fold_train_countries"] = train_countries
    metadata["analysis_role"] = args.analysis_role

    global_row, _ = build_country_rows(summary, draws)
    global_score = pd.DataFrame([global_row])

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.artifact_prefix
    summary.to_csv(out_dir / f"{prefix}_bi_summary.csv", index=False)
    draws.to_csv(out_dir / f"{prefix}_bi_draws.csv", index=False)
    global_score.to_csv(out_dir / f"{prefix}_bi_global_score.csv", index=False)
    (out_dir / f"{prefix}_bi_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(summary[["case", "val_country", "metric", "observed", "empirical_pvalue"]].to_string(index=False))
    print()
    print(global_score[["case", "val_country", "observed_smax", "smax_empirical_pvalue", "dominant_metric"]].to_string(index=False))
    print()
    print(f"Saved confirmatory BI artifacts to {out_dir}")


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
