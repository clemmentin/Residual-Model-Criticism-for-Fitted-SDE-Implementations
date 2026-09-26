"""Post-hoc Monte Carlo-resolution check for the corrected SIR audit.

The frozen 250-path pilot bank from the existing geometry-aware corrected
audit is retained exactly. A fresh, independent evaluation bank is generated
from the same corrected checkpoint and tangent-I path-to-score map. The result
refines the Monte Carlo grid only; it is not a new registered confirmation and
does not replace the registered 250-reference primary result.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


from experiments import run_sir_bi_bootstrap as bi, run_sir_tangent_map_comparison as tangent
from experiments.country_bi_scoring import build_country_rows_from_banks


COUNTRIES = ("FIN", "NOR", "SWE")
PILOT_ROOT = (
    ROOT / "cache" / "summaries" / "corrected_country_bi"
    / "tangent_geometry"
)
MODEL_ROOT = (
    ROOT / "models" / "country_cv"
    / "corrected_fixednorm_matchedsolver"
)
DEFAULT_OUT_DIR = (
    ROOT / "cache" / "summaries" / "corrected_country_bi"
    / "tangent_geometry_precision_m5000"
)
EVALUATION_SEEDS = {"FIN": 20260720, "NOR": 20260721, "SWE": 20260722}


def _load_frozen_pilot(country: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    country_dir = PILOT_ROOT / f"recov28_frzgamma_val{country}"
    draws_path = country_dir / "geometry_bi_draws.csv"
    summary_path = country_dir / "geometry_bi_summary.csv"
    metadata_path = country_dir / "geometry_bi_metadata.json"
    draws = pd.read_csv(draws_path)
    summary = pd.read_csv(summary_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["n_bootstrap"]) != 500 or int(metadata["seed"]) != 20260706:
        raise ValueError(f"{country}: frozen geometry bank is not the expected 500-path run")
    ids = pd.to_numeric(draws["bootstrap_id"], errors="raise").astype(int)
    pilot = draws.loc[ids.mod(2).eq(0)].copy()
    if len(pilot) != 250:
        raise ValueError(f"{country}: expected 250 frozen pilot paths; found {len(pilot)}")
    return pilot, summary, draws_path


def _run_country(
    country: str,
    evaluation_paths: int,
    evaluation_seed: int,
    out_root: Path,
) -> dict[str, object]:
    pilot, frozen_summary, pilot_source = _load_frozen_pilot(country)
    cfg = tangent._config(country, MODEL_ROOT)
    bi._validate_supported_config(cfg)
    data = tangent.load_cached_data(cfg)
    model, model_path = bi._load_model(cfg, data)
    start_pos = bi._start_position(data, cfg)
    start_date = str(data.val_features_df.index[start_pos].date())

    _, observed_tangent, _, _ = tangent._observed_residuals(
        model, data, cfg, start_pos
    )
    observed_metrics = bi.compute_brownian_inversion_metrics_from_residuals(
        observed_tangent
    )
    for metric in frozen_summary["metric"].astype(str):
        observed = float(observed_metrics[metric])
        recorded = float(
            frozen_summary.loc[frozen_summary["metric"].eq(metric), "observed"].iloc[0]
        )
        if not np.isclose(float(observed), recorded, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"{country}: observed {metric} changed from {recorded} to {observed}"
            )

    _, tangent_raw, cov01, cov11, clip_fraction = tangent._simulated_residuals(
        model,
        data,
        cfg,
        start_pos,
        evaluation_paths,
        evaluation_seed,
    )
    evaluation = tangent._metric_draws(
        "recov28_frzgamma_tangent_geometry_precision",
        country,
        str(frozen_summary["experiment"].iloc[0]),
        start_date,
        tangent_raw,
        clip_fraction,
    )
    global_row, profile = build_country_rows_from_banks(
        frozen_summary, pilot, evaluation
    )
    pvalue = float(global_row["smax_empirical_pvalue"])
    exceedances = int(round(pvalue * (evaluation_paths + 1) - 1))

    out_dir = out_root / f"recov28_frzgamma_val{country}"
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(out_dir / "precision_evaluation_draws.csv", index=False)
    pd.DataFrame([global_row]).to_csv(
        out_dir / "precision_global_score.csv", index=False
    )
    pd.DataFrame([profile]).to_csv(
        out_dir / "precision_max_adjusted_profile.csv", index=False
    )
    metadata = {
        "role": (
            "post-hoc Monte Carlo-resolution check for the corrected "
            "geometry-aware audit; not registered confirmation"
        ),
        "country": country,
        "pilot_paths": 250,
        "pilot_rule": "exact even bootstrap IDs from the frozen 20260717 bank",
        "pilot_source": str(pilot_source.relative_to(ROOT)).replace("\\", "/"),
        "evaluation_paths": evaluation_paths,
        "evaluation_seed": evaluation_seed,
        "rank_denominator": evaluation_paths + 1,
        "minimum_rank_pvalue": 1.0 / (evaluation_paths + 1),
        "reference_exceedance_count": exceedances,
        "global_rank_pvalue": pvalue,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "mean_tangent_cov_SI": float(np.mean(cov01)),
        "mean_tangent_cov_II": float(np.mean(cov11)),
        "mean_clip_fraction": float(np.mean(clip_fraction)),
    }
    (out_dir / "precision_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "country": country,
        "pilot_paths": 250,
        "evaluation_paths": evaluation_paths,
        "evaluation_seed": evaluation_seed,
        "observed_smax": float(global_row["observed_smax"]),
        "reference_exceedance_count": exceedances,
        "global_rank_pvalue": pvalue,
        "minimum_rank_pvalue": 1.0 / (evaluation_paths + 1),
        "dominant_metric": str(global_row["dominant_metric"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", nargs="+", default=list(COUNTRIES), choices=COUNTRIES)
    parser.add_argument("--evaluation-paths", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluation_paths < 1:
        raise ValueError("--evaluation-paths must be positive")
    rows = []
    for country in [str(value).upper() for value in args.val]:
        print(f"=== 5000-reference precision check: {country} ===", flush=True)
        row = _run_country(
            country,
            evaluation_paths=args.evaluation_paths,
            evaluation_seed=EVALUATION_SEEDS[country],
            out_root=args.out_dir,
        )
        rows.append(row)
        print(pd.Series(row).to_string(), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "precision_global_scores.csv", index=False)
    print()
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
