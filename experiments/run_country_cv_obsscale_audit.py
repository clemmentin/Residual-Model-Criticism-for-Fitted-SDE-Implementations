"""
Cross-country observation-scale Brownian-inversion audit for SIR LOO models.

This script consumes fold artifacts produced by experiments/run_country_cv.py and
runs two cross-country checks:

1. probability-coordinate geometry on every available LOO fold;
2. a country-LOO fitted-null bootstrap on selected countries/configs.

Run from the sde directory, for example:
    python experiments/run_country_cv_obsscale_audit.py geometry
    python experiments/run_country_cv_obsscale_audit.py bootstrap-smoke --countries GBR DEU ESP --n-bootstrap 50
    python experiments/run_country_cv_obsscale_audit.py bootstrap-smoke --n-bootstrap 500 --write-full-aliases
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import config as config_module
from brownian_inversion import (  # noqa: E402
    _evaluate_bi_residuals,
    compute_brownian_inversion_metrics_from_residuals,
)

from experiments._bi_audit_common import load_cached_data as _load_data
from experiments._sir_model_loader import load_first_model as _load_model
from experiments.run_sir_bi_bootstrap import (
    METRIC_TAILS,
    _empirical_pvalue,
    _simulate_null_residuals,
    _start_position,
)
from experiments.run_sir_discrepancy_diagnostic import _metrics as _discrepancy_metrics


ALL_COUNTRIES = ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE", "GBR"]
SUMMARY_DIR = ROOT / "cache" / "summaries" / "country_cv"

CASE_UPDATES = {
    "baseline": {"SIR_RECOVERY_DAYS": 14.0, "FREEZE_GAMMA": False},
    "baseline_frzgamma": {"SIR_RECOVERY_DAYS": 14.0, "FREEZE_GAMMA": True},
    "recov28": {"SIR_RECOVERY_DAYS": 28.0, "FREEZE_GAMMA": False},
    "recov28_frzgamma": {"SIR_RECOVERY_DAYS": 28.0, "FREEZE_GAMMA": True},
}

GEOMETRY_OUT = SUMMARY_DIR / "country_cv_obsscale_geometry.csv"
GEOMETRY_AGG_OUT = SUMMARY_DIR / "country_cv_obsscale_geometry_aggregate.csv"
BOOTSTRAP_SUMMARY_OUT = SUMMARY_DIR / "country_cv_bi_bootstrap_smoke_summary.csv"
BOOTSTRAP_DRAWS_OUT = SUMMARY_DIR / "country_cv_bi_bootstrap_smoke_draws.csv"
BOOTSTRAP_FULL_SUMMARY_OUT = SUMMARY_DIR / "country_cv_bi_bootstrap_full_summary.csv"
BOOTSTRAP_FULL_DRAWS_OUT = SUMMARY_DIR / "country_cv_bi_bootstrap_full_draws.csv"


def _fold_config(case: str, val_country: str) -> config_module.Config:
    if case not in CASE_UPDATES:
        raise ValueError(f"Unknown case {case!r}. Expected one of {sorted(CASE_UPDATES)}.")

    cfg = config_module.Config()
    for key, value in CASE_UPDATES[case].items():
        setattr(cfg, key, value)
    cfg.VAL_COUNTRY = val_country
    cfg.TRAIN_COUNTRIES = [iso for iso in ALL_COUNTRIES if iso != val_country]
    cfg.MODEL_SAVE_PATH = Path("models/country_cv") / f"neural_sde_sir_val{val_country}.eqx"
    return cfg




def _fold_context(case: str, val_country: str):
    cfg = _fold_config(case, val_country)
    data = _load_data(cfg)
    model, model_path = _load_model(cfg, data)
    start_pos = _start_position(data, cfg)
    start_date = str(data.val_features_df.index[start_pos].date())
    return cfg, data, model, model_path, start_pos, start_date


def run_geometry(cases: list[str], countries: list[str], lags: list[int], grid_size: int) -> None:
    rows: list[dict] = []
    for case in cases:
        for country in countries:
            print(f"geometry: {case} val={country}")
            cfg, data, model, model_path, _start_pos, start_date = _fold_context(case, country)
            residuals = _evaluate_bi_residuals(model, start_date, data, cfg)
            if residuals is None:
                raise RuntimeError(f"No residuals for {case} val={country}")
            for condition in ["raw_phi", "center_scale_phi", "shuffle_phi"]:
                metrics = _discrepancy_metrics(
                    residuals,
                    condition=condition,
                    lags=lags,
                    grid_size=grid_size,
                    seed=20260617 + len(rows),
                )
                rows.append(
                    {
                        "case": case,
                        "val_country": country,
                        "experiment": config_module.get_experiment_tag(cfg),
                        "start_date": start_date,
                        "model_path": str(model_path),
                        **metrics,
                    }
                )

    df = pd.DataFrame(rows)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GEOMETRY_OUT, index=False)

    metric_cols = [
        "interval_discrepancy_1d",
        "lag1_rect_discrepancy_2d",
        "lag7_rect_discrepancy_2d",
        "lag1_star_discrepancy_2d_grid",
        "lag7_star_discrepancy_2d_grid",
        "acf1",
        "acf7",
        "z_std",
    ]
    present = [col for col in metric_cols if col in df.columns]
    agg = (
        df.groupby(["case", "condition"], as_index=False)[present]
        .agg(["mean", "median", "max"])
        .reset_index()
    )
    agg.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]
    agg.to_csv(GEOMETRY_AGG_OUT, index=False)
    print(f"Saved {GEOMETRY_OUT}")
    print(f"Saved {GEOMETRY_AGG_OUT}")
    subset = agg[
        [
            "case",
            "condition",
            "interval_discrepancy_1d_mean",
            "lag1_rect_discrepancy_2d_mean",
            "lag7_rect_discrepancy_2d_mean",
            "acf1_mean",
            "acf7_mean",
        ]
    ]
    print(subset.to_string(index=False))


def _summarize_bootstrap_metric(
    case: str,
    country: str,
    experiment: str,
    start_date: str,
    n_bootstrap: int,
    metric: str,
    observed: float,
    null_values: np.ndarray,
) -> dict:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    tail = METRIC_TAILS[metric]
    pvalue = _empirical_pvalue(observed, values, tail)
    return {
        "case": case,
        "val_country": country,
        "experiment": experiment,
        "start_date": start_date,
        "n_bootstrap": int(n_bootstrap),
        "metric": metric,
        "tail": tail,
        "observed": float(observed),
        "null_mean": float(np.mean(values)),
        "null_std": float(np.std(values, ddof=1)),
        "null_q025": float(np.quantile(values, 0.025)),
        "null_q500": float(np.quantile(values, 0.500)),
        "null_q975": float(np.quantile(values, 0.975)),
        "empirical_percentile": float(np.mean(values <= observed)),
        "empirical_pvalue": float(pvalue),
        "significant_05": bool(pvalue < 0.05),
    }


def run_bootstrap_smoke(
    cases: list[str],
    countries: list[str],
    n_bootstrap: int,
    seed: int,
    write_full_aliases: bool = False,
) -> None:
    summary_rows: list[dict] = []
    draw_rows: list[dict] = []
    for case_idx, case in enumerate(cases):
        for country_idx, country in enumerate(countries):
            print(f"bootstrap-smoke: {case} val={country} n={n_bootstrap}")
            cfg, data, model, model_path, start_pos, start_date = _fold_context(case, country)
            observed_residuals = _evaluate_bi_residuals(model, start_date, data, cfg)
            if observed_residuals is None:
                raise RuntimeError(f"No observed residuals for {case} val={country}")
            observed = compute_brownian_inversion_metrics_from_residuals(observed_residuals)
            null_residuals, clip_fraction = _simulate_null_residuals(
                model,
                data,
                cfg,
                start_pos=start_pos,
                n_bootstrap=n_bootstrap,
                seed=seed + 100 * case_idx + country_idx,
            )
            case_draws = []
            for bootstrap_id, residuals in enumerate(null_residuals):
                row = {
                    "case": case,
                    "val_country": country,
                    "experiment": config_module.get_experiment_tag(cfg),
                    "start_date": start_date,
                    "model_path": str(model_path),
                    "bootstrap_id": int(bootstrap_id),
                    "clip_fraction": float(clip_fraction[bootstrap_id]),
                    **compute_brownian_inversion_metrics_from_residuals(residuals),
                }
                draw_rows.append(row)
                case_draws.append(row)
            draws_df = pd.DataFrame(case_draws)
            for metric in METRIC_TAILS:
                summary_rows.append(
                    _summarize_bootstrap_metric(
                        case=case,
                        country=country,
                        experiment=config_module.get_experiment_tag(cfg),
                        start_date=start_date,
                        n_bootstrap=n_bootstrap,
                        metric=metric,
                        observed=float(observed[metric]),
                        null_values=draws_df[metric].to_numpy(dtype=float),
                    )
                )

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    draws_df = pd.DataFrame(draw_rows)
    summary_df.to_csv(BOOTSTRAP_SUMMARY_OUT, index=False)
    draws_df.to_csv(BOOTSTRAP_DRAWS_OUT, index=False)
    print(f"Saved {BOOTSTRAP_SUMMARY_OUT}")
    print(f"Saved {BOOTSTRAP_DRAWS_OUT}")
    if write_full_aliases:
        summary_df.to_csv(BOOTSTRAP_FULL_SUMMARY_OUT, index=False)
        draws_df.to_csv(BOOTSTRAP_FULL_DRAWS_OUT, index=False)
        print(f"Saved {BOOTSTRAP_FULL_SUMMARY_OUT}")
        print(f"Saved {BOOTSTRAP_FULL_DRAWS_OUT}")

    summary = summary_df
    key_metrics = summary[summary["metric"].isin(["bi_z_std", "bi_acf1", "bi_acf7", "bi_lb_stat"])]
    print(
        key_metrics[
            [
                "case",
                "val_country",
                "metric",
                "observed",
                "null_q025",
                "null_q975",
                "empirical_pvalue",
                "significant_05",
            ]
        ].to_string(index=False)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["geometry", "bootstrap-smoke"])
    parser.add_argument("--cases", nargs="+", default=list(CASE_UPDATES))
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES)
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 7])
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--write-full-aliases",
        action="store_true",
        help="Also write the paper-facing country_cv_bi_bootstrap_full_*.csv aliases.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "geometry":
        run_geometry(
            cases=args.cases,
            countries=args.countries,
            lags=args.lags,
            grid_size=args.grid_size,
        )
    elif args.mode == "bootstrap-smoke":
        run_bootstrap_smoke(
            cases=args.cases,
            countries=args.countries,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            write_full_aliases=args.write_full_aliases,
        )
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
