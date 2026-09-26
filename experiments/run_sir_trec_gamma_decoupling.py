"""
run_sir_trec_gamma_decoupling.py

Build a clean T_rec x gamma table from cached SIR diagnostics.

Primary 2x2 grid:
  observation T_rec in {14, 28} x gamma residual {trainable, frozen}

Extra row:
  T_rec=28 with gamma_base decoupled to 1/14 via SIR_RECOVERY_DAYS_DYN=14.

The source diagnostic is cache/summaries/sir_swd_kernel_diagnostic.csv. Re-run
experiments/run_sir_swd_kernel_diagnostic.py first if that file is missing.

Output
------
cache/summaries/sir_trec_gamma_decoupling.csv
"""

from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_DIR = ROOT / "cache" / "summaries"
SOURCE_PATH = SUMMARY_DIR / "sir_swd_kernel_diagnostic.csv"
OUT_PATH = SUMMARY_DIR / "sir_trec_gamma_decoupling.csv"

CASE_SPECS = [
    {
        "case": "obs14_gamma_trainable",
        "source_case": "baseline",
        "obs_recovery_days": 14.0,
        "gamma_mode": "trainable",
        "gamma_dyn_days": np.nan,
        "gamma_decoupled": False,
        "primary_grid": True,
    },
    {
        "case": "obs14_gamma_frozen",
        "source_case": "baseline_frzgamma",
        "obs_recovery_days": 14.0,
        "gamma_mode": "frozen",
        "gamma_dyn_days": np.nan,
        "gamma_decoupled": False,
        "primary_grid": True,
    },
    {
        "case": "obs28_gamma_trainable",
        "source_case": "recov28",
        "obs_recovery_days": 28.0,
        "gamma_mode": "trainable",
        "gamma_dyn_days": np.nan,
        "gamma_decoupled": False,
        "primary_grid": True,
    },
    {
        "case": "obs28_gamma_frozen",
        "source_case": "recov28_frzgamma",
        "obs_recovery_days": 28.0,
        "gamma_mode": "frozen",
        "gamma_dyn_days": np.nan,
        "gamma_decoupled": False,
        "primary_grid": True,
    },
    {
        "case": "obs28_gamma_dyn14_trainable",
        "source_case": "recov28_dyn14",
        "obs_recovery_days": 28.0,
        "gamma_mode": "trainable",
        "gamma_dyn_days": 14.0,
        "gamma_decoupled": True,
        "primary_grid": False,
    },
]

SUMMARY_COLUMNS = [
    "forecast_mae",
    "forecast_rmse",
    "forecast_smape",
    "rolling_mae",
    "rolling_rmse",
    "rolling_coverage_90",
    "rolling_width_mean",
]

OUTPUT_COLUMNS = [
    "case",
    "source_case",
    "experiment",
    "obs_recovery_days",
    "gamma_mode",
    "gamma_dyn_days",
    "gamma_decoupled",
    "primary_grid",
    "start_date",
    "model_path",
    "model_exists",
    "summary_status",
    *SUMMARY_COLUMNS,
    "val_total_loss",
    "val_transition_nll",
    "val_loss_mse",
    "val_loss_bi_z_std",
    "own_swd_logI_path",
    "own_swd_dlogI_path",
    "own_w1_terminal_logI",
    "own_median_path_mae_logI",
    "ref28_swd_logI_path",
    "ref28_swd_dlogI_path",
    "ref28_w1_terminal_logI",
    "ref28_median_path_mae_logI",
    "bi_z_std",
    "bi_acf1",
    "bi_acf7",
    "bi_ks_pvalue",
    "bi_lb_pvalue",
]


def _load_summary_metrics(experiment: str) -> tuple[dict[str, float], str]:
    summary_path = SUMMARY_DIR / f"summary_{experiment}.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing forecast summary: {summary_path}")

    summary = pd.read_csv(summary_path)
    if len(summary) != 1:
        raise ValueError(f"Expected one forecast-summary row: {summary_path}")
    missing = set(SUMMARY_COLUMNS).difference(summary.columns)
    if missing:
        raise ValueError(f"Missing forecast columns in {summary_path}: {sorted(missing)}")
    metrics = {col: float(summary.iloc[0][col]) for col in SUMMARY_COLUMNS}
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError(f"Non-finite forecast metrics in {summary_path}")
    return metrics, "present"


def _model_path_for(experiment: str) -> tuple[str, bool]:
    path = ROOT / "models" / f"neural_sde_sir_{experiment}.eqx"
    display = str(path.relative_to(ROOT))
    return display, path.exists()


def build_table() -> pd.DataFrame:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_PATH}. Run experiments/run_sir_swd_kernel_diagnostic.py first."
        )

    source = pd.read_csv(SOURCE_PATH)
    rows = []
    for spec in CASE_SPECS:
        matched = source[source["case"] == spec["source_case"]]
        if matched.empty:
            raise ValueError(f"Source case {spec['source_case']!r} is missing from {SOURCE_PATH}")
        source_row = matched.iloc[0].to_dict()
        experiment = str(source_row["experiment"])
        summary_metrics, summary_status = _load_summary_metrics(experiment)
        model_path, model_exists = _model_path_for(experiment)

        row = {
            **spec,
            "experiment": experiment,
            "start_date": source_row.get("start_date"),
            "model_path": model_path,
            "model_exists": bool(model_exists),
            "summary_status": summary_status,
            **summary_metrics,
        }
        for col in OUTPUT_COLUMNS:
            if col not in row and col in source_row:
                row[col] = source_row[col]
        rows.append(row)

    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def main() -> None:
    table = build_table()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False, float_format="%.8g")

    cols = [
        "case",
        "summary_status",
        "forecast_smape",
        "bi_z_std",
        "bi_acf1",
        "bi_acf7",
        "own_swd_logI_path",
        "ref28_swd_logI_path",
    ]
    print(f"Saved T_rec x gamma decoupling table to {OUT_PATH}")
    print(table[cols].to_string(index=False))


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
