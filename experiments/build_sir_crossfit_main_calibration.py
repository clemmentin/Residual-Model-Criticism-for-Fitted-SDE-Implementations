"""
Build the paper-facing country-LOO held-out model-check summary.

This is a derived table over the existing country leave-one-out fitted-null
artifacts.  It does not train models and does not simulate new null paths.
Each row summarizes one model configuration across the nine held-out countries:

  * the pre-specified split-calibrated global S_max fitted-null score;
  * componentwise fitted-null localization counts;
  * country-LOO path/BI means for context.

Output:
  cache/summaries/country_cv/country_cv_crossfit_main_calibration.csv
"""

from __future__ import annotations


from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "cache" / "summaries"
COUNTRY_DIR = SUMMARY_DIR / "country_cv"

GLOBAL_SCORE = COUNTRY_DIR / "country_cv_bi_global_score.csv"
BOOTSTRAP_SUMMARY = COUNTRY_DIR / "country_cv_bi_bootstrap_full_summary.csv"
COMPONENT_PROFILE = COUNTRY_DIR / "country_cv_bi_component_max_adjusted_profile.csv"
OUT = COUNTRY_DIR / "country_cv_crossfit_main_calibration.csv"

CASES = ("baseline", "baseline_frzgamma", "recov28", "recov28_frzgamma")
METRICS = ("bi_z_std", "bi_acf1", "bi_acf7", "bi_lb_stat", "bi_ks_stat")
TEMPORAL_METRICS = ("bi_acf1", "bi_acf7", "bi_lb_stat")


def _country_summary_path(case: str) -> Path:
    return COUNTRY_DIR / f"country_cv_summary_{case}.csv"




def _count_significant(profile: pd.DataFrame, case: str, metric: str) -> int:
    sub = profile.loc[profile["case"].eq(case) & profile["metric"].eq(metric)]
    pvalues = pd.to_numeric(sub["pointwise_fitted_null_pvalue"], errors="coerce")
    return int((pvalues <= 0.05).sum())


def _all_temporal_significant_count(profile: pd.DataFrame, case: str) -> int:
    sub = profile.loc[profile["case"].eq(case) & profile["metric"].isin(TEMPORAL_METRICS)].copy()
    sub["split_significant_05"] = (
        pd.to_numeric(sub["pointwise_fitted_null_pvalue"], errors="coerce") <= 0.05
    )
    wide = sub.pivot_table(
        index="val_country",
        columns="metric",
        values="split_significant_05",
        aggfunc="first",
    )
    present = [metric for metric in TEMPORAL_METRICS if metric in wide.columns]
    if len(present) != len(TEMPORAL_METRICS):
        return 0
    flags = wide[present].astype(bool)
    return int(flags.all(axis=1).sum())


def build() -> pd.DataFrame:
    global_score = pd.read_csv(GLOBAL_SCORE)
    bootstrap = pd.read_csv(BOOTSTRAP_SUMMARY)
    component_profile = pd.read_csv(COMPONENT_PROFILE)
    rows: list[dict[str, object]] = []

    for case in CASES:
        score_sub = global_score.loc[global_score["case"].eq(case)].copy()
        boot_sub = bootstrap.loc[bootstrap["case"].eq(case)].copy()
        country_summary = pd.read_csv(_country_summary_path(case))

        pvalues = pd.to_numeric(score_sub["smax_empirical_pvalue"], errors="coerce")
        smape = pd.to_numeric(country_summary["forecast_smape"], errors="coerce")
        z_std = pd.to_numeric(country_summary["bi_z_std"], errors="coerce")
        acf1 = pd.to_numeric(country_summary["bi_acf1"], errors="coerce")
        acf7 = pd.to_numeric(country_summary["bi_acf7"], errors="coerce")

        row: dict[str, object] = {
            "case": case,
            "audit_design": "country_leave_one_out_frozen_protocol",
            "n_countries": int(score_sub["val_country"].nunique()),
            "n_bootstrap_per_cell": int(boot_sub["n_bootstrap"].max()),
            "score_rule": "split_calibrated_max_null_standardized_departure",
            "global_smax_significant_05_countries": int((pvalues < 0.05).sum()),
            "global_smax_median_pvalue": float(pvalues.median()),
            "global_smax_max_pvalue": float(pvalues.max()),
            "temporal_all_three_significant_05_countries": _all_temporal_significant_count(
                component_profile, case
            ),
            "forecast_smape_mean": float(smape.mean()),
            "bi_z_std_mean": float(z_std.mean()),
            "bi_acf1_mean": float(acf1.mean()),
            "bi_acf7_mean": float(acf7.mean()),
            "dominant_metrics": ",".join(sorted(score_sub["dominant_metric"].dropna().unique())),
        }
        for metric in METRICS:
            row[f"{metric}_significant_05_countries"] = _count_significant(
                component_profile, case, metric
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table = build()
    table.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(table)} rows)")
    print(
        table[
            [
                "case",
                "n_countries",
                "global_smax_significant_05_countries",
                "global_smax_max_pvalue",
                "temporal_all_three_significant_05_countries",
                "forecast_smape_mean",
                "bi_z_std_mean",
                "dominant_metrics",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
