"""
Build a SIR Brownian-inversion audit scorecard from existing summary CSVs.

The scorecard is an aggregation/reporting layer, not a new model fit.  It joins
path/forecast metrics, BI diagnostics, and fitted-null bootstrap effect sizes.

Outputs:
  cache/summaries/sir_bi_audit_scorecard.csv
  cache/summaries/sir_bi_audit_main_table.csv
"""

from __future__ import annotations


import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "cache" / "summaries"

BI_COMPARISON = SUMMARY_DIR / "sir_bi_extended_comparison.csv"
PATH_DIAGNOSTIC = SUMMARY_DIR / "sir_swd_kernel_diagnostic.csv"
TRECGAMMA = SUMMARY_DIR / "sir_trec_gamma_decoupling.csv"
BOOTSTRAP = SUMMARY_DIR / "bi_bootstrap" / "sir_bi_bootstrap_summary.csv"

OUT_SCORECARD = SUMMARY_DIR / "sir_bi_audit_scorecard.csv"
OUT_MAIN_TABLE = SUMMARY_DIR / "sir_bi_audit_main_table.csv"


PAPER_ROLES = {
    "baseline": "trainable baseline; context for baseline failure",
    "baseline_frzgamma": "core baseline Brownian failure",
    "ctrl_zero_frzgamma": "control ablation; signature/context sanity check",
    "ctrl_shuffle_frzgamma": "control ablation; temporal-order sanity check",
    "recov28": "T_obs=28 trainable partial repair",
    "recov28_frzgamma": "core partial repair: path/scale improve, ACF remains",
    "recov28_dyn14": "T_obs/T_dyn decoupling probe",
    "baseline_zwd0p1_frzgamma": "weak marginal-residual regularization probe",
    "baseline_zwd1_frzgamma": "negative control: marginal/path improve, memory worsens",
}

INTERPRETATIONS = {
    "baseline_frzgamma": "Baseline fails scale and temporal fitted-null audits.",
    "recov28_frzgamma": "Path and marginal scale improve, but fitted-null temporal failure remains.",
    "baseline_zwd1_frzgamma": "Marginal/path-oriented training worsens temporal dependence.",
}




def _first_nonnull(series: pd.Series) -> float | str | None:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def _rms(values: list[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return np.nan
    return math.sqrt(float(np.mean(np.square(vals))))


def _format_score(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_base_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if BI_COMPARISON.exists():
        bi = pd.read_csv(BI_COMPARISON)
        frames.append(
            bi.rename(
                columns={
                    "bi_z_std": "z_std",
                    "bi_acf1": "acf1",
                    "bi_acf7": "acf7",
                    "bi_ks_pvalue": "ks_pvalue",
                    "bi_lb_pvalue": "lb_pvalue",
                    "bi_max_w_norm": "max_w_norm",
                }
            )[
                [
                    "case",
                    "z_std",
                    "acf1",
                    "acf7",
                    "ks_pvalue",
                    "lb_pvalue",
                    "max_w_norm",
                ]
            ]
        )

    if PATH_DIAGNOSTIC.exists():
        path = pd.read_csv(PATH_DIAGNOSTIC)
        frames.append(
            path.rename(
                columns={
                    "own_swd_logI_path": "path_swd",
                    "own_w1_terminal_logI": "terminal_w1",
                    "own_median_path_mae_logI": "median_path_mae",
                    "bi_z_std": "z_std",
                    "bi_acf1": "acf1",
                    "bi_acf7": "acf7",
                    "bi_ks_pvalue": "ks_pvalue",
                    "bi_lb_pvalue": "lb_pvalue",
                }
            )[
                [
                    "case",
                    "path_swd",
                    "terminal_w1",
                    "median_path_mae",
                    "z_std",
                    "acf1",
                    "acf7",
                    "ks_pvalue",
                    "lb_pvalue",
                ]
            ]
        )

    if TRECGAMMA.exists():
        trec = pd.read_csv(TRECGAMMA)
        trec = trec.assign(case=trec["source_case"])
        frames.append(
            trec.rename(
                columns={
                    "forecast_smape": "smape",
                    "own_swd_logI_path": "path_swd",
                    "own_w1_terminal_logI": "terminal_w1",
                    "own_median_path_mae_logI": "median_path_mae",
                    "bi_z_std": "z_std",
                    "bi_acf1": "acf1",
                    "bi_acf7": "acf7",
                    "bi_ks_pvalue": "ks_pvalue",
                    "bi_lb_pvalue": "lb_pvalue",
                }
            )[
                [
                    "case",
                    "smape",
                    "path_swd",
                    "terminal_w1",
                    "median_path_mae",
                    "z_std",
                    "acf1",
                    "acf7",
                    "ks_pvalue",
                    "lb_pvalue",
                ]
            ]
        )

    joined = pd.concat(frames, ignore_index=True, sort=False)
    numeric_cols = [c for c in joined.columns if c != "case"]
    for col in numeric_cols:
        joined[col] = pd.to_numeric(joined[col], errors="coerce")

    base = (
        joined.groupby("case", as_index=False)
        .agg({col: _first_nonnull for col in numeric_cols})
        .sort_values("case")
        .reset_index(drop=True)
    )
    return base


def add_bootstrap_effects(base: pd.DataFrame) -> pd.DataFrame:
    if not BOOTSTRAP.exists():
        base["has_fitted_null"] = False
        return base

    boot = pd.read_csv(BOOTSTRAP)
    boot["observed"] = pd.to_numeric(boot["observed"], errors="coerce")
    boot["null_mean"] = pd.to_numeric(boot["null_mean"], errors="coerce")
    boot["null_std"] = pd.to_numeric(boot["null_std"], errors="coerce")
    boot["effect"] = (boot["observed"] - boot["null_mean"]) / boot["null_std"]

    effects = boot.pivot_table(index="case", columns="metric", values="effect", aggfunc="first")
    effects = effects.rename(
        columns={
            "bi_z_std": "bootstrap_effect_z_std",
            "bi_ks_stat": "bootstrap_effect_ks_stat",
            "bi_acf1": "bootstrap_effect_acf1",
            "bi_acf7": "bootstrap_effect_acf7",
            "bi_lb_stat": "bootstrap_effect_lb",
            "bi_max_w_norm": "bootstrap_effect_max_w_norm",
        }
    ).reset_index()

    observed_stats = boot.pivot_table(index="case", columns="metric", values="observed", aggfunc="first")
    observed_stats = observed_stats.rename(
        columns={
            "bi_ks_stat": "ks_stat",
            "bi_lb_stat": "ljungbox_stat",
        }
    )[["ks_stat", "ljungbox_stat"]].reset_index()

    out = base.merge(effects, on="case", how="left").merge(observed_stats, on="case", how="left")
    out["has_fitted_null"] = out["bootstrap_effect_z_std"].notna()
    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["z_std_gap"] = (out["z_std"] - 1.0).abs()

    baseline_path = out.loc[out["case"].eq("baseline_frzgamma"), "path_swd"]
    baseline_z_gap = out.loc[out["case"].eq("baseline_frzgamma"), "z_std_gap"]
    baseline_acf7 = out.loc[out["case"].eq("baseline_frzgamma"), "acf7"]

    if not baseline_path.empty and pd.notna(baseline_path.iloc[0]):
        out["path_swd_ratio_vs_baseline_frzgamma"] = out["path_swd"] / float(baseline_path.iloc[0])
        out["path_swd_improvement_pct_vs_baseline_frzgamma"] = 100.0 * (
            1.0 - out["path_swd_ratio_vs_baseline_frzgamma"]
        )

    if not baseline_z_gap.empty and pd.notna(baseline_z_gap.iloc[0]):
        out["z_gap_improvement_pct_vs_baseline_frzgamma"] = 100.0 * (
            1.0 - out["z_std_gap"] / float(baseline_z_gap.iloc[0])
        )

    if not baseline_acf7.empty and pd.notna(baseline_acf7.iloc[0]):
        out["acf7_change_vs_baseline_frzgamma"] = out["acf7"] - float(baseline_acf7.iloc[0])

    out["marginal_null_distance_rms"] = out.apply(
        lambda r: _rms([r.get("bootstrap_effect_z_std"), r.get("bootstrap_effect_ks_stat")]),
        axis=1,
    )
    out["temporal_acf_null_distance_rms"] = out.apply(
        lambda r: _rms([r.get("bootstrap_effect_acf1"), r.get("bootstrap_effect_acf7")]),
        axis=1,
    )
    out["temporal_null_distance_rms"] = out.apply(
        lambda r: _rms(
            [
                r.get("bootstrap_effect_acf1"),
                r.get("bootstrap_effect_acf7"),
                r.get("bootstrap_effect_lb"),
            ]
        ),
        axis=1,
    )
    out["paper_role"] = out["case"].map(PAPER_ROLES).fillna("supporting SIR configuration")
    return out


def build_main_table(scorecard: pd.DataFrame) -> pd.DataFrame:
    cases = ["baseline_frzgamma", "recov28_frzgamma", "baseline_zwd1_frzgamma"]
    main = scorecard[scorecard["case"].isin(cases)].copy()
    main["model"] = pd.Categorical(main["case"], categories=cases, ordered=True)
    main = main.sort_values("model")

    return pd.DataFrame(
        {
            "Model": main["case"].astype(str),
            "Path score (own SWD; lower better)": main["path_swd"].map(lambda x: _format_score(x, 3)),
            "Marginal score (null-distance RMS; lower better)": main[
                "marginal_null_distance_rms"
            ].map(lambda x: _format_score(x, 2)),
            "Temporal score (null-distance RMS; lower better)": main[
                "temporal_null_distance_rms"
            ].map(lambda x: _format_score(x, 2)),
            "ACF-only temporal score": main["temporal_acf_null_distance_rms"].map(
                lambda x: _format_score(x, 2)
            ),
            "Interpretation": main["case"].map(INTERPRETATIONS),
        }
    )


def main() -> None:
    scorecard = add_scores(add_bootstrap_effects(build_base_table()))

    preferred_cols = [
        "case",
        "paper_role",
        "has_fitted_null",
        "smape",
        "path_swd",
        "terminal_w1",
        "median_path_mae",
        "z_std",
        "z_std_gap",
        "ks_stat",
        "ks_pvalue",
        "acf1",
        "acf7",
        "ljungbox_stat",
        "lb_pvalue",
        "max_w_norm",
        "bootstrap_effect_z_std",
        "bootstrap_effect_ks_stat",
        "bootstrap_effect_acf1",
        "bootstrap_effect_acf7",
        "bootstrap_effect_lb",
        "bootstrap_effect_max_w_norm",
        "marginal_null_distance_rms",
        "temporal_acf_null_distance_rms",
        "temporal_null_distance_rms",
        "path_swd_ratio_vs_baseline_frzgamma",
        "path_swd_improvement_pct_vs_baseline_frzgamma",
        "z_gap_improvement_pct_vs_baseline_frzgamma",
        "acf7_change_vs_baseline_frzgamma",
    ]
    scorecard = scorecard[[c for c in preferred_cols if c in scorecard.columns]]
    scorecard.to_csv(OUT_SCORECARD, index=False)

    main_table = build_main_table(scorecard)
    main_table.to_csv(OUT_MAIN_TABLE, index=False)

    print(f"Wrote {OUT_SCORECARD.relative_to(ROOT)} ({len(scorecard)} rows)")
    print(f"Wrote {OUT_MAIN_TABLE.relative_to(ROOT)} ({len(main_table)} rows)")


if __name__ == "__main__":
    main()
