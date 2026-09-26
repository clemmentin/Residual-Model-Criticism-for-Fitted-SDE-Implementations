"""
Split-null calibration summary for the two-layer SIR BI audit.

This is a lightweight diagnostic over the fitted-null draws already produced by
run_sir_two_layer_bi_audit.py.  For each case, the null draws are split into two
halves; each half is used as the reference bank for the other half.  The output
checks whether rank p-values for the new two-layer metrics are grossly
miscalibrated under the fitted null.

It is not an end-to-end Neural SDE retraining calibration experiment.

Run from the sde directory:
    python experiments/summarize_two_layer_bi_null_split.py
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
from scipy.stats import kstest


ROOT = Path(__file__).resolve().parents[1]


from experiments.audit_statistics import upper_rank
from experiments.run_sir_two_layer_bi_audit import METRIC_TAILS


DEFAULT_DIR = ROOT / "cache" / "summaries" / "two_layer_bi"
ALPHAS = (0.01, 0.05, 0.10)


def _component_extremeness(values: np.ndarray, center: float, scale: float, tail: str) -> np.ndarray:
    safe_scale = max(float(scale), 1e-12)
    z = (np.asarray(values, dtype=float) - center) / safe_scale
    if tail == "upper":
        return np.maximum(z, 0.0)
    if tail == "centered":
        return np.abs(z)
    raise ValueError(f"Unknown tail: {tail}")


def _upper_rank_pvalue(observed: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if not np.isfinite(observed) or reference.size == 0:
        return float("nan")
    return upper_rank(observed, reference)


def _rank_pvalues(eval_values: np.ndarray, ref_values: np.ndarray, tail: str) -> np.ndarray:
    eval_values = np.asarray(eval_values, dtype=float)
    ref_values = np.asarray(ref_values, dtype=float)
    ref_values = ref_values[np.isfinite(ref_values)]
    out = np.full(eval_values.shape, np.nan, dtype=float)
    if ref_values.size == 0:
        return out

    if tail == "upper":
        sorted_ref = np.sort(ref_values)
        ok = np.isfinite(eval_values)
        n_ge = ref_values.size - np.searchsorted(sorted_ref, eval_values[ok], side="left")
        out[ok] = (1.0 + n_ge) / (1.0 + ref_values.size)
        return out

    if tail == "centered":
        center = float(np.median(ref_values))
        ref_dist = np.sort(np.abs(ref_values - center))
        ok = np.isfinite(eval_values)
        eval_dist = np.abs(eval_values[ok] - center)
        n_ge = ref_dist.size - np.searchsorted(ref_dist, eval_dist, side="left")
        out[ok] = (1.0 + n_ge) / (1.0 + ref_dist.size)
        return out

    raise ValueError(f"Unknown tail: {tail}")


def _twofold_pvalues(case_df: pd.DataFrame, metric: str, tail: str) -> np.ndarray:
    ordered = case_df.sort_values("bootstrap_id").reset_index(drop=True)
    even = ordered.index.to_numpy() % 2 == 0
    values = ordered[metric].to_numpy(dtype=float)
    p_even = _rank_pvalues(values[even], values[~even], tail)
    p_odd = _rank_pvalues(values[~even], values[even], tail)
    out = np.full(values.shape, np.nan, dtype=float)
    out[even] = p_even
    out[~even] = p_odd
    return out


def _observed_split_profile(draws: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, case_df in draws.groupby("case", sort=True):
        ordered = case_df.sort_values("bootstrap_id").reset_index(drop=True)
        bootstrap_id = ordered["bootstrap_id"].astype(int).to_numpy()
        pilot = ordered.loc[bootstrap_id % 2 == 0].copy()
        reference = ordered.loc[bootstrap_id % 2 == 1].copy()
        if pilot.empty or reference.empty:
            continue

        summary_case = summary.loc[summary["case"].eq(case)]
        for metric, tail in METRIC_TAILS.items():
            if metric not in ordered.columns:
                continue
            summary_row = summary_case.loc[summary_case["metric"].eq(metric)]
            if summary_row.empty:
                continue
            observed = float(summary_row["observed"].iloc[0])
            pilot_values = pd.to_numeric(pilot[metric], errors="coerce").to_numpy(dtype=float)
            reference_values = pd.to_numeric(reference[metric], errors="coerce").to_numpy(dtype=float)
            pilot_values = pilot_values[np.isfinite(pilot_values)]
            reference_values = reference_values[np.isfinite(reference_values)]
            if pilot_values.size == 0 or reference_values.size == 0:
                continue

            center = float(np.median(pilot_values))
            scale = float(np.std(pilot_values, ddof=1)) if pilot_values.size > 1 else 0.0
            observed_extremeness = float(_component_extremeness([observed], center, scale, tail)[0])
            reference_extremeness = _component_extremeness(reference_values, center, scale, tail)
            rows.append(
                {
                    "case": case,
                    "metric": metric,
                    "tail": tail,
                    "observed": observed,
                    "null_q025": float(np.quantile(reference_values, 0.025)),
                    "null_q975": float(np.quantile(reference_values, 0.975)),
                    "split_pointwise_pvalue": _upper_rank_pvalue(
                        observed_extremeness,
                        reference_extremeness,
                    ),
                    "pilot_n": int(pilot_values.size),
                    "reference_n": int(reference_values.size),
                    "split_rule": "even_bootstrap_id_pilot_odd_bootstrap_id_reference",
                }
            )
    return pd.DataFrame(rows)


def _centering_comparison(
    observed_summary: pd.DataFrame,
    observed_profile: pd.DataFrame,
) -> pd.DataFrame:
    required = {"diff_perp_rms", "diff_step_perp_rms", "mean_gap_perp_rms"}
    if observed_summary.empty or observed_profile.empty:
        return pd.DataFrame()
    if not required.issubset(set(observed_summary["metric"])):
        return pd.DataFrame()

    rows = []
    for case in sorted(set(observed_summary["case"])):
        summary_case = observed_summary.loc[observed_summary["case"].eq(case)]
        profile_case = observed_profile.loc[observed_profile["case"].eq(case)]
        if not required.issubset(set(summary_case["metric"])):
            continue
        if not required.issubset(set(profile_case["metric"])):
            continue

        def summary_value(metric: str, column: str) -> float:
            return float(summary_case.loc[summary_case["metric"].eq(metric), column].iloc[0])

        def profile_value(metric: str, column: str) -> float:
            return float(profile_case.loc[profile_case["metric"].eq(metric), column].iloc[0])

        local_observed = summary_value("diff_perp_rms", "observed")
        step_observed = summary_value("diff_step_perp_rms", "observed")
        local_null_median = summary_value("diff_perp_rms", "null_q500")
        step_null_median = summary_value("diff_step_perp_rms", "null_q500")
        rows.append(
            {
                "case": case,
                "localmean_perp_rms_observed": local_observed,
                "stepmean_perp_rms_observed": step_observed,
                "observed_change_pct": 100.0 * (step_observed / local_observed - 1.0),
                "localmean_perp_rms_null_median": local_null_median,
                "stepmean_perp_rms_null_median": step_null_median,
                "null_median_change_pct": 100.0
                * (step_null_median / local_null_median - 1.0),
                "localmean_perp_rms_split_pvalue": profile_value(
                    "diff_perp_rms", "split_pointwise_pvalue"
                ),
                "stepmean_perp_rms_split_pvalue": profile_value(
                    "diff_step_perp_rms", "split_pointwise_pvalue"
                ),
                "mean_gap_perp_rms_observed": summary_value(
                    "mean_gap_perp_rms", "observed"
                ),
                "mean_gap_perp_rms_null_median": summary_value(
                    "mean_gap_perp_rms", "null_q500"
                ),
                "mean_gap_perp_rms_split_pvalue": profile_value(
                    "mean_gap_perp_rms", "split_pointwise_pvalue"
                ),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.dir.resolve()
    in_path = output_dir / "sir_two_layer_bi_draws.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing two-layer BI draws: {in_path}")
    draws = pd.read_csv(in_path)
    summary_path = output_dir / "sir_two_layer_bi_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing two-layer BI summary: {summary_path}")
    observed_summary = pd.read_csv(summary_path)

    pvalue_rows = []
    summary_rows = []
    for case, case_df in draws.groupby("case", sort=True):
        for metric, tail in METRIC_TAILS.items():
            if metric not in case_df.columns:
                continue
            pvals = _twofold_pvalues(case_df, metric, tail)
            finite = pvals[np.isfinite(pvals)]
            if finite.size == 0:
                continue
            ks_stat, ks_p = kstest(finite, "uniform")
            row = {
                "case": case,
                "metric": metric,
                "tail": tail,
                "n_eval": int(finite.size),
                "p_mean": float(np.mean(finite)),
                "p_sd": float(np.std(finite, ddof=1)),
                "p_q05": float(np.quantile(finite, 0.05)),
                "p_q50": float(np.quantile(finite, 0.50)),
                "p_q95": float(np.quantile(finite, 0.95)),
                "uniform_ks_stat": float(ks_stat),
                "uniform_ks_pvalue": float(ks_p),
            }
            for alpha in ALPHAS:
                row[f"reject_rate_alpha_{alpha:.2f}"] = float(np.mean(finite <= alpha))
            summary_rows.append(row)
            for bootstrap_id, pvalue in zip(case_df.sort_values("bootstrap_id")["bootstrap_id"], pvals):
                pvalue_rows.append(
                    {
                        "case": case,
                        "metric": metric,
                        "bootstrap_id": int(bootstrap_id),
                        "split_null_pvalue": float(pvalue),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    pvalues = pd.DataFrame(pvalue_rows)
    observed_profile = _observed_split_profile(draws, observed_summary)
    centering_comparison = _centering_comparison(observed_summary, observed_profile)
    summary_path = output_dir / "sir_two_layer_bi_null_split_uniformity.csv"
    pvalue_path = output_dir / "sir_two_layer_bi_null_split_pvalues.csv"
    observed_profile_path = output_dir / "sir_two_layer_bi_observed_split_profile.csv"
    centering_comparison_path = output_dir / "sir_two_layer_bi_centering_comparison.csv"
    summary.to_csv(summary_path, index=False)
    pvalues.to_csv(pvalue_path, index=False)
    observed_profile.to_csv(observed_profile_path, index=False)
    if not centering_comparison.empty:
        centering_comparison.to_csv(centering_comparison_path, index=False)

    headline = summary[
        summary["metric"].isin(
            [
                "step_z1_std",
                "step_z1_acf1",
                "step_z1_acf7",
                "step_zero_rms",
                "diff_z_std",
                "diff_z_acf1",
                "diff_z_acf7",
                "diff_perp_rms",
                "diff_step_z_std",
                "diff_step_z_acf1",
                "diff_step_z_acf7",
                "diff_step_perp_rms",
                "mean_gap_perp_rms",
            ]
        )
    ][
        [
            "case",
            "metric",
            "n_eval",
            "p_mean",
            "reject_rate_alpha_0.05",
            "uniform_ks_pvalue",
        ]
    ]
    print(headline.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved pvalues: {pvalue_path}")
    if not centering_comparison.empty:
        print(f"Saved centering comparison: {centering_comparison_path}")


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
