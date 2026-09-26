"""
Build pre-specified global fitted-null scores for SIR innovation diagnostics.

This is an aggregation/calibration layer over existing frozen fitted-null
bootstrap draws.  It does not refit models.  The primary score is

    S_max = max_j d_j,

where d_j is the null-standardized departure of a pre-fixed component
diagnostic.  The paper-facing score uses a deterministic split of the
fitted-null bank: one half estimates the component standardizers, and the
other half is the ranking reference.  This keeps a ranked reference draw out
of its own standardization step.

Outputs:
  cache/summaries/bi_bootstrap/sir_bi_global_score.csv
  cache/summaries/bi_bootstrap/sir_bi_component_max_adjusted_profile.csv
  cache/summaries/country_cv/country_cv_bi_global_score.csv
  cache/summaries/country_cv/country_cv_bi_component_max_adjusted_profile.csv
  cache/summaries/country_cv/country_cv_bi_global_score_aggregate.csv
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from pathlib import Path

import numpy as np
import pandas as pd

from experiments.audit_statistics import (
    departure as _component_departures,
    standardize as _signed_standardized,
    upper_rank,
)
from experiments.country_bi_scoring import COMPONENTS, TAILS


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "cache" / "summaries"

MAIN_DRAWS = SUMMARY_DIR / "bi_bootstrap" / "sir_bi_bootstrap_draws.csv"
MAIN_SUMMARY = SUMMARY_DIR / "bi_bootstrap" / "sir_bi_bootstrap_summary.csv"
MAIN_OUT = SUMMARY_DIR / "bi_bootstrap" / "sir_bi_global_score.csv"
MAIN_PROFILE_OUT = SUMMARY_DIR / "bi_bootstrap" / "sir_bi_component_max_adjusted_profile.csv"

COUNTRY_DRAWS = SUMMARY_DIR / "country_cv" / "country_cv_bi_bootstrap_full_draws.csv"
COUNTRY_SUMMARY = SUMMARY_DIR / "country_cv" / "country_cv_bi_bootstrap_full_summary.csv"
COUNTRY_OUT = SUMMARY_DIR / "country_cv" / "country_cv_bi_global_score.csv"
COUNTRY_PROFILE_OUT = SUMMARY_DIR / "country_cv" / "country_cv_bi_component_max_adjusted_profile.csv"
COUNTRY_AGG_OUT = SUMMARY_DIR / "country_cv" / "country_cv_bi_global_score_aggregate.csv"


COMPONENT_LABELS = {
    "bi_z_std": "z_std",
    "bi_acf1": "ACF(1)",
    "bi_acf7": "ACF(7)",
    "bi_ks_stat": "KS stat",
    "bi_lb_stat": "Ljung-Box",
    "bi_max_w_norm": "max |W|/sqrt K",
}




def _empirical_upper_pvalue(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or null_values.size == 0:
        return float("nan")
    return upper_rank(observed, null_values)


def _observed_metrics(summary_group: pd.DataFrame) -> dict[str, float]:
    observed = {}
    for metric in COMPONENTS:
        row = summary_group.loc[summary_group["metric"].eq(metric)]
        if row.empty:
            raise ValueError(f"Missing observed metric {metric}")
        observed[metric] = float(row["observed"].iloc[0])
    return observed


def _standardizers(draws_group: pd.DataFrame) -> dict[str, tuple[float, float, str]]:
    standardizers = {}
    for metric in COMPONENTS:
        values = pd.to_numeric(draws_group[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite fitted-null values for {metric}")
        center = float(np.median(values))
        scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        standardizers[metric] = (center, scale, TAILS[metric])
    return standardizers


def _score_frame(
    frame: pd.DataFrame,
    standardizers: dict[str, tuple[float, float, str]],
) -> tuple[np.ndarray, np.ndarray]:
    component_departures = []
    for metric in COMPONENTS:
        center, scale, tail = standardizers[metric]
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
        component_departures.append(_component_departures(values, center, scale, tail))
    matrix = np.column_stack(component_departures)
    finite_rows = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite_rows]
    return np.max(matrix, axis=1), np.sqrt(np.mean(matrix * matrix, axis=1))


def _score_observed(
    observed: dict[str, float],
    standardizers: dict[str, tuple[float, float, str]],
) -> tuple[float, float, dict[str, float]]:
    departures = {}
    for metric in COMPONENTS:
        center, scale, tail = standardizers[metric]
        departures[metric] = float(
            _component_departures(np.array([observed[metric]]), center, scale, tail)[0]
        )
    vector = np.array([departures[metric] for metric in COMPONENTS], dtype=float)
    return float(np.max(vector)), float(np.sqrt(np.mean(vector * vector))), departures


def _split_mask(draws_group: pd.DataFrame) -> np.ndarray:
    ids = pd.to_numeric(draws_group["bootstrap_id"], errors="coerce")
    if ids.notna().all():
        return ids.astype(int).mod(2).eq(0).to_numpy()
    return np.arange(len(draws_group)) % 2 == 0


def _score_group(
    keys: dict[str, object],
    draws_group: pd.DataFrame,
    summary_group: pd.DataFrame,
) -> dict[str, object]:
    n_bootstrap = int(draws_group["bootstrap_id"].nunique())
    observed = _observed_metrics(summary_group)

    same_bank_standardizers = _standardizers(draws_group)
    same_bank_smax, same_bank_l2 = _score_frame(draws_group, same_bank_standardizers)
    same_obs_smax, same_obs_l2, same_observed_departures = _score_observed(
        observed,
        same_bank_standardizers,
    )

    calibration_mask = _split_mask(draws_group)
    calibration_group = draws_group.loc[calibration_mask].copy()
    reference_group = draws_group.loc[~calibration_mask].copy()
    if calibration_group.empty or reference_group.empty:
        raise ValueError(f"Cannot split fitted-null bank for {keys}")

    split_standardizers = _standardizers(calibration_group)
    null_smax, null_l2 = _score_frame(reference_group, split_standardizers)
    obs_smax, obs_l2, observed_departures = _score_observed(observed, split_standardizers)
    obs_vector = np.array([observed_departures[metric] for metric in COMPONENTS], dtype=float)
    dominant_idx = int(np.argmax(obs_vector))
    dominant_metric = COMPONENTS[dominant_idx]
    smax_pvalue = _empirical_upper_pvalue(obs_smax, null_smax)
    l2_pvalue = _empirical_upper_pvalue(obs_l2, null_l2)

    row: dict[str, object] = {
        **keys,
        "n_bootstrap": n_bootstrap,
        "score_rule": "split_calibrated_max_null_standardized_departure",
        "component_metrics": ",".join(COMPONENTS),
        "split_standardization_n": int(calibration_group["bootstrap_id"].nunique()),
        "split_reference_n": int(reference_group["bootstrap_id"].nunique()),
        "observed_smax": obs_smax,
        "null_smax_mean": float(np.mean(null_smax)),
        "null_smax_q025": float(np.quantile(null_smax, 0.025)),
        "null_smax_q500": float(np.quantile(null_smax, 0.500)),
        "null_smax_q975": float(np.quantile(null_smax, 0.975)),
        "smax_empirical_pvalue": smax_pvalue,
        "observed_l2": obs_l2,
        "null_l2_mean": float(np.mean(null_l2)),
        "null_l2_q025": float(np.quantile(null_l2, 0.025)),
        "null_l2_q500": float(np.quantile(null_l2, 0.500)),
        "null_l2_q975": float(np.quantile(null_l2, 0.975)),
        "l2_empirical_pvalue": l2_pvalue,
        "dominant_metric": dominant_metric,
        "dominant_component_departure": float(obs_vector[dominant_idx]),
        "significant_05": bool(smax_pvalue < 0.05),
        "same_bank_observed_smax": same_obs_smax,
        "same_bank_smax_empirical_pvalue": _empirical_upper_pvalue(
            same_obs_smax,
            same_bank_smax,
        ),
        "same_bank_observed_l2": same_obs_l2,
        "same_bank_l2_empirical_pvalue": _empirical_upper_pvalue(
            same_obs_l2,
            same_bank_l2,
        ),
    }
    for metric, departure in observed_departures.items():
        row[f"{metric}_departure"] = departure
    for metric, departure in same_observed_departures.items():
        row[f"same_bank_{metric}_departure"] = departure
    for metric, (center, scale, _tail) in split_standardizers.items():
        row[f"{metric}_null_median"] = center
        row[f"{metric}_null_std"] = scale
    for metric, (center, scale, _tail) in same_bank_standardizers.items():
        row[f"same_bank_{metric}_null_median"] = center
        row[f"same_bank_{metric}_null_std"] = scale
    return row


def _component_profile_group(
    keys: dict[str, object],
    draws_group: pd.DataFrame,
    summary_group: pd.DataFrame,
) -> list[dict[str, object]]:
    calibration_mask = _split_mask(draws_group)
    calibration_group = draws_group.loc[calibration_mask].copy()
    reference_group = draws_group.loc[~calibration_mask].copy()
    if calibration_group.empty or reference_group.empty:
        raise ValueError(f"Cannot split fitted-null bank for {keys}")

    split_standardizers = _standardizers(calibration_group)
    null_smax, _null_l2 = _score_frame(reference_group, split_standardizers)
    observed = _observed_metrics(summary_group)

    rows = []
    for metric in COMPONENTS:
        center, scale, tail = split_standardizers[metric]
        observed_value = float(observed[metric])
        signed_departure = float(
            _signed_standardized(np.array([observed_value]), center, scale)[0]
        )
        extremeness = float(
            _component_departures(np.array([observed_value]), center, scale, tail)[0]
        )
        reference_values = pd.to_numeric(reference_group[metric], errors="coerce").to_numpy(
            dtype=float
        )
        reference_extremeness = _component_departures(reference_values, center, scale, tail)
        summary_row = summary_group.loc[summary_group["metric"].eq(metric)]
        if summary_row.empty:
            raise ValueError(f"Missing summary row for {metric} and {keys}")
        summary_row = summary_row.iloc[0]
        finite_reference_values = reference_values[np.isfinite(reference_values)]
        max_adjusted_pvalue = _empirical_upper_pvalue(extremeness, null_smax)
        rows.append(
            {
                **keys,
                "metric": metric,
                "metric_label": COMPONENT_LABELS[metric],
                "tail": tail,
                "observed": observed_value,
                "null_q025": float(np.quantile(finite_reference_values, 0.025)),
                "null_q975": float(np.quantile(finite_reference_values, 0.975)),
                "pointwise_fitted_null_pvalue": _empirical_upper_pvalue(
                    extremeness,
                    reference_extremeness,
                ),
                "signed_split_departure": signed_departure,
                "extremeness": extremeness,
                "max_adjusted_pvalue": max_adjusted_pvalue,
                "simultaneous_flag_05": bool(max_adjusted_pvalue <= 0.05),
                "split_standardization_n": int(calibration_group["bootstrap_id"].nunique()),
                "split_reference_n": int(reference_group["bootstrap_id"].nunique()),
                "score_rule": "split_calibrated_max_null_standardized_departure",
            }
        )
    return rows


def build_global_scores(
    draws: pd.DataFrame,
    summary: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for group_key, draws_group in draws.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        keys = dict(zip(group_cols, group_key))
        mask = np.ones(len(summary), dtype=bool)
        for col, value in keys.items():
            mask &= summary[col].eq(value).to_numpy()
        summary_group = summary.loc[mask]
        rows.append(_score_group(keys, draws_group, summary_group))
    return pd.DataFrame(rows)


def build_component_profiles(
    draws: pd.DataFrame,
    summary: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for group_key, draws_group in draws.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        keys = dict(zip(group_cols, group_key))
        mask = np.ones(len(summary), dtype=bool)
        for col, value in keys.items():
            mask &= summary[col].eq(value).to_numpy()
        summary_group = summary.loc[mask]
        rows.extend(_component_profile_group(keys, draws_group, summary_group))
    return pd.DataFrame(rows)


def build_country_aggregate(country_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, group in country_scores.groupby("case", sort=True):
        pvalues = pd.to_numeric(group["smax_empirical_pvalue"], errors="coerce")
        rows.append(
            {
                "case": case,
                "n_countries": int(group["val_country"].nunique()),
                "mean_smax": float(group["observed_smax"].mean()),
                "median_smax_pvalue": float(pvalues.median()),
                "max_smax_pvalue": float(pvalues.max()),
                "significant_05_countries": int((pvalues < 0.05).sum()),
                "significant_10_countries": int((pvalues < 0.10).sum()),
                "dominant_metrics": ",".join(sorted(group["dominant_metric"].dropna().unique())),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    main_draws = pd.read_csv(MAIN_DRAWS)
    main_summary = pd.read_csv(MAIN_SUMMARY)
    main_scores = build_global_scores(
        draws=main_draws,
        summary=main_summary,
        group_cols=["case"],
    )
    main_profiles = build_component_profiles(
        draws=main_draws,
        summary=main_summary,
        group_cols=["case"],
    )
    MAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    main_scores.to_csv(MAIN_OUT, index=False)
    main_profiles.to_csv(MAIN_PROFILE_OUT, index=False)
    country_draws = pd.read_csv(COUNTRY_DRAWS)
    country_summary = pd.read_csv(COUNTRY_SUMMARY)
    country_scores = build_global_scores(
        draws=country_draws,
        summary=country_summary,
        group_cols=["case", "val_country"],
    )
    country_profiles = build_component_profiles(
        draws=country_draws,
        summary=country_summary,
        group_cols=["case", "val_country"],
    )
    country_scores.to_csv(COUNTRY_OUT, index=False)
    country_profiles.to_csv(COUNTRY_PROFILE_OUT, index=False)
    country_aggregate = build_country_aggregate(country_scores)
    country_aggregate.to_csv(COUNTRY_AGG_OUT, index=False)

    print(f"Wrote {MAIN_OUT.relative_to(ROOT)} ({len(main_scores)} rows)")
    print(f"Wrote {MAIN_PROFILE_OUT.relative_to(ROOT)} ({len(main_profiles)} rows)")
    print(f"Wrote {COUNTRY_OUT.relative_to(ROOT)} ({len(country_scores)} rows)")
    print(f"Wrote {COUNTRY_PROFILE_OUT.relative_to(ROOT)} ({len(country_profiles)} rows)")
    print(f"Wrote {COUNTRY_AGG_OUT.relative_to(ROOT)} ({len(country_aggregate)} rows)")
    print()
    print(
        main_scores[
            [
                "case",
                "observed_smax",
                "smax_empirical_pvalue",
                "observed_l2",
                "l2_empirical_pvalue",
                "dominant_metric",
            ]
        ].to_string(index=False)
    )
    print()
    print(country_aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
