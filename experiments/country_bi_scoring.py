"""Shared split-calibrated country-BI score construction.

The confirmatory runner and the deterministic paper-table builder must use the
same component transformations, pilot/evaluation split, and rank convention.
Keeping that logic here prevents the two reporting paths from drifting.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from experiments.audit_statistics import departure as departure, upper_rank


COMPONENTS = (
    "bi_z_std",
    "bi_acf1",
    "bi_acf7",
    "bi_ks_stat",
    "bi_lb_stat",
    "bi_max_w_norm",
)

TAILS = {
    "bi_z_std": "centered",
    "bi_acf1": "centered",
    "bi_acf7": "centered",
    "bi_ks_stat": "upper",
    "bi_lb_stat": "upper",
    "bi_max_w_norm": "upper",
}


def upper_rank_pvalue(observed: float, reference: np.ndarray) -> float:
    """Compute the finite-sample upper-rank p-value with the plus-one rule."""
    finite_reference = np.asarray(reference, dtype=float)
    finite_reference = finite_reference[np.isfinite(finite_reference)]
    if finite_reference.size == 0:
        raise ValueError("Rank reference is empty.")
    return upper_rank(observed, finite_reference)


def split_country_banks(
    draws: pd.DataFrame,
    *,
    expected_split_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split archived draws into the registered even-pilot/odd-evaluation banks."""
    bootstrap_id = pd.to_numeric(draws["bootstrap_id"], errors="raise").astype(int)
    calibration = draws.loc[bootstrap_id.mod(2).eq(0)].copy()
    reference = draws.loc[bootstrap_id.mod(2).eq(1)].copy()
    if calibration.empty or reference.empty:
        raise ValueError("Cannot split fitted-null bank into nonempty pilot and evaluation sets.")
    if expected_split_size is not None and (
        len(calibration) != expected_split_size or len(reference) != expected_split_size
    ):
        raise ValueError(
            "Paper-facing country audit requires exactly "
            f"{expected_split_size} pilot and {expected_split_size} evaluation paths; "
            f"found {len(calibration)} and {len(reference)}."
        )
    return calibration, reference


def _bank_departures(
    summary: pd.DataFrame, calibration: pd.DataFrame, reference: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    """Construct components once; callers choose marginal or global ranking."""
    observed_departures: list[float] = []
    reference_departures: list[np.ndarray] = []
    standardizers: dict[str, tuple[float, float]] = {}
    for metric in COMPONENTS:
        metric_row = summary.loc[summary["metric"].eq(metric)]
        if len(metric_row) != 1:
            raise ValueError(f"Expected one summary row for {metric}; found {len(metric_row)}.")
        observed = float(metric_row["observed"].iloc[0])
        pilot_values = pd.to_numeric(calibration[metric], errors="coerce").to_numpy(dtype=float)
        center = float(np.nanmedian(pilot_values))
        scale = float(np.nanstd(pilot_values, ddof=1))
        standardizers[metric] = (center, scale)
        observed_departures.append(
            departure(np.array([observed]), center, scale, TAILS[metric])[0]
        )
        evaluation_values = pd.to_numeric(
            reference[metric], errors="coerce"
        ).to_numpy(dtype=float)
        reference_departures.append(
            departure(evaluation_values, center, scale, TAILS[metric])
        )

    observed_vector = np.asarray(observed_departures, dtype=float)
    reference_matrix = np.column_stack(reference_departures)
    return observed_vector, reference_matrix, standardizers


def build_single_component_profile(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    expected_split_size: int | None = None,
) -> dict[str, float | str]:
    """Return separately calibrated ranks for each component using the frozen split."""
    calibration, reference = split_country_banks(
        draws,
        expected_split_size=expected_split_size,
    )
    row: dict[str, float | str] = {
        "country": str(summary["val_country"].iloc[0]),
    }
    observed, evaluation, _ = _bank_departures(summary, calibration, reference)
    for index, metric in enumerate(COMPONENTS):
        row[metric] = upper_rank_pvalue(float(observed[index]), evaluation[:, index])
    return row


def build_country_rows(
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    expected_split_size: int | None = None,
) -> tuple[dict[str, object], dict[str, float]]:
    """Build the global-score row and its max-adjusted component profile.

    Even ``bootstrap_id`` values form the pilot bank used for component
    standardization; odd values form the evaluation bank used for ranks.
    ``expected_split_size`` is set by the paper-table builder to enforce the
    registered 250/250 design and left unset by the general runner.
    """
    calibration, reference = split_country_banks(
        draws,
        expected_split_size=expected_split_size,
    )
    return build_country_rows_from_banks(summary, calibration, reference)


def build_country_rows_from_banks(
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, float]]:
    """Build score rows from explicit pilot and evaluation banks.

    This entry point is used by post-hoc Monte Carlo-resolution checks whose
    evaluation bank is larger than the frozen pilot bank. It leaves the
    registered even/odd split in build_country_rows unchanged.
    """
    if calibration.empty or reference.empty:
        raise ValueError("Pilot and evaluation banks must both be nonempty.")
    observed_vector, reference_matrix, standardizers = _bank_departures(summary, calibration, reference)
    reference_matrix = reference_matrix[np.isfinite(reference_matrix).all(axis=1)]
    if reference_matrix.size == 0:
        raise ValueError("No complete evaluation rows remain after finite-value filtering.")
    reference_smax = np.max(reference_matrix, axis=1)
    observed_smax = float(np.max(observed_vector))
    dominant_index = int(np.argmax(observed_vector))

    global_row: dict[str, object] = {
        "case": str(summary["case"].iloc[0]),
        "val_country": str(summary["val_country"].iloc[0]),
        "score_rule": "split_calibrated_max_null_standardized_departure",
        "component_metrics": ",".join(COMPONENTS),
        "split_standardization_n": int(len(calibration)),
        "split_reference_n": int(len(reference_smax)),
        "observed_smax": observed_smax,
        "null_smax_mean": float(np.mean(reference_smax)),
        "null_smax_q025": float(np.quantile(reference_smax, 0.025)),
        "null_smax_q500": float(np.quantile(reference_smax, 0.500)),
        "null_smax_q975": float(np.quantile(reference_smax, 0.975)),
        "smax_empirical_pvalue": upper_rank_pvalue(observed_smax, reference_smax),
        "dominant_metric": COMPONENTS[dominant_index],
        "dominant_component_departure": float(observed_vector[dominant_index]),
    }
    for metric, value in zip(COMPONENTS, observed_vector):
        global_row[f"{metric}_departure"] = float(value)
        global_row[f"{metric}_null_median"] = standardizers[metric][0]
        global_row[f"{metric}_null_std"] = standardizers[metric][1]

    adjusted_row = {
        "country": str(summary["val_country"].iloc[0]),
        "global": float(global_row["smax_empirical_pvalue"]),
    }
    for metric, value in zip(COMPONENTS, observed_vector):
        adjusted_row[metric] = upper_rank_pvalue(float(value), reference_smax)
    return global_row, adjusted_row
