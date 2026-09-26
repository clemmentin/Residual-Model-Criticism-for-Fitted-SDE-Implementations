"""Full-refit calibration statistics, independent of model fitting."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest

from experiments.audit_statistics import wilson_interval as _wilson_interval

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    ROOT
    / "cache"
    / "summaries"
    / "full_refit_internal_pvalue_calibration_r5000_p500_e999_final"
)


DEFAULT_ALPHAS = (0.01, 0.025, 0.05, 0.10)


COMPONENT_PVALUE_ORDER = (
    ("z_std_abs", "z_std"),
    ("z_mean_abs", "z_mean"),
    ("acf1_abs", "ACF(1)"),
    ("acf7_abs", "ACF(7)"),
    ("lb10_stat", "LB(10)"),
    ("ks_stat", "KS"),
)


def _alpha_table(
    pvalues: np.ndarray,
    alphas: tuple[float, ...],
    discrete_grid_size: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = int(len(pvalues))
    for alpha in alphas:
        reject_count = int(np.sum(pvalues <= alpha))
        reject_rate = reject_count / n if n else float("nan")
        ci_low, ci_high = _wilson_interval(reject_count, n)
        discrete_nominal = (
            math.floor(float(discrete_grid_size) * float(alpha)) / float(discrete_grid_size)
            if discrete_grid_size
            else float("nan")
        )
        rows.append(
            {
                "alpha": float(alpha),
                "n_replicates": n,
                "reject_count": reject_count,
                "reject_rate": float(reject_rate),
                "wilson95_low": float(ci_low),
                "wilson95_high": float(ci_high),
                "nominal_rate": float(alpha),
                "nominal_in_wilson95": bool(ci_low <= alpha <= ci_high) if n else False,
                "discrete_nominal_rate": float(discrete_nominal),
                "discrete_nominal_in_wilson95": (
                    bool(ci_low <= discrete_nominal <= ci_high)
                    if n and np.isfinite(discrete_nominal)
                    else False
                ),
            }
        )
    return pd.DataFrame(rows)


def _discrete_uniform_ks_stat(pvalues: np.ndarray, grid_size: int) -> float:
    pvalues = np.asarray(pvalues, dtype=float)
    if len(pvalues) == 0 or grid_size <= 0:
        return float("nan")
    if not np.isfinite(pvalues).all():
        raise ValueError("P-values must be finite.")
    ranks = np.rint(pvalues * float(grid_size)).astype(int)
    if (np.any(ranks < 1) or np.any(ranks > grid_size)
            or not np.allclose(pvalues, ranks / grid_size, rtol=0, atol=1e-12)):
        raise ValueError("P-values are not on the expected discrete rank grid.")
    counts = np.bincount(ranks, minlength=grid_size + 1)[1:]
    cumulative = np.cumsum(counts, dtype=np.int64)
    numerator = np.max(np.abs(grid_size * cumulative - len(pvalues) * np.arange(1, grid_size + 1)))
    return float(numerator / (len(pvalues) * grid_size))


def _discrete_uniform_ks_mc_pvalue(
    observed_stat: float,
    n_samples: int,
    grid_size: int,
    mc_reps: int,
    seed: int,
    batch_size: int = 10_000,
) -> float:
    if not np.isfinite(observed_stat) or n_samples <= 0 or grid_size <= 0 or mc_reps <= 0:
        return float("nan")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    rng = np.random.default_rng(seed)
    probs = np.full(grid_size, 1.0 / float(grid_size), dtype=float)
    # Both CDFs are rational. Compare their integer numerators so that equal
    # KS distances are counted as ties, regardless of floating-point subtraction.
    observed_numerator = int(round(observed_stat * n_samples * grid_size))
    reference_numerator = n_samples * np.arange(1, grid_size + 1, dtype=np.int64)
    n_extreme = 0
    n_done = 0
    while n_done < mc_reps:
        batch = min(batch_size, mc_reps - n_done)
        counts = rng.multinomial(n_samples, probs, size=batch)
        cumulative = np.cumsum(counts, axis=1, dtype=np.int64)
        numerators = np.max(np.abs(grid_size * cumulative - reference_numerator), axis=1)
        n_extreme += int(np.sum(numerators >= observed_numerator))
        n_done += batch
    return float((1 + n_extreme) / float(mc_reps + 1))


def _summary_table(
    pvalues: np.ndarray,
    discrete_grid_size: int,
    discrete_ks_mc_reps: int,
    discrete_ks_mc_seed: int,
    scenario: str,
) -> pd.DataFrame:
    pvalues = np.asarray(pvalues, dtype=float)
    if len(pvalues) == 0:
        return pd.DataFrame()
    continuous_ks_stat, continuous_ks_pvalue = kstest(pvalues, "uniform")
    discrete_ks_stat = _discrete_uniform_ks_stat(pvalues, discrete_grid_size)
    discrete_ks_mc_pvalue = _discrete_uniform_ks_mc_pvalue(
        discrete_ks_stat,
        n_samples=len(pvalues),
        grid_size=discrete_grid_size,
        mc_reps=discrete_ks_mc_reps,
        seed=discrete_ks_mc_seed,
    )
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "pvalue_definition": "internal_evaluation_bank_rank",
                "reference_distribution": f"discrete_uniform_rank_grid_{discrete_grid_size}",
                "n_replicates": int(len(pvalues)),
                "p_mean": float(np.mean(pvalues)),
                "p_median": float(np.median(pvalues)),
                "p_q01": float(np.quantile(pvalues, 0.01)),
                "p_q05": float(np.quantile(pvalues, 0.05)),
                "p_q95": float(np.quantile(pvalues, 0.95)),
                "p_q99": float(np.quantile(pvalues, 0.99)),
                "discrete_uniform_ks_stat": float(discrete_ks_stat),
                "discrete_uniform_ks_mc_pvalue": float(discrete_ks_mc_pvalue),
                "discrete_uniform_ks_mc_reps": int(discrete_ks_mc_reps),
                "continuous_uniform_ks_stat": float(continuous_ks_stat),
                "continuous_uniform_ks_pvalue": float(continuous_ks_pvalue),
            }
        ]
    )


def _ecdf_table(pvalues: np.ndarray) -> pd.DataFrame:
    p_sorted = np.sort(np.asarray(pvalues, dtype=float))
    n = int(len(p_sorted))
    ranks = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame(
        {
            "rank": ranks.astype(int),
            "pvalue": p_sorted,
            "ecdf": ranks / float(n) if n else [],
            "uniform_cdf": p_sorted,
            "ecdf_minus_uniform": ranks / float(n) - p_sorted if n else [],
            "qq_uniform_quantile": (ranks - 0.5) / float(n) if n else [],
        }
    )


def _component_pvalue_calibration_table(
    trials: pd.DataFrame,
    grid_size: int,
    mc_reps: int,
    seed: int,
    scenario: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = int(len(trials))
    discrete_nominal = math.floor(float(grid_size) * float(alpha)) / float(grid_size)
    for idx, (metric, label) in enumerate(COMPONENT_PVALUE_ORDER):
        column = f"p_{metric}_internal"
        if column not in trials.columns:
            continue
        pvalues = trials[column].to_numpy(dtype=float)
        dks_stat = _discrete_uniform_ks_stat(pvalues, grid_size)
        dks_pvalue = _discrete_uniform_ks_mc_pvalue(
            dks_stat,
            n_samples=n,
            grid_size=grid_size,
            mc_reps=mc_reps,
            seed=seed + 1009 * (idx + 1),
        )
        reject_count = int(np.sum(pvalues <= alpha))
        reject_rate = reject_count / n if n else float("nan")
        ci_low, ci_high = _wilson_interval(reject_count, n)
        rows.append(
            {
                "scenario": scenario,
                "component": label,
                "column": column,
                "n": n,
                "grid": int(grid_size),
                "dks_stat": float(dks_stat),
                "dks_mc_pvalue": float(dks_pvalue),
                "dks_mc_reps": int(mc_reps),
                "reject_count_05": reject_count,
                "rejection_rate_05": float(reject_rate),
                "size_05": float(reject_rate),
                "wilson95_low": float(ci_low),
                "wilson95_high": float(ci_high),
                "discrete_nominal_05": float(discrete_nominal),
                "discrete_nominal_in_wilson95": bool(ci_low <= discrete_nominal <= ci_high),
                "p_mean": float(np.mean(pvalues)),
                "p_median": float(np.median(pvalues)),
            }
        )
    return pd.DataFrame(rows)
