"""Matched-size incremental-power benchmark for BI restriction diagnostics.

Each outer replicate simulates paths, refits/selects the neural transition
model, freezes it, computes one held-out audit path, and ranks every statistic
against an independent fitted-null evaluation bank.  A separate pilot bank
standardizes the global max scores.

The key ``paired_sign`` alternative has exact N(0,1) innovation marginals and
zero linear autocorrelation in expectation, but adjacent magnitudes repeat:

    (Z_2k, Z_2k+1) = (E_k, S_k E_k),  S_k in {-1,+1} independently.

It is therefore a direct test of incremental value beyond ACF/Ljung-Box/KS.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import matplotlib
import numpy as np
import pandas as pd
from scipy.special import ndtr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


from experiments.audit_statistics import upper_rank, wilson_interval as _wilson
from experiments.bi_synthetic_end_to_end_neural_sde import (
    ExperimentConfig,
    LOOKBACK,
    _control_from_prefix,
    _true_drift,
    residuals_for_path,
    select_model,
    simulate_fitted_null_residuals,
)


OUT_DIR = ROOT / "cache" / "summaries" / "synthetic_incremental_power_split_r200"
DEFAULT_SCENARIOS = (
    "matched_null",
    "ma_innovation",
    "paired_sign",
    "jump_innovation",
)
SEED = 20260716

COMPONENTS = (
    "z_mean_abs",
    "z_std_abs",
    "acf1_abs",
    "acf7_abs",
    "ks_stat",
    "lb10_stat",
    "max_w_norm",
    "excess_kurtosis_abs",
    "acf1_squared_abs",
    "lag1_rank_copula",
    "lag1_star_cs",
)
CLASSIC_COMPONENTS = (
    "z_mean_abs",
    "z_std_abs",
    "acf1_abs",
    "acf7_abs",
    "ks_stat",
    "lb10_stat",
    "max_w_norm",
)
EXTENDED_COMPONENTS = CLASSIC_COMPONENTS + (
    "excess_kurtosis_abs",
    "acf1_squared_abs",
    "lag1_rank_copula",
)

METHOD_COLUMNS = {
    "Scale only": "p_z_std_abs",
    "ACF(1)": "p_acf1_abs",
    "Ljung-Box(10)": "p_lb10_stat",
    "KS": "p_ks_stat",
    "Cumulative max": "p_max_w_norm",
    "Excess kurtosis": "p_excess_kurtosis_abs",
    "Squared ACF(1)": "p_acf1_squared_abs",
    "Rank-copula lag-1": "p_lag1_rank_copula",
    "Lag-pair joint CDF": "p_lag1_star_cs",
    "Classic global": "p_global_classic",
    "Extended global": "p_global_extended",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    null_replicates: int = 200
    alt_replicates: int = 200
    train_paths: int = 32
    selection_paths: int = 8
    steps: int = 96
    pilot_paths: int = 250
    evaluation_paths: int = 499
    epochs: int = 100
    candidate_widths: tuple[int, ...] = (8, 16)
    ma_rho: float = 0.75
    jump_probability: float = 0.025
    jump_size: float = 5.0
    geometry_grid_size: int = 9
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS


def _validate_config(config: BenchmarkConfig) -> None:
    if config.steps <= LOOKBACK + 2:
        raise ValueError(
            f"steps must exceed LOOKBACK + 2 ({LOOKBACK + 2}); got {config.steps}"
        )
    if min(config.train_paths, config.selection_paths, config.pilot_paths, config.evaluation_paths) < 1:
        raise ValueError("All path-bank sizes must be positive.")
    if config.geometry_grid_size < 2:
        raise ValueError("geometry_grid_size must be at least 2.")
    unknown = sorted(set(config.scenarios) - set(DEFAULT_SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")
    if len(set(config.scenarios)) != len(config.scenarios):
        raise ValueError("Scenario list contains duplicates.")


def _noise(
    rng: np.random.Generator,
    n_paths: int,
    steps: int,
    scenario: str,
    config: BenchmarkConfig,
) -> np.ndarray:
    if scenario == "matched_null":
        return rng.normal(size=(n_paths, steps)).astype(np.float32)
    if scenario == "ma_innovation":
        raw = rng.normal(size=(n_paths, steps + 1))
        return (
            (raw[:, 1:] + config.ma_rho * raw[:, :-1])
            / math.sqrt(1.0 + config.ma_rho**2)
        ).astype(np.float32)
    if scenario == "paired_sign":
        out = np.empty((n_paths, steps), dtype=np.float32)
        for path_idx in range(n_paths):
            position = 0
            if bool(rng.integers(0, 2)):
                out[path_idx, 0] = float(rng.normal())
                position = 1
            while position + 1 < steps:
                value = float(rng.normal())
                sign = float(rng.choice((-1.0, 1.0)))
                out[path_idx, position] = value
                out[path_idx, position + 1] = sign * value
                position += 2
            if position < steps:
                out[path_idx, position] = float(rng.normal())
        return out
    if scenario == "jump_innovation":
        gaussian = rng.normal(size=(n_paths, steps))
        jump_mask = rng.random(size=(n_paths, steps)) < config.jump_probability
        jump_sign = rng.choice((-1.0, 1.0), size=(n_paths, steps))
        raw = gaussian + jump_mask * jump_sign * config.jump_size
        variance = 1.0 + config.jump_probability * config.jump_size**2
        return (raw / math.sqrt(variance)).astype(np.float32)
    raise ValueError(f"Unknown scenario: {scenario}")


def _simulate_paths(
    rng: np.random.Generator,
    n_paths: int,
    scenario: str,
    config: BenchmarkConfig,
) -> np.ndarray:
    paths = np.empty((n_paths, config.steps + 1), dtype=np.float32)
    paths[:, 0] = rng.normal(0.0, 0.7, size=n_paths).astype(np.float32)
    noise = _noise(rng, n_paths, config.steps, scenario, config)
    sigma = 0.42
    for step in range(config.steps):
        for path_idx in range(n_paths):
            controls = _control_from_prefix(paths[path_idx], step)
            drift = _true_drift(float(paths[path_idx, step]), controls)
            paths[path_idx, step + 1] = (
                paths[path_idx, step] + drift + sigma * noise[path_idx, step]
            )
    return paths


def _acf(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= lag + 2:
        return float("nan")
    left = values[:-lag] - np.mean(values[:-lag])
    right = values[lag:] - np.mean(values[lag:])
    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    return float(np.sum(left * right) / denom) if denom > 1e-12 else 0.0


def _ks_stat(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    n = ordered.size
    cdf = ndtr(ordered)
    right = np.arange(1, n + 1, dtype=float) / n
    left = np.arange(0, n, dtype=float) / n
    return float(max(np.max(right - cdf), np.max(cdf - left)))


def _ljung_box(values: np.ndarray, lag: int = 10) -> float:
    n = len(values)
    total = 0.0
    for current_lag in range(1, min(lag, n - 2) + 1):
        rho = _acf(values, current_lag)
        total += rho**2 / max(n - current_lag, 1)
    return float(n * (n + 2) * total)


def _lag1_star_center_scaled(values: np.ndarray, grid_size: int) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    centered /= max(float(np.std(centered, ddof=1)), 1e-12)
    uniforms = ndtr(centered)
    left, right = uniforms[:-1], uniforms[1:]
    grid = np.arange(1, grid_size + 1, dtype=float) / (grid_size + 1.0)
    worst = 0.0
    for x_bound in grid:
        left_mask = left <= x_bound
        for y_bound in grid:
            empirical = float(np.mean(left_mask & (right <= y_bound)))
            worst = max(worst, abs(empirical - x_bound * y_bound))
    return float(worst)


def _lag1_rank_copula(values: np.ndarray, grid_size: int) -> float:
    """Lag-one empirical-copula discrepancy after removing marginal shape."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    uniforms = ranks / (values.size + 1.0)
    left, right = uniforms[:-1], uniforms[1:]
    grid = np.arange(1, grid_size + 1, dtype=float) / (grid_size + 1.0)
    worst = 0.0
    for x_bound in grid:
        left_mask = left <= x_bound
        for y_bound in grid:
            empirical = float(np.mean(left_mask & (right <= y_bound)))
            worst = max(worst, abs(empirical - x_bound * y_bound))
    return float(worst)


def _diagnostics(values: np.ndarray, grid_size: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    centered /= max(float(np.std(values, ddof=1)), 1e-12)
    return {
        "z_mean_abs": float(abs(np.mean(values))),
        "z_std_abs": float(abs(np.std(values, ddof=1) - 1.0)),
        "acf1_abs": float(abs(_acf(values, 1))),
        "acf7_abs": float(abs(_acf(values, 7))),
        "ks_stat": _ks_stat(values),
        "lb10_stat": _ljung_box(values, 10),
        "max_w_norm": float(np.max(np.abs(np.cumsum(values))) / np.sqrt(len(values))),
        "excess_kurtosis_abs": float(abs(np.mean(centered**4) - 3.0)),
        "acf1_squared_abs": float(abs(_acf(values**2, 1))),
        "lag1_rank_copula": _lag1_rank_copula(values, grid_size),
        "lag1_star_cs": _lag1_star_center_scaled(values, grid_size),
        "acf1_signed": float(_acf(values, 1)),
        "acf1_squared": float(_acf(values**2, 1)),
    }


def _diagnostic_frame(paths: np.ndarray, grid_size: int) -> pd.DataFrame:
    values = np.asarray(paths, dtype=float)
    n_paths, n_steps = values.shape
    means = np.mean(values, axis=1)
    stds = np.std(values, axis=1, ddof=1)

    def row_acf(array: np.ndarray, lag: int) -> np.ndarray:
        left = array[:, :-lag]
        right = array[:, lag:]
        left = left - np.mean(left, axis=1, keepdims=True)
        right = right - np.mean(right, axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
        return np.divide(
            np.sum(left * right, axis=1),
            denominator,
            out=np.zeros(n_paths, dtype=float),
            where=denominator > 1e-12,
        )

    acf1 = row_acf(values, 1)
    acf7 = row_acf(values, 7)
    squared_acf1 = row_acf(values**2, 1)
    lb = np.zeros(n_paths, dtype=float)
    for lag in range(1, min(10, n_steps - 2) + 1):
        rho = row_acf(values, lag)
        lb += rho**2 / max(n_steps - lag, 1)
    lb *= n_steps * (n_steps + 2)

    ordered = np.sort(values, axis=1)
    cdf = ndtr(ordered)
    right_ranks = np.arange(1, n_steps + 1, dtype=float) / n_steps
    left_ranks = np.arange(0, n_steps, dtype=float) / n_steps
    ks = np.maximum(
        np.max(right_ranks[None, :] - cdf, axis=1),
        np.max(cdf - left_ranks[None, :], axis=1),
    )

    centered = (values - means[:, None]) / np.maximum(stds[:, None], 1e-12)
    uniforms = ndtr(centered)
    grid = np.arange(1, grid_size + 1, dtype=float) / (grid_size + 1.0)
    left_indicators = uniforms[:, :-1, None] <= grid[None, None, :]
    right_indicators = uniforms[:, 1:, None] <= grid[None, None, :]
    joint = np.einsum(
        "nti,ntj->nij",
        left_indicators.astype(np.float64),
        right_indicators.astype(np.float64),
        optimize=True,
    ) / float(n_steps - 1)
    target = grid[:, None] * grid[None, :]
    geometry = np.max(np.abs(joint - target[None, :, :]), axis=(1, 2))

    ordinal_ranks = np.argsort(np.argsort(values, axis=1), axis=1) + 1
    rank_uniforms = ordinal_ranks / float(n_steps + 1)
    rank_left = rank_uniforms[:, :-1, None] <= grid[None, None, :]
    rank_right = rank_uniforms[:, 1:, None] <= grid[None, None, :]
    rank_joint = np.einsum(
        "nti,ntj->nij",
        rank_left.astype(np.float64),
        rank_right.astype(np.float64),
        optimize=True,
    ) / float(n_steps - 1)
    rank_copula = np.max(np.abs(rank_joint - target[None, :, :]), axis=(1, 2))

    return pd.DataFrame(
        {
            "z_mean_abs": np.abs(means),
            "z_std_abs": np.abs(stds - 1.0),
            "acf1_abs": np.abs(acf1),
            "acf7_abs": np.abs(acf7),
            "ks_stat": ks,
            "lb10_stat": lb,
            "max_w_norm": np.max(np.abs(np.cumsum(values, axis=1)), axis=1)
            / np.sqrt(n_steps),
            "excess_kurtosis_abs": np.abs(np.mean(centered**4, axis=1) - 3.0),
            "acf1_squared_abs": np.abs(squared_acf1),
            "lag1_rank_copula": rank_copula,
            "lag1_star_cs": geometry,
            "acf1_signed": acf1,
            "acf1_squared": squared_acf1,
        }
    )


def _upper_rank(observed: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    return upper_rank(observed, reference)


def _global_scores(
    observed: dict[str, float],
    pilot: pd.DataFrame,
    evaluation: pd.DataFrame,
    components: tuple[str, ...],
) -> tuple[float, np.ndarray]:
    center = pilot[list(components)].median(axis=0).to_numpy(dtype=float)
    scale = pilot[list(components)].std(axis=0, ddof=1).to_numpy(dtype=float)
    scale = np.where(scale < 1e-10, 1.0, scale)
    observed_vector = np.asarray([observed[key] for key in components], dtype=float)
    evaluation_matrix = evaluation[list(components)].to_numpy(dtype=float)
    observed_score = float(np.max(np.maximum((observed_vector - center) / scale, 0.0)))
    evaluation_scores = np.max(
        np.maximum((evaluation_matrix - center) / scale, 0.0), axis=1
    )
    return observed_score, evaluation_scores


def _inner_config(config: BenchmarkConfig) -> ExperimentConfig:
    return ExperimentConfig(
        train_paths=config.train_paths,
        selection_paths=config.selection_paths,
        steps=config.steps,
        bootstrap_paths=config.evaluation_paths,
        epochs=config.epochs,
        candidate_widths=config.candidate_widths,
        ma_rho=config.ma_rho,
    )


def _one_replicate(
    scenario: str,
    replicate: int,
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    train = _simulate_paths(rng, config.train_paths, scenario, config)
    selection = _simulate_paths(rng, config.selection_paths, scenario, config)
    audit_path = _simulate_paths(rng, 1, scenario, config)[0]
    fit, candidates = select_model(train, selection, seed, _inner_config(config))

    observed_residuals = residuals_for_path(fit, audit_path)
    pilot_residuals = simulate_fitted_null_residuals(
        fit, audit_path, config.pilot_paths, rng
    )
    evaluation_residuals = simulate_fitted_null_residuals(
        fit, audit_path, config.evaluation_paths, rng
    )
    observed = _diagnostics(observed_residuals, config.geometry_grid_size)
    pilot = _diagnostic_frame(pilot_residuals, config.geometry_grid_size)
    evaluation = _diagnostic_frame(evaluation_residuals, config.geometry_grid_size)

    row: dict[str, object] = {
        "scenario": scenario,
        "replicate": replicate,
        "outer_seed": seed,
        "selected_width": int(fit["width"]),
        "selected_sigma": float(fit["sigma"]),
        "selected_validation_nll": float(fit["validation_nll"]),
        "n_candidate_models": len(candidates),
        **{f"obs_{key}": value for key, value in observed.items()},
    }
    for component in COMPONENTS:
        row[f"p_{component}"] = _upper_rank(
            observed[component], evaluation[component].to_numpy(dtype=float)
        )

    for label, components in (
        ("classic", CLASSIC_COMPONENTS),
        ("extended", EXTENDED_COMPONENTS),
    ):
        observed_score, evaluation_scores = _global_scores(
            observed, pilot, evaluation, components
        )
        row[f"obs_global_{label}"] = observed_score
        row[f"p_global_{label}"] = _upper_rank(
            observed_score, evaluation_scores
        )
    return row


def _add_full_refit_pvalues(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()
    null_mask = trials["scenario"].eq("matched_null")
    null_indices = trials.index[null_mask].to_numpy()
    if len(null_indices) < 2:
        raise ValueError(
            "Full-refit calibration requires at least two matched_null replicates. "
            "Run the complete default scenario set to obtain the reported table."
        )
    for _method, column in METHOD_COLUMNS.items():
        if column not in trials.columns:
            raise ValueError(f"Missing internal p-value column: {column}")
        if not np.all(np.isfinite(trials[column].to_numpy(dtype=float))):
            raise ValueError(f"Non-finite internal p-values in column: {column}")
        output = f"full_refit_{column}"
        null_values = trials.loc[null_mask, column].to_numpy(dtype=float)
        for position, row_index in enumerate(null_indices):
            others = np.delete(null_values, position)
            trials.loc[row_index, output] = (
                1.0 + np.sum(others <= null_values[position])
            ) / float(len(null_values))
        for scenario in trials.loc[~null_mask, "scenario"].unique():
            scenario_mask = trials["scenario"].eq(scenario)
            values = trials.loc[scenario_mask, column].to_numpy(dtype=float)
            trials.loc[scenario_mask, output] = [
                (1.0 + np.sum(null_values <= value)) / float(len(null_values) + 1)
                for value in values
            ]
    return trials


def _summary(
    trials: pd.DataFrame,
    alpha: float = 0.05,
    full_refit_calibrated: bool = True,
) -> pd.DataFrame:
    rows = []
    for scenario, group in trials.groupby("scenario", sort=False):
        for method, column in METHOD_COLUMNS.items():
            if full_refit_calibrated:
                column = f"full_refit_{column}"
            pvalues = group[column].to_numpy(dtype=float)
            rejects = int(np.sum(pvalues <= alpha))
            low, high = _wilson(rejects, len(pvalues))
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "pvalue_column": column,
                    "n_replicates": int(len(pvalues)),
                    "alpha": alpha,
                    "full_refit_calibrated": full_refit_calibrated,
                    "reject_count": rejects,
                    "rejection_rate": rejects / len(pvalues),
                    "wilson95_low": low,
                    "wilson95_high": high,
                    "p_median": float(np.median(pvalues)),
                    "mean_observed_acf1": float(group["obs_acf1_signed"].mean()),
                    "mean_observed_squared_acf1": float(
                        group["obs_acf1_squared"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.pivot(index="scenario", columns="method", values="rejection_rate")
    return table.reindex(index=[s for s in DEFAULT_SCENARIOS if s in table.index]).reset_index()


def _plot(summary: pd.DataFrame, path: Path) -> None:
    scenarios = [s for s in DEFAULT_SCENARIOS if s in set(summary["scenario"])]
    methods = list(METHOD_COLUMNS)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 4.3), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for axis, scenario in zip(axes, scenarios):
        subset = summary.loc[summary["scenario"].eq(scenario)].set_index("method")
        values = subset.loc[methods, "rejection_rate"].to_numpy(float)
        low = subset.loc[methods, "wilson95_low"].to_numpy(float)
        high = subset.loc[methods, "wilson95_high"].to_numpy(float)
        positions = np.arange(len(methods))
        colors = ["#777777"] * len(methods)
        axis.bar(positions, values, color=colors)
        axis.errorbar(
            positions,
            values,
            yerr=np.maximum(np.vstack([values - low, high - values]), 0.0),
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=0.8,
        )
        axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
        axis.set_title(scenario.replace("_", " "))
        axis.set_xticks(positions)
        axis.set_xticklabels(methods, rotation=55, ha="right", fontsize=8)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Rejection rate at alpha=0.05")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config: BenchmarkConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_config(config)
    rows = []
    for scenario_idx, scenario in enumerate(config.scenarios):
        count = config.null_replicates if scenario == "matched_null" else config.alt_replicates
        for replicate in range(count):
            outer_seed = seed + 100_000 * scenario_idx + replicate
            print(
                f"incremental-power {scenario} {replicate + 1}/{count} seed={outer_seed}",
                flush=True,
            )
            rows.append(_one_replicate(scenario, replicate, outer_seed, config))
    trials = pd.DataFrame(rows)
    if int(trials["scenario"].eq("matched_null").sum()) >= 2:
        trials = _add_full_refit_pvalues(trials)
        summary = _summary(trials, full_refit_calibrated=True)
    else:
        # A partial scenario run can report internal fitted-null ranks, but the
        # reported table uses the complete default scenario set above.
        summary = _summary(trials, full_refit_calibrated=False)
    return trials, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-replicates", type=int, default=BenchmarkConfig.null_replicates)
    parser.add_argument("--alt-replicates", type=int, default=BenchmarkConfig.alt_replicates)
    parser.add_argument("--train-paths", type=int, default=BenchmarkConfig.train_paths)
    parser.add_argument("--selection-paths", type=int, default=BenchmarkConfig.selection_paths)
    parser.add_argument("--steps", type=int, default=BenchmarkConfig.steps)
    parser.add_argument("--pilot-paths", type=int, default=BenchmarkConfig.pilot_paths)
    parser.add_argument("--evaluation-paths", type=int, default=BenchmarkConfig.evaluation_paths)
    parser.add_argument("--epochs", type=int, default=BenchmarkConfig.epochs)
    parser.add_argument("--geometry-grid-size", type=int, default=BenchmarkConfig.geometry_grid_size)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = BenchmarkConfig(
        null_replicates=args.null_replicates,
        alt_replicates=args.alt_replicates,
        train_paths=args.train_paths,
        selection_paths=args.selection_paths,
        steps=args.steps,
        pilot_paths=args.pilot_paths,
        evaluation_paths=args.evaluation_paths,
        epochs=args.epochs,
        geometry_grid_size=args.geometry_grid_size,
        scenarios=tuple(part.strip() for part in args.scenarios.split(",") if part.strip()),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials, summary = run(config, args.seed)
    paper_table = _paper_table(summary)

    trials_path = args.out_dir / "synthetic_incremental_power_trials.csv"
    summary_path = args.out_dir / "synthetic_incremental_power_summary.csv"
    table_path = args.out_dir / "synthetic_incremental_power_paper_table.csv"
    plot_path = args.out_dir / "synthetic_incremental_power.png"
    metadata_path = args.out_dir / "synthetic_incremental_power_metadata.json"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    paper_table.to_csv(table_path, index=False)
    _plot(summary, plot_path)
    metadata = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "argv": sys.argv,
        "seed": args.seed,
        "config": asdict(config),
        "lookback": LOOKBACK,
        "diagnostic_residual_steps": config.steps - LOOKBACK,
        "methods": METHOD_COLUMNS,
        "classic_global_components": CLASSIC_COMPONENTS,
        "extended_global_components": EXTENDED_COMPONENTS,
        "paired_sign_definition": "(E_k, S_k E_k), E_k~N(0,1), S_k independent Rademacher",
        "pvalue_definition": "independent fitted-null evaluation-bank upper rank",
        "global_standardization": "independent pilot-bank median and standard deviation",
        "outer_full_refit_calibration_applied": bool(
            summary["full_refit_calibrated"].all()
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n", paper_table.to_string(index=False))
    print("\nMean signed ACF diagnostics")
    print(
        summary.groupby("scenario", sort=False)[
            ["mean_observed_acf1", "mean_observed_squared_acf1"]
        ].first().to_string()
    )


if __name__ == "__main__":
    main()
