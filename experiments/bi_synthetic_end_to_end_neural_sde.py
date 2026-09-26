"""
End-to-end synthetic Neural SDE fitted-null audit.

This experiment is intentionally different from the legacy residual-level
calibration, which calibrates rank p-values when the residuals are already
known.  Here the whole fitted pipeline is
inside the Monte Carlo loop:

1. simulate synthetic paths from a finite-grid Brownian SDE with endogenous
   history controls;
2. train candidate neural transition/SDE models by Gaussian one-step NLL;
3. choose the network width using validation paths;
4. freeze the selected model and audit both a held-out path and one training
   path;
5. simulate fitted-null reference paths from the selected model, recomputing the
   same endogenous controls along every simulated path;
6. rank observed diagnostics against the fitted-null diagnostics.

The matched-null rows check post-training/model-selection calibration under a
synthetic fitted null.  The MA-innovation rows keep the same drift and marginal
innovation variance, but make innovations temporally correlated before training;
they are a power/localization check, not a calibrated null.  The scale-only
rows use iid innovations with a different marginal scale; because the fitted
transition model estimates a scalar sigma, this is mainly a negative-control
alternative for the end-to-end fit/select/audit loop rather than a guaranteed
power case.

Outputs:
    cache/summaries/bi_synthetic_end_to_end_trials.csv
    cache/summaries/bi_synthetic_end_to_end_uniformity.csv
    cache/summaries/bi_synthetic_end_to_end_power.csv
    cache/summaries/bi_synthetic_end_to_end_metadata.json
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
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.special import ndtr
from scipy.stats import kstest

from experiments.audit_statistics import upper_rank


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "cache" / "summaries"

SEED = 20260703
LOOKBACK = 7
PERIOD = 20
INPUT_DIM = 6
METRICS = (
    "z_mean_abs",
    "z_std_abs",
    "acf1_abs",
    "acf7_abs",
    "ks_stat",
    "lb10_stat",
    "global_smax",
)
COMPONENT_METRICS = METRICS[:-1]
ALPHAS = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class ExperimentConfig:
    null_trials: int = 12
    alt_trials: int = 12
    audit_paths: int = 3
    train_paths: int = 48
    selection_paths: int = 12
    steps: int = 112
    bootstrap_paths: int = 200
    epochs: int = 160
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    candidate_widths: tuple[int, ...] = (8, 16)
    ma_rho: float = 0.75
    scale_factor: float = 1.35


def _control_from_prefix(path: np.ndarray, k: int) -> np.ndarray:
    """Predictable controls available before transition k -> k+1."""
    left = max(0, k - LOOKBACK)
    hist = path[left : k + 1]
    if len(hist) > 1:
        increments = np.diff(hist)
        last_inc = float(increments[-1])
        mean_inc = float(np.mean(increments))
    else:
        last_inc = 0.0
        mean_inc = 0.0
    level_mean = float(np.mean(hist))
    phase = 2.0 * math.pi * (k % PERIOD) / PERIOD
    return np.array(
        [last_inc, mean_inc, level_mean, math.sin(phase), math.cos(phase)],
        dtype=np.float32,
    )


def _input_from_path(path: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate(([float(path[k])], _control_from_prefix(path, k))).astype(np.float32)


def _true_drift(x: float, controls: np.ndarray) -> float:
    last_inc, mean_inc, level_mean, sin_phase, cos_phase = controls
    return float(
        -0.22 * x
        + 0.12 * np.tanh(1.5 * x)
        + 0.18 * mean_inc
        + 0.04 * level_mean
        + 0.06 * sin_phase
        - 0.025 * cos_phase
        + 0.02 * last_inc
    )


def _generate_noise(
    rng: np.random.Generator,
    n_paths: int,
    steps: int,
    scenario: str,
    ma_rho: float,
    scale_factor: float,
) -> np.ndarray:
    if scenario == "matched_null":
        return rng.normal(size=(n_paths, steps)).astype(np.float32)
    if scenario == "ma_innovation":
        raw = rng.normal(size=(n_paths, steps + 1)).astype(np.float32)
        return ((raw[:, 1:] + ma_rho * raw[:, :-1]) / math.sqrt(1.0 + ma_rho**2)).astype(
            np.float32
        )
    if scenario == "scale_innovation":
        return (rng.normal(size=(n_paths, steps)) * scale_factor).astype(np.float32)
    raise ValueError(f"Unknown scenario: {scenario}")


def simulate_synthetic_paths(
    rng: np.random.Generator,
    n_paths: int,
    steps: int,
    scenario: str,
    ma_rho: float,
    scale_factor: float,
) -> np.ndarray:
    """Simulate one-dimensional finite-grid SDE paths."""
    paths = np.empty((n_paths, steps + 1), dtype=np.float32)
    paths[:, 0] = rng.normal(loc=0.0, scale=0.7, size=n_paths).astype(np.float32)
    noise = _generate_noise(rng, n_paths, steps, scenario, ma_rho, scale_factor)
    sigma = 0.42
    for k in range(steps):
        for i in range(n_paths):
            controls = _control_from_prefix(paths[i], k)
            drift = _true_drift(float(paths[i, k]), controls)
            paths[i, k + 1] = paths[i, k] + drift + sigma * noise[i, k]
    return paths


def build_transition_dataset(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    dx_rows: list[float] = []
    for path in paths:
        for k in range(LOOKBACK, len(path) - 1):
            x_rows.append(_input_from_path(path, k))
            dx_rows.append(float(path[k + 1] - path[k]))
    return np.stack(x_rows).astype(np.float32), np.asarray(dx_rows, dtype=np.float32)


def init_params(width: int, key: jax.Array) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (INPUT_DIM, width)) * 0.15,
        "b1": jnp.zeros((width,)),
        "w2": jax.random.normal(k2, (width,)) * 0.10,
        "b2": jnp.zeros(()),
        "log_sigma": jnp.array(math.log(0.45), dtype=jnp.float32),
    }


def model_mean(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    hidden = jnp.tanh(x @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def model_sigma(params: dict[str, jax.Array]) -> jax.Array:
    return jnp.exp(params["log_sigma"]) + 1e-5


def l2_penalty(params: dict[str, jax.Array]) -> jax.Array:
    return jnp.sum(params["w1"] ** 2) + jnp.sum(params["w2"] ** 2)


def nll_loss(
    params: dict[str, jax.Array],
    x: jax.Array,
    dx: jax.Array,
    weight_decay: float,
) -> jax.Array:
    mu = model_mean(params, x)
    sigma = model_sigma(params)
    z = (dx - mu) / sigma
    nll = 0.5 * z**2 + jnp.log(sigma) + 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.mean(nll) + weight_decay * l2_penalty(params)


# Keep Adam's state transformation independent of the step size so the
# ExperimentConfig.learning_rate field actually controls optimization.
OPTIMIZER = optax.scale_by_adam()


@jax.jit
def train_step(
    params: dict[str, jax.Array],
    opt_state: optax.OptState,
    x: jax.Array,
    dx: jax.Array,
    weight_decay: float,
    learning_rate: float,
) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
    loss_value, grads = jax.value_and_grad(nll_loss)(params, x, dx, weight_decay)
    updates, opt_state = OPTIMIZER.update(grads, opt_state, params)
    updates = jax.tree_util.tree_map(lambda update: -learning_rate * update, updates)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


@jax.jit
def train_epochs(
    params: dict[str, jax.Array],
    opt_state: optax.OptState,
    x: jax.Array,
    dx: jax.Array,
    weight_decay: float,
    learning_rate: float,
    epochs: int,
) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
    def body(_idx, carry):
        p, s, _last = carry
        p, s, loss_value = train_step(p, s, x, dx, weight_decay, learning_rate)
        return p, s, loss_value

    return jax.lax.fori_loop(0, epochs, body, (params, opt_state, jnp.array(float("nan"))))


@jax.jit
def eval_loss(params: dict[str, jax.Array], x: jax.Array, dx: jax.Array) -> jax.Array:
    return nll_loss(params, x, dx, 0.0)


def standardize(
    train_x: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0, ddof=1)
    scale = np.where(scale < 1e-6, 1.0, scale)
    transformed = [((arr - mean) / scale).astype(np.float32) for arr in (train_x, *others)]
    return mean.astype(np.float32), scale.astype(np.float32), transformed


def fit_candidate(
    train_x: np.ndarray,
    train_dx: np.ndarray,
    val_x: np.ndarray,
    val_dx: np.ndarray,
    width: int,
    seed: int,
    config: ExperimentConfig,
) -> dict[str, object]:
    mean, scale, arrays = standardize(train_x, val_x)
    train_x_s, val_x_s = arrays
    params = init_params(width, jax.random.PRNGKey(seed))
    opt_state = OPTIMIZER.init(params)
    train_x_j = jnp.asarray(train_x_s)
    train_dx_j = jnp.asarray(train_dx)
    val_x_j = jnp.asarray(val_x_s)
    val_dx_j = jnp.asarray(val_dx)

    params, opt_state, _loss = train_epochs(
        params,
        opt_state,
        train_x_j,
        train_dx_j,
        config.weight_decay,
        config.learning_rate,
        config.epochs,
    )

    return {
        "params": params,
        "input_mean": mean,
        "input_scale": scale,
        "width": width,
        "train_nll": float(eval_loss(params, train_x_j, train_dx_j)),
        "validation_nll": float(eval_loss(params, val_x_j, val_dx_j)),
        "sigma": float(model_sigma(params)),
    }


def select_model(
    train_paths: np.ndarray,
    selection_paths: np.ndarray,
    seed: int,
    config: ExperimentConfig,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    train_x, train_dx = build_transition_dataset(train_paths)
    val_x, val_dx = build_transition_dataset(selection_paths)
    rows: list[dict[str, float]] = []
    fitted: list[dict[str, object]] = []
    for idx, width in enumerate(config.candidate_widths):
        fit = fit_candidate(
            train_x,
            train_dx,
            val_x,
            val_dx,
            width,
            seed + 1009 * (idx + 1),
            config,
        )
        rows.append(
            {
                "width": float(width),
                "train_nll": float(fit["train_nll"]),
                "validation_nll": float(fit["validation_nll"]),
                "sigma": float(fit["sigma"]),
            }
        )
        fitted.append(fit)
    best = min(fitted, key=lambda item: float(item["validation_nll"]))
    return best, rows


def predict_increment(fit: dict[str, object], x_row: np.ndarray) -> tuple[float, float]:
    x = ((x_row.astype(np.float32) - fit["input_mean"]) / fit["input_scale"]).astype(np.float32)
    params = fit["params"]
    mu = float(model_mean(params, jnp.asarray(x)))
    sigma = float(model_sigma(params))
    return mu, sigma


def residuals_for_path(fit: dict[str, object], path: np.ndarray) -> np.ndarray:
    residuals: list[float] = []
    for k in range(LOOKBACK, len(path) - 1):
        x_row = _input_from_path(path, k)
        mu, sigma = predict_increment(fit, x_row)
        residuals.append((float(path[k + 1] - path[k]) - mu) / sigma)
    return np.asarray(residuals, dtype=np.float64)


def simulate_fitted_null_residuals(
    fit: dict[str, object],
    observed_path: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    horizon = len(observed_path) - LOOKBACK - 1
    out = np.empty((n_paths, horizon), dtype=np.float64)
    initial_history = observed_path[: LOOKBACK + 1].astype(np.float32)
    sim = np.empty((n_paths, len(observed_path)), dtype=np.float32)
    sim[:, : LOOKBACK + 1] = initial_history
    sigma = float(model_sigma(fit["params"]))
    mean = fit["input_mean"]
    scale = fit["input_scale"]
    params = fit["params"]
    for h in range(horizon):
        k = LOOKBACK + h
        x_rows = np.stack([_input_from_path(path, k) for path in sim]).astype(np.float32)
        x_rows = ((x_rows - mean) / scale).astype(np.float32)
        mu = np.asarray(model_mean(params, jnp.asarray(x_rows)), dtype=np.float32)
        eps = rng.normal(size=n_paths).astype(np.float32)
        sim[:, k + 1] = sim[:, k] + mu + sigma * eps
        out[:, h] = eps.astype(np.float64)
    return out


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 2:
        return float("nan")
    left = x[:-lag] - np.mean(x[:-lag])
    right = x[lag:] - np.mean(x[lag:])
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    if denom <= 0:
        return 0.0
    return float(np.sum(left * right) / denom)


def _ks_stat_normal(z: np.ndarray) -> float:
    z_sorted = np.sort(z)
    n = len(z_sorted)
    cdf = ndtr(z_sorted)
    right = np.arange(1, n + 1, dtype=float) / n
    left = np.arange(0, n, dtype=float) / n
    return float(max(np.max(right - cdf), np.max(cdf - left)))


def _ljung_box(z: np.ndarray, lag: int = 10) -> float:
    n = len(z)
    q = 0.0
    for k in range(1, min(lag, n - 2) + 1):
        rho = _acf(z, k)
        q += rho * rho / max(n - k, 1)
    return float(n * (n + 2) * q)


def diagnostic_components(z: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=np.float64)
    return {
        "z_mean_abs": float(abs(np.mean(z))),
        "z_std_abs": float(abs(np.std(z, ddof=1) - 1.0)),
        "acf1_abs": float(abs(_acf(z, 1))),
        "acf7_abs": float(abs(_acf(z, 7))),
        "ks_stat": _ks_stat_normal(z),
        "lb10_stat": _ljung_box(z, 10),
    }


def diagnostics_matrix(residuals: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([diagnostic_components(row) for row in residuals])


def audit_path(
    fit: dict[str, object],
    path: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[dict[str, float], dict[str, float]]:
    obs_residuals = residuals_for_path(fit, path)
    obs_stats = diagnostic_components(obs_residuals)
    null_residuals = simulate_fitted_null_residuals(fit, path, n_bootstrap, rng)
    null_stats = diagnostics_matrix(null_residuals)

    pvalues: dict[str, float] = {}
    for metric in COMPONENT_METRICS:
        values = null_stats[metric].to_numpy(dtype=float)
        pvalues[f"p_{metric}"] = upper_rank(obs_stats[metric], values)

    means = null_stats[list(COMPONENT_METRICS)].mean(axis=0).to_numpy(dtype=float)
    stds = null_stats[list(COMPONENT_METRICS)].std(axis=0, ddof=1).to_numpy(dtype=float)
    stds = np.where(stds < 1e-8, 1.0, stds)
    null_smax = np.max((null_stats[list(COMPONENT_METRICS)].to_numpy(dtype=float) - means) / stds, axis=1)
    obs_smax = float(np.max((np.array([obs_stats[m] for m in COMPONENT_METRICS]) - means) / stds))
    obs_stats["global_smax"] = obs_smax
    pvalues["p_global_smax"] = upper_rank(obs_smax, null_smax)
    return obs_stats, pvalues


def run_one_trial(
    scenario: str,
    trial: int,
    seed: int,
    config: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    train_paths = simulate_synthetic_paths(
        rng,
        config.train_paths,
        config.steps,
        scenario,
        config.ma_rho,
        config.scale_factor,
    )
    selection_paths = simulate_synthetic_paths(
        rng,
        config.selection_paths,
        config.steps,
        scenario,
        config.ma_rho,
        config.scale_factor,
    )
    audit_paths = simulate_synthetic_paths(
        rng,
        config.audit_paths,
        config.steps,
        scenario,
        config.ma_rho,
        config.scale_factor,
    )
    fit, candidates = select_model(train_paths, selection_paths, seed, config)

    rows: list[dict[str, object]] = []
    for split, audit_id, path in [("train_reuse", 0, train_paths[0]), *[
        ("heldout", idx, audit_paths[idx]) for idx in range(config.audit_paths)
    ]]:
        stats, pvalues = audit_path(fit, path, rng, config.bootstrap_paths)
        row: dict[str, object] = {
            "scenario": scenario,
            "trial": trial,
            "audit_split": split,
            "audit_id": audit_id,
            "selected_width": int(fit["width"]),
            "selected_train_nll": float(fit["train_nll"]),
            "selected_validation_nll": float(fit["validation_nll"]),
            "selected_sigma": float(fit["sigma"]),
        }
        row.update({f"obs_{key}": value for key, value in stats.items()})
        row.update(pvalues)
        rows.append(row)

    candidate_rows = []
    for candidate in candidates:
        candidate_rows.append(
            {
                "scenario": scenario,
                "trial": float(trial),
                **candidate,
                "selected": float(candidate["width"] == fit["width"]),
            }
        )
    return rows, candidate_rows


def summarize_pvalues(trials: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = trials[trials["scenario"] == scenario]
    for audit_split, group in subset.groupby("audit_split"):
        for metric in METRICS:
            values = group[f"p_{metric}"].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            row = {
                "scenario": scenario,
                "audit_split": audit_split,
                "metric": metric,
                "n_pvalues": int(len(values)),
                "p_mean": float(np.mean(values)),
                "p_median": float(np.median(values)),
                "p_q05": float(np.quantile(values, 0.05)),
                "p_q95": float(np.quantile(values, 0.95)),
            }
            if scenario == "matched_null":
                ks_stat, ks_p = kstest(values, "uniform")
                row["uniform_ks_stat"] = float(ks_stat)
                row["uniform_ks_pvalue"] = float(ks_p)
            else:
                row["uniform_ks_stat"] = float("nan")
                row["uniform_ks_pvalue"] = float("nan")
            for alpha in ALPHAS:
                row[f"reject_rate_alpha_{alpha:.2f}"] = float(np.mean(values <= alpha))
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-trials", type=int, default=ExperimentConfig.null_trials)
    parser.add_argument("--alt-trials", type=int, default=ExperimentConfig.alt_trials)
    parser.add_argument("--audit-paths", type=int, default=ExperimentConfig.audit_paths)
    parser.add_argument("--bootstrap-paths", type=int, default=ExperimentConfig.bootstrap_paths)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--scale-factor", type=float, default=ExperimentConfig.scale_factor)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        null_trials=args.null_trials,
        alt_trials=args.alt_trials,
        audit_paths=args.audit_paths,
        bootstrap_paths=args.bootstrap_paths,
        epochs=args.epochs,
        scale_factor=args.scale_factor,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    all_candidates: list[dict[str, float]] = []
    scenarios = [
        ("matched_null", config.null_trials),
        ("ma_innovation", config.alt_trials),
        ("scale_innovation", config.alt_trials),
    ]
    for scenario_idx, (scenario, n_trials) in enumerate(scenarios):
        for trial in range(n_trials):
            trial_seed = int(args.seed + 10_000 * scenario_idx + trial)
            print(f"Running {scenario} trial {trial + 1}/{n_trials} (seed={trial_seed})", flush=True)
            rows, candidates = run_one_trial(scenario, trial, trial_seed, config)
            all_rows.extend(rows)
            all_candidates.extend(candidates)

    trials = pd.DataFrame(all_rows)
    candidate_df = pd.DataFrame(all_candidates)
    uniformity = summarize_pvalues(trials, "matched_null")
    power = pd.concat(
        [
            summarize_pvalues(trials, "ma_innovation"),
            summarize_pvalues(trials, "scale_innovation"),
        ],
        ignore_index=True,
    )

    trials_path = args.out_dir / "bi_synthetic_end_to_end_trials.csv"
    candidate_path = args.out_dir / "bi_synthetic_end_to_end_model_selection.csv"
    uniformity_path = args.out_dir / "bi_synthetic_end_to_end_uniformity.csv"
    power_path = args.out_dir / "bi_synthetic_end_to_end_power.csv"
    metadata_path = args.out_dir / "bi_synthetic_end_to_end_metadata.json"

    trials.to_csv(trials_path, index=False, float_format="%.6f")
    candidate_df.to_csv(candidate_path, index=False, float_format="%.6f")
    uniformity.to_csv(uniformity_path, index=False, float_format="%.6f")
    power.to_csv(power_path, index=False, float_format="%.6f")
    metadata = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "argv": sys.argv,
        "seed": int(args.seed),
        "config": asdict(config),
        "metrics": METRICS,
        "component_metrics_for_global_smax": COMPONENT_METRICS,
        "python": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "outputs": {
            "trials": str(trials_path.relative_to(ROOT)),
            "model_selection": str(candidate_path.relative_to(ROOT)),
            "uniformity": str(uniformity_path.relative_to(ROOT)),
            "power": str(power_path.relative_to(ROOT)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nMatched-null p-value uniformity")
    print(uniformity.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAlternative power/localization")
    print(power.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved {trials_path.relative_to(ROOT)}")
    print(f"Saved {candidate_path.relative_to(ROOT)}")
    print(f"Saved {uniformity_path.relative_to(ROOT)}")
    print(f"Saved {power_path.relative_to(ROOT)}")
    print(f"Saved {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
