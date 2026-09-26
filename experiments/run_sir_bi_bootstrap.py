"""
Parametric-bootstrap calibration for SIR Brownian-inversion diagnostics.

For each fitted Neural SDE configuration:
1. Compute BI metrics on the held-out observed trajectory.
2. Simulate Brownian-consistent trajectories from the fitted model.
3. Re-invert every simulated trajectory with the same one-step BI procedure.
4. Report empirical null intervals, percentiles, and bootstrap p-values.

Run from the source-release root, for example:
    python experiments/run_sir_bi_bootstrap.py baseline_frzgamma --n-bootstrap 500
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import logging
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

import brownian_inversion
import config as config_module
import sir_data
import sir_training
from trajectory import JAXSignatureExtractor
from brownian_inversion import compute_brownian_inversion_metrics_from_residuals

from experiments._bi_audit_common import (
    apply_config_updates,
    empirical_pvalue,
    load_cached_data,
    summarize_metric,
)
from experiments._sir_model_loader import load_first_model


# Backward-compatible names for supporting diagnostics that historically
# imported these helpers from this runner.
_apply_updates = apply_config_updates
_load_cached_data = load_cached_data
_empirical_pvalue = empirical_pvalue


DEFAULT_OUT_DIR = ROOT / "cache" / "summaries" / "bi_bootstrap"

CASES = {
    "baseline": {"FREEZE_GAMMA": False},
    "baseline_frzgamma": {"FREEZE_GAMMA": True},
    "baseline_zwd0p1_frzgamma": {
        "FREEZE_GAMMA": True,
        "ZSCORE_WD_WEIGHT": 0.1,
    },
    "baseline_zwd1_frzgamma": {
        "FREEZE_GAMMA": True,
        "ZSCORE_WD_WEIGHT": 1.0,
    },
    "recov28": {
        "SIR_RECOVERY_DAYS": 28.0,
        "FREEZE_GAMMA": False,
    },
    "recov28_frzgamma": {
        "SIR_RECOVERY_DAYS": 28.0,
        "FREEZE_GAMMA": True,
    },
    "recov28_dyn14": {
        "SIR_RECOVERY_DAYS": 28.0,
        "SIR_RECOVERY_DAYS_DYN": 14.0,
        "FREEZE_GAMMA": False,
    },
}

DEFAULT_CASES = [
    "baseline_frzgamma",
    "recov28_frzgamma",
    "baseline_zwd1_frzgamma",
]

# Centered metrics flag unusually large deviations in either direction.
# Upper-tail metrics flag unusually large departures from the fitted null.
METRIC_TAILS = {
    "bi_z_mean": "centered",
    "bi_z_std": "centered",
    "bi_acf1": "centered",
    "bi_acf7": "centered",
    "bi_ks_stat": "upper",
    "bi_lb_stat": "upper",
    "bi_max_w_norm": "upper",
    "bi_z_kurt": "centered",
}

PLOT_METRICS = [
    "bi_z_std",
    "bi_acf1",
    "bi_acf7",
    "bi_lb_stat",
    "bi_ks_stat",
    "bi_max_w_norm",
]


def _load_model(cfg: config_module.Config, data: config_module.TrainingData):
    model, model_path = load_first_model(cfg, data)
    sir_training.assert_model_normalization(model, data)
    return model, model_path


def _start_position(data: config_module.TrainingData, cfg: config_module.Config) -> int:
    raw_lookback = sir_data.get_sir_control_lookback(cfg) + (
        1 if data.ar_phi is not None else 0
    )
    return max(
        len(data.val_features_df["I"].values) // 2,
        raw_lookback,
    )


def _validate_supported_config(cfg: config_module.Config) -> None:
    if str(cfg.CONTROL_ABLATION).lower() != "none":
        raise ValueError("Bootstrap BI currently supports CONTROL_ABLATION='none' only.")


def _simulate_null_residuals(
    model,
    data: config_module.TrainingData,
    cfg: config_module.Config,
    start_pos: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate fitted-null paths and BI-invert each one step by step."""
    _validate_supported_config(cfg)

    max_lookback = sir_data.get_sir_control_lookback(cfg)
    raw_lookback = max_lookback + (1 if data.ar_phi is not None else 0)
    horizon = min(cfg.FORECAST_HORIZON, len(data.val_features_df) - 1 - start_pos)
    if horizon < 10:
        raise ValueError("Not enough held-out observations for bootstrap BI.")

    # BI at day t uses a history window ending at day t.
    initial_history = jnp.asarray(
        data.val_features_df.iloc[start_pos - raw_lookback + 1 : start_pos + 1].values
    )
    signature_extractors = {
        length: JAXSignatureExtractor(
            depth=cfg.SIGNATURE_DEPTH,
            augment_time=True,
            lead_lag=cfg.SIGNATURE_LEAD_LAG,
        )
        for length in cfg.SIGNATURE_PATH_LENGTHS
    }
    dt = cfg.SDE_SUBSTEP_DT
    n_substeps = cfg.SDE_NUM_SUBSTEPS
    sqrt_dt = jnp.sqrt(dt)
    norm_mean = jnp.asarray(data.norm_mean)
    norm_std = jnp.asarray(data.norm_std)

    def _clip_physical(state):
        # Match forecast_sir exactly: cached models are simulated in the
        # component-wise z-score coordinates stored with the training data.
        phys = state * norm_std + norm_mean
        clipped = jnp.clip(phys, jnp.array([1e-6, -20.0]), jnp.array([1.0 - 1e-6, 0.0]))
        was_clipped = jnp.any(jnp.abs(clipped - phys) > 1e-12)
        return (clipped - norm_mean) / norm_std, was_clipped

    ar_phi = None if data.ar_phi is None else jnp.asarray(data.ar_phi)

    def _control_vector(history, weekday):
        control_history = history
        if ar_phi is not None:
            control_history = history[1:] - ar_phi[jnp.newaxis, :] * history[:-1]
        signatures = [
            signature_extractors[length](control_history[-length:])
            for length in cfg.SIGNATURE_PATH_LENGTHS
        ]
        explicit = sir_data._explicit_recon_control_vector_jax(
            control_history,
            cfg,
            weekday,
        )
        return jnp.concatenate((*signatures, explicit))

    def _plugin_transition_coordinates(state, control_vec):
        """Noise-free location and accumulated local scale for the audit map.

        This is a fixed plug-in coordinate map, not the exact conditional mean
        and variance of the stochastic projected micro-step simulator below.
        Applying this same map to the observed and simulated paths preserves
        the fitted-null score comparison.
        """
        def micro_step(carry, _unused):
            y_curr, var_acc = carry
            drift = model._calculate_drift(y_curr, control_vec)
            diffusion = model._calculate_diffusion(y_curr, control_vec) + cfg.DIFFUSION_REG
            return (
                y_curr + drift * dt,
                var_acc + (diffusion**2) * dt,
            ), None

        (y_final, var_final), _ = jax.lax.scan(
            micro_step,
            (state, jnp.zeros_like(state)),
            jnp.arange(n_substeps),
        )
        return y_final - state, jnp.sqrt(jnp.maximum(var_final, 1e-12))

    def _simulate_one(path_noise):
        def day_step(history, day_input):
            day_noise, weekday = day_input
            control_vec = _control_vector(history, weekday)
            state = history[-1]
            effective_drift, effective_sigma = _plugin_transition_coordinates(state, control_vec)

            def stochastic_micro_step(carry, dw_unit):
                current_state, n_clipped = carry
                drift = model._calculate_drift(current_state, control_vec)
                diffusion = model._calculate_diffusion(current_state, control_vec)
                proposed = current_state + drift * dt + diffusion * dw_unit * sqrt_dt
                next_state, was_clipped = _clip_physical(proposed)
                return (next_state, n_clipped + was_clipped.astype(jnp.int32)), None

            (next_state, n_clipped), _ = jax.lax.scan(
                stochastic_micro_step,
                (state, jnp.zeros((), dtype=jnp.int32)),
                day_noise,
            )
            residual = (
                next_state[1] - state[1] - effective_drift[1]
            ) / effective_sigma[1]
            next_history = jnp.roll(history, shift=-1, axis=0).at[-1].set(next_state)
            return next_history, (residual, n_clipped)

        _, (residuals, n_clipped) = jax.lax.scan(
            day_step,
            initial_history,
            (
                path_noise,
                (
                    jnp.array(data.val_features_df.index[start_pos].weekday())
                    + jnp.arange(horizon)
                )
                % 7,
            ),
        )
        return residuals, jnp.sum(n_clipped)

    @eqx.filter_jit
    def _simulate_batch(path_noise):
        return jax.vmap(_simulate_one)(path_noise)

    noise = jax.random.normal(
        jax.random.PRNGKey(seed),
        shape=(n_bootstrap, horizon, n_substeps),
    )
    residuals, n_clipped = _simulate_batch(noise)
    clip_fraction = n_clipped / float(horizon * n_substeps)
    return np.asarray(residuals), np.asarray(clip_fraction)


def _plot_case(
    case_name: str,
    observed_metrics: dict,
    draws_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, PLOT_METRICS):
        values = draws_df[metric].dropna().values
        observed = float(observed_metrics[metric])
        ax.hist(values, bins=30, alpha=0.75, color="#4c78a8")
        ax.axvline(observed, color="#d62728", lw=2, label=f"observed={observed:.3g}")
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(f"SIR BI fitted-null bootstrap: {case_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _run_case(
    case_name: str,
    updates: dict,
    n_bootstrap: int,
    seed: int,
    out_dir: Path,
) -> tuple[list[dict], pd.DataFrame, dict]:
    cfg = apply_config_updates(updates)
    _validate_supported_config(cfg)
    data = load_cached_data(cfg)
    model, model_path = _load_model(cfg, data)

    start_pos = _start_position(data, cfg)
    start_date = str(data.val_features_df.index[start_pos].date())
    audit_horizon = min(
        cfg.FORECAST_HORIZON,
        len(data.val_features_df) - 1 - start_pos,
    )
    end_date = str(data.val_features_df.index[start_pos + audit_horizon].date())
    observed_metrics = brownian_inversion.compute_brownian_inversion_metrics(
        model,
        start_date,
        data,
        cfg,
    )
    if observed_metrics is None:
        raise RuntimeError(f"No observed BI metrics produced for {case_name}.")

    residual_draws, clip_fraction = _simulate_null_residuals(
        model=model,
        data=data,
        cfg=cfg,
        start_pos=start_pos,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    draw_rows = []
    for bootstrap_id, residuals in enumerate(residual_draws):
        draw_rows.append(
            {
                "case": case_name,
                "experiment": config_module.get_experiment_tag(cfg),
                "start_date": start_date,
                "bootstrap_id": bootstrap_id,
                "clip_fraction": float(clip_fraction[bootstrap_id]),
                **compute_brownian_inversion_metrics_from_residuals(residuals),
            }
        )
    draws_df = pd.DataFrame(draw_rows)

    summary_rows = [
        summarize_metric(
            case_name=case_name,
            experiment=config_module.get_experiment_tag(cfg),
            start_date=start_date,
            n_bootstrap=n_bootstrap,
            metric=metric,
            observed=float(observed_metrics[metric]),
            null_values=draws_df[metric].values,
            metric_tails=METRIC_TAILS,
        )
        for metric in METRIC_TAILS
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_case(
        case_name,
        observed_metrics,
        draws_df,
        out_dir / f"sir_bi_bootstrap_{case_name}.png",
    )
    metadata = {
        "case": case_name,
        "experiment": config_module.get_experiment_tag(cfg),
        "start_date": start_date,
        "end_date": end_date,
        "start_position": int(start_pos),
        "start_rule": "max(validation midpoint, required control lookback)",
        "audit_horizon_increments": int(audit_horizon),
        "transition_coordinate": (
            "deterministic_substep_plugin_location_and_accumulated_local_scale"
        ),
        "transition_coordinate_is_exact_conditional_moment": False,
        "transition_coordinate_projects_each_substep": False,
        "null_generator_projects_each_substep": True,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "model_path": str(model_path),
        "mean_clip_fraction": float(draws_df["clip_fraction"].mean()),
        "max_clip_fraction": float(draws_df["clip_fraction"].max()),
        "observed_metrics": observed_metrics,
    }
    return summary_rows, draws_df, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        choices=sorted(CASES),
        help=f"Configurations to audit. Default: {', '.join(DEFAULT_CASES)}",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Number of fitted-null trajectories per configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260601,
        help="Base random seed for null-path simulation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for CSV, JSON, and PNG artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2.")

    logging.getLogger().setLevel(logging.WARNING)
    selected_cases = args.cases or DEFAULT_CASES
    all_summary_rows = []
    all_draws = []
    metadata = []

    for case_idx, case_name in enumerate(selected_cases):
        print(f"Running fitted-null BI bootstrap: {case_name} ({args.n_bootstrap} paths)")
        summary_rows, draws_df, case_metadata = _run_case(
            case_name=case_name,
            updates=CASES[case_name],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + case_idx,
            out_dir=args.out_dir,
        )
        all_summary_rows.extend(summary_rows)
        all_draws.append(draws_df)
        metadata.append(case_metadata)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_summary_rows)
    draws_df = pd.concat(all_draws, ignore_index=True)
    summary_path = args.out_dir / "sir_bi_bootstrap_summary.csv"
    draws_path = args.out_dir / "sir_bi_bootstrap_draws.csv"
    metadata_path = args.out_dir / "sir_bi_bootstrap_metadata.json"
    summary_df.to_csv(summary_path, index=False)
    draws_df.to_csv(draws_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    headline = summary_df[
        summary_df["metric"].isin(["bi_z_std", "bi_acf1", "bi_acf7", "bi_lb_stat"])
    ][
        [
            "case",
            "metric",
            "observed",
            "null_q025",
            "null_q500",
            "null_q975",
            "empirical_percentile",
            "empirical_pvalue",
        ]
    ]
    print()
    print(headline.to_string(index=False))
    print()
    print(f"Saved summary:  {summary_path}")
    print(f"Saved draws:    {draws_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
