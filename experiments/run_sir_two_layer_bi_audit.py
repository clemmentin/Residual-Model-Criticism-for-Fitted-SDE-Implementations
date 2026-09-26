"""
Two-layer Brownian-innovation audit for fitted SIR Neural SDEs.

This script complements the older scalar infected-coordinate BI residual.
It computes two fitted-null-calibrated diagnostics on the same validation
window:

1. Tangent-linear finite-step transition-covariance audit.
   The observed one-day transition residual is whitened by the model-implied
   one-day conditional covariance approximation in normalized state
   coordinates.  The covariance recursion propagates injected noise through
   the local drift linearization over solver substeps.

2. Instantaneous diffusion-subspace audit.
   The same observed increment is decomposed into the local diffusion loading
   direction and its orthogonal complement.  The loading coefficient is
   standardized by the square root of the observation interval.

All p-values are empirical fitted-null rank comparisons.  Null paths are
simulated from the fitted model, signature controls are recomputed along each
simulated path, and the same audit maps are reapplied.

Run from the sde directory, for example:
    python experiments/run_sir_two_layer_bi_audit.py recov28_frzgamma --n-bootstrap 500
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
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, kstest
from statsmodels.stats.diagnostic import acorr_ljungbox


ROOT = Path(__file__).resolve().parents[1]

import config as config_module
from config import apply_control_ablation_array  # noqa: E402
from trajectory import JAXSignatureExtractor  # noqa: E402

from experiments._bi_audit_common import apply_config_updates, load_cached_data, summarize_metric
from experiments._sir_model_loader import load_first_model as _load_model


DEFAULT_OUT_DIR = ROOT / "cache" / "summaries" / "two_layer_bi"

CASES = {
    "baseline": {"FREEZE_GAMMA": False},
    "baseline_frzgamma": {"FREEZE_GAMMA": True},
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

METRIC_TAILS = {
    "step_z1_mean": "centered",
    "step_z1_std": "centered",
    "step_z1_acf1": "centered",
    "step_z1_acf7": "centered",
    "step_z1_ks_stat": "upper",
    "step_z1_lb_stat": "upper",
    "step_z1_max_w_norm": "upper",
    "step_energy_mean": "upper",
    "step_energy_acf1": "centered",
    "step_zero_rms": "upper",
    "step_zero_max": "upper",
    "diff_z_mean": "centered",
    "diff_z_std": "centered",
    "diff_z_acf1": "centered",
    "diff_z_acf7": "centered",
    "diff_z_ks_stat": "upper",
    "diff_z_lb_stat": "upper",
    "diff_z_max_w_norm": "upper",
    "diff_perp_rms": "upper",
    "diff_perp_max": "upper",
    "diff_perp_acf1": "centered",
    "diff_step_z_mean": "centered",
    "diff_step_z_std": "centered",
    "diff_step_z_acf1": "centered",
    "diff_step_z_acf7": "centered",
    "diff_step_z_ks_stat": "upper",
    "diff_step_z_lb_stat": "upper",
    "diff_step_z_max_w_norm": "upper",
    "diff_step_perp_rms": "upper",
    "diff_step_perp_max": "upper",
    "diff_step_perp_acf1": "centered",
    "mean_gap_perp_rms": "upper",
    "mean_gap_perp_max": "upper",
}




def _validate_supported_config(cfg: config_module.Config) -> None:
    if str(cfg.CONTROL_ABLATION).lower() != "none":
        raise ValueError("Two-layer BI currently supports CONTROL_ABLATION='none' only.")
    if str(cfg.EXPLICIT_RECON_CONTROLS).lower() != "none":
        raise ValueError("Two-layer BI currently supports EXPLICIT_RECON_CONTROLS='none' only.")
    if cfg.USE_TEMPORAL_PREWHITEN:
        raise ValueError("Two-layer BI currently expects USE_TEMPORAL_PREWHITEN=False.")


def _start_position(data: config_module.TrainingData, cfg: config_module.Config) -> int:
    return max(len(data.val_features_df["I"].values) // 2, max(cfg.SIGNATURE_PATH_LENGTHS))


def _observed_arrays(
    data: config_module.TrainingData,
    cfg: config_module.Config,
    start_pos: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, str]:
    """Return states, next states, controls, end dates, and start date."""
    val_df = data.val_features_df
    max_lookback = max(cfg.SIGNATURE_PATH_LENGTHS)
    start_idx = max(int(start_pos), max_lookback)
    end_idx = min(start_idx + cfg.FORECAST_HORIZON, len(val_df) - 1)
    horizon = end_idx - start_idx
    if horizon < 10:
        raise ValueError("Not enough held-out observations for two-layer BI.")

    full_val_history = jnp.asarray(val_df.values)
    control_history = full_val_history
    if data.ar_phi is not None:
        phi = jnp.asarray(data.ar_phi)
        control_history = control_history.at[1:].set(
            control_history[1:] - phi[jnp.newaxis, :] * control_history[:-1]
        )

    signature_extractors = {
        length: JAXSignatureExtractor(
            depth=cfg.SIGNATURE_DEPTH,
            augment_time=True,
            lead_lag=cfg.SIGNATURE_LEAD_LAG,
        )
        for length in cfg.SIGNATURE_PATH_LENGTHS
    }

    raw_control_bank = []
    for end_pos in range(max_lookback, len(val_df) + 1):
        signatures = []
        for length in cfg.SIGNATURE_PATH_LENGTHS:
            window = jax.lax.dynamic_slice(
                control_history,
                (end_pos - length, 0),
                (length, control_history.shape[1]),
            )
            signatures.append(signature_extractors[length]._compute_single(window))
        raw_control_bank.append(np.asarray(jnp.concatenate(signatures)))

    control_bank = apply_control_ablation_array(
        np.stack(raw_control_bank, axis=0),
        cfg,
        seed_offset=2000,
    )
    control_start = start_idx - (max_lookback - 1)
    controls = control_bank[control_start : control_start + horizon]
    if controls.shape[0] < horizon:
        raise ValueError("Not enough controls for observed two-layer BI.")

    states = val_df[["S", "I"]].iloc[start_idx:end_idx].to_numpy(dtype=float)
    next_states = val_df[["S", "I"]].iloc[start_idx + 1 : end_idx + 1].to_numpy(dtype=float)
    end_dates = pd.DatetimeIndex(val_df.index[start_idx + 1 : end_idx + 1])
    start_date = str(val_df.index[start_idx].date())
    return states, next_states, np.asarray(controls, dtype=float), end_dates, start_date


def _project_rank_one_residual(
    residual: jax.Array,
    local_diffusion: jax.Array,
    observation_h: float,
    rank_abs_tol: float,
    pinv_rel_tol: float,
) -> tuple[jax.Array, jax.Array]:
    """Return ``G^+ residual / sqrt(H)`` and ``(I - GG^+) residual``.

    The raw pseudoinverse coefficient must be used to reconstruct the
    projection.  Only the reported instantaneous-basis coordinate is divided
    by ``sqrt(H)``; applying that normalization to the projection itself would
    change the orthogonal residual whenever ``H != 1``.
    """
    g_norm2 = jnp.dot(local_diffusion, local_diffusion)
    pinv_cutoff = jnp.maximum(rank_abs_tol, pinv_rel_tol * g_norm2)
    has_diffusion = g_norm2 > pinv_cutoff
    raw_coefficient = jnp.where(
        has_diffusion,
        jnp.dot(local_diffusion, residual) / jnp.maximum(g_norm2, rank_abs_tol),
        jnp.nan,
    )
    standardized_coefficient = raw_coefficient / jnp.sqrt(observation_h)
    projected = jnp.where(
        has_diffusion,
        local_diffusion * raw_coefficient,
        jnp.zeros_like(local_diffusion),
    )
    return standardized_coefficient, residual - projected


def _make_audit_functions(
    model,
    cfg: config_module.Config,
    country_idx: int,
    rank_rel_tol: float,
    rank_abs_tol: float,
    pinv_rel_tol: float,
):
    country_idx_j = jnp.array(country_idx, dtype=jnp.int32)
    dt = float(cfg.SDE_SUBSTEP_DT)
    n_substeps = int(cfg.SDE_NUM_SUBSTEPS)
    observation_h = dt * n_substeps
    if not np.isfinite(observation_h) or observation_h <= 0.0:
        raise ValueError("The observation interval must be finite and positive.")

    @eqx.filter_jit
    def audit_batch(m, states, next_states, controls):
        def transition_moments(state, control_vec):
            eye = jnp.eye(state.shape[0], dtype=state.dtype)

            def micro_step(carry, _unused):
                y_curr, cov_acc = carry
                def drift_at(y_value):
                    return m._calculate_drift(y_value, control_vec, country_idx=country_idx_j)

                drift = drift_at(y_curr)
                drift_jac = jax.jacfwd(drift_at)(y_curr)
                diffusion = m._calculate_diffusion(y_curr, control_vec, country_idx=country_idx_j)
                transition_jac = eye + drift_jac * dt
                cov_next = transition_jac @ cov_acc @ transition_jac.T
                cov_next = cov_next + jnp.outer(diffusion, diffusion) * dt
                return (y_curr + drift * dt, cov_next), None

            (y_final, cov_final), _ = jax.lax.scan(
                micro_step,
                (state, jnp.zeros((state.shape[0], state.shape[0]), dtype=state.dtype)),
                jnp.arange(n_substeps),
            )
            cov_final = 0.5 * (cov_final + cov_final.T)
            return y_final - state, cov_final

        def audit_one(state, next_state, control_vec):
            increment = next_state - state
            step_drift, step_cov = transition_moments(state, control_vec)
            delta = increment - step_drift

            evals_asc, evecs_asc = jnp.linalg.eigh(step_cov)
            evals = evals_asc[::-1]
            evecs = evecs_asc[:, ::-1]
            evals = jnp.maximum(evals, 0.0)
            max_eval = jnp.maximum(evals[0], 0.0)
            eig_cutoff = jnp.maximum(rank_abs_tol, rank_rel_tol * max_eval)
            rank_mask = evals > eig_cutoff
            rank = jnp.sum(rank_mask.astype(jnp.int32))

            coords = evecs.T @ delta
            safe_evals = jnp.where(rank_mask, jnp.maximum(evals, rank_abs_tol), 1.0)
            z_all = coords / jnp.sqrt(safe_evals)
            z_all = jnp.where(rank_mask, z_all, 0.0)
            z1 = jnp.where(rank_mask[0], z_all[0], jnp.nan)
            z2 = jnp.where(rank_mask[1], z_all[1], jnp.nan)
            step_energy = jnp.sum(z_all * z_all)
            zero_coords = jnp.where(rank_mask, 0.0, coords)
            zero_norm = jnp.sqrt(jnp.sum(zero_coords * zero_coords))

            local_drift = m._calculate_drift(state, control_vec, country_idx=country_idx_j)
            local_diffusion = m._calculate_diffusion(state, control_vec, country_idx=country_idx_j)
            local_mean = local_drift * observation_h
            local_residual = increment - local_mean
            mean_gap = step_drift - local_mean
            g_norm2 = jnp.dot(local_diffusion, local_diffusion)
            pinv_cutoff = jnp.maximum(rank_abs_tol, pinv_rel_tol * g_norm2)
            has_diffusion = g_norm2 > pinv_cutoff

            projection_args = (
                local_diffusion,
                observation_h,
                rank_abs_tol,
                pinv_rel_tol,
            )
            diff_coef, diff_perp = _project_rank_one_residual(
                local_residual, *projection_args
            )
            diff_step_coef, diff_step_perp = _project_rank_one_residual(
                delta, *projection_args
            )
            _gap_coef, mean_gap_perp = _project_rank_one_residual(
                mean_gap, *projection_args
            )
            diff_perp_norm = jnp.linalg.norm(diff_perp)
            diff_step_perp_norm = jnp.linalg.norm(diff_step_perp)
            mean_gap_perp_norm = jnp.linalg.norm(mean_gap_perp)

            # The SIR state is two-dimensional.  A signed orthogonal coordinate
            # makes the identity local = step-centered + propagated-mean gap
            # directly checkable before norms and path summaries are taken.
            safe_g_norm = jnp.sqrt(jnp.maximum(g_norm2, rank_abs_tol))
            unit_perp = jnp.where(
                has_diffusion,
                jnp.stack([-local_diffusion[1], local_diffusion[0]]) / safe_g_norm,
                jnp.zeros_like(local_diffusion),
            )
            diff_perp_signed = jnp.dot(unit_perp, diff_perp)
            diff_step_perp_signed = jnp.dot(unit_perp, diff_step_perp)
            mean_gap_perp_signed = jnp.dot(unit_perp, mean_gap_perp)

            return {
                "step_z1": z1,
                "step_z2": z2,
                "step_energy": step_energy,
                "step_zero_norm": zero_norm,
                "step_rank": rank.astype(jnp.float32),
                "step_eig1": evals[0],
                "step_eig2": evals[1],
                "step_eig_ratio": evals[1] / jnp.maximum(evals[0], rank_abs_tol),
                "diff_z": diff_coef,
                "diff_perp_norm": diff_perp_norm,
                "diff_perp_signed": diff_perp_signed,
                "diff_step_z": diff_step_coef,
                "diff_step_perp_norm": diff_step_perp_norm,
                "diff_step_perp_signed": diff_step_perp_signed,
                "mean_gap_perp_norm": mean_gap_perp_norm,
                "mean_gap_perp_signed": mean_gap_perp_signed,
                "diff_g_norm": jnp.sqrt(jnp.maximum(g_norm2, 0.0)),
            }

        return jax.vmap(audit_one)(states, next_states, controls)

    return audit_batch


def _lag_corr(values: np.ndarray, lag: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if lag <= 0 or x.size <= lag:
        return float("nan")
    left = x[:-lag]
    right = x[lag:]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _series_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_skew": float("nan"),
            f"{prefix}_kurt": float("nan"),
            f"{prefix}_acf1": float("nan"),
            f"{prefix}_acf7": float("nan"),
            f"{prefix}_ks_stat": float("nan"),
            f"{prefix}_lb_stat": float("nan"),
            f"{prefix}_max_w_norm": float("nan"),
        }
    try:
        ks_stat = float(kstest(x, "norm", args=(0, 1))[0])
    except Exception:
        ks_stat = float("nan")
    try:
        lb = acorr_ljungbox(x, lags=[10], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[-1])
    except Exception:
        lb_stat = float("nan")
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=1)),
        f"{prefix}_skew": float(skew(x)),
        f"{prefix}_kurt": float(kurtosis(x, fisher=True)),
        f"{prefix}_acf1": _lag_corr(x, 1),
        f"{prefix}_acf7": _lag_corr(x, 7),
        f"{prefix}_ks_stat": ks_stat,
        f"{prefix}_lb_stat": lb_stat,
        f"{prefix}_max_w_norm": float(np.max(np.abs(np.cumsum(x))) / np.sqrt(x.size)),
    }


def _path_metrics(path: dict[str, np.ndarray]) -> dict[str, float]:
    step_z1 = np.asarray(path["step_z1"], dtype=float)
    step_energy = np.asarray(path["step_energy"], dtype=float)
    step_zero = np.asarray(path["step_zero_norm"], dtype=float)
    diff_z = np.asarray(path["diff_z"], dtype=float)
    diff_perp = np.asarray(path["diff_perp_norm"], dtype=float)
    diff_step_z = np.asarray(path["diff_step_z"], dtype=float)
    diff_step_perp = np.asarray(path["diff_step_perp_norm"], dtype=float)
    mean_gap_perp = np.asarray(path["mean_gap_perp_norm"], dtype=float)
    ranks = np.asarray(path["step_rank"], dtype=float)
    eig_ratio = np.asarray(path["step_eig_ratio"], dtype=float)

    metrics = {}
    metrics.update(_series_stats("step_z1", step_z1))
    metrics["step_energy_mean"] = float(np.nanmean(step_energy))
    metrics["step_energy_acf1"] = _lag_corr(step_energy, 1)
    metrics["step_zero_rms"] = float(np.sqrt(np.nanmean(step_zero**2)))
    metrics["step_zero_max"] = float(np.nanmax(step_zero))
    metrics["step_rank_mean"] = float(np.nanmean(ranks))
    metrics["step_rank1_fraction"] = float(np.nanmean(ranks == 1.0))
    metrics["step_rank2_fraction"] = float(np.nanmean(ranks == 2.0))
    metrics["step_eig_ratio_median"] = float(np.nanmedian(eig_ratio))
    metrics["step_eig_ratio_max"] = float(np.nanmax(eig_ratio))
    metrics.update(_series_stats("diff_z", diff_z))
    metrics["diff_perp_rms"] = float(np.sqrt(np.nanmean(diff_perp**2)))
    metrics["diff_perp_max"] = float(np.nanmax(diff_perp))
    metrics["diff_perp_acf1"] = _lag_corr(diff_perp, 1)
    metrics.update(_series_stats("diff_step_z", diff_step_z))
    metrics["diff_step_perp_rms"] = float(np.sqrt(np.nanmean(diff_step_perp**2)))
    metrics["diff_step_perp_max"] = float(np.nanmax(diff_step_perp))
    metrics["diff_step_perp_acf1"] = _lag_corr(diff_step_perp, 1)
    metrics["mean_gap_perp_rms"] = float(np.sqrt(np.nanmean(mean_gap_perp**2)))
    metrics["mean_gap_perp_max"] = float(np.nanmax(mean_gap_perp))
    local_signed = np.asarray(path["diff_perp_signed"], dtype=float)
    step_signed = np.asarray(path["diff_step_perp_signed"], dtype=float)
    gap_signed = np.asarray(path["mean_gap_perp_signed"], dtype=float)
    metrics["perp_identity_max_abs_error"] = float(
        np.nanmax(np.abs(local_signed - step_signed - gap_signed))
    )
    metrics["diff_g_norm_mean"] = float(np.nanmean(path["diff_g_norm"]))
    metrics["n_steps"] = int(np.sum(np.isfinite(step_z1)))
    return metrics


def _simulate_null_paths(
    model,
    cfg: config_module.Config,
    data: config_module.TrainingData,
    start_pos: int,
    n_bootstrap: int,
    seed: int,
    audit_batch,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    max_lookback = max(cfg.SIGNATURE_PATH_LENGTHS)
    horizon = min(cfg.FORECAST_HORIZON, len(data.val_features_df) - 1 - start_pos)
    if horizon < 10:
        raise ValueError("Not enough held-out observations for null simulation.")

    initial_history = jnp.asarray(
        data.val_features_df.iloc[start_pos - max_lookback + 1 : start_pos + 1].values
    )
    signature_extractors = {
        length: JAXSignatureExtractor(
            depth=cfg.SIGNATURE_DEPTH,
            augment_time=True,
            lead_lag=cfg.SIGNATURE_LEAD_LAG,
        )
        for length in cfg.SIGNATURE_PATH_LENGTHS
    }
    dt = float(cfg.SDE_SUBSTEP_DT)
    n_substeps = int(cfg.SDE_NUM_SUBSTEPS)
    sqrt_dt = jnp.sqrt(dt)
    norm_mean = jnp.asarray(data.norm_mean)
    norm_std = jnp.asarray(data.norm_std)

    def _clip_physical(state):
        phys = state * norm_std + norm_mean
        clipped = jnp.clip(phys, jnp.array([1e-6, -20.0]), jnp.array([1.0 - 1e-6, 0.0]))
        was_clipped = jnp.any(jnp.abs(clipped - phys) > 1e-12)
        return (clipped - norm_mean) / norm_std, was_clipped

    def _control_vector(history):
        signatures = [
            signature_extractors[length](history[-length:])
            for length in cfg.SIGNATURE_PATH_LENGTHS
        ]
        return jnp.concatenate(signatures)

    @eqx.filter_jit
    def simulate_paths(m, path_noise):
        def simulate_one(noise_one):
            def day_step(history, day_noise):
                control_vec = _control_vector(history)
                state = history[-1]

                def stochastic_micro_step(carry, dw_unit):
                    current_state, n_clipped = carry
                    drift = m._calculate_drift(current_state, control_vec)
                    diffusion = m._calculate_diffusion(current_state, control_vec)
                    proposed = current_state + drift * dt + diffusion * dw_unit * sqrt_dt
                    next_state, was_clipped = _clip_physical(proposed)
                    return (next_state, n_clipped + was_clipped.astype(jnp.int32)), None

                (next_state, n_clipped), _ = jax.lax.scan(
                    stochastic_micro_step,
                    (state, jnp.zeros((), dtype=jnp.int32)),
                    day_noise,
                )
                next_history = jnp.roll(history, shift=-1, axis=0).at[-1].set(next_state)
                return next_history, (state, next_state, control_vec, n_clipped)

            _, (states, next_states, controls, n_clipped) = jax.lax.scan(
                day_step,
                initial_history,
                noise_one,
            )
            return states, next_states, controls, jnp.sum(n_clipped)

        return jax.vmap(simulate_one)(path_noise)

    noise = jax.random.normal(
        jax.random.PRNGKey(seed),
        shape=(n_bootstrap, horizon, n_substeps),
    )
    states, next_states, controls, n_clipped = simulate_paths(model, noise)
    audited = audit_batch(
        model,
        states.reshape((-1, states.shape[-1])),
        next_states.reshape((-1, next_states.shape[-1])),
        controls.reshape((-1, controls.shape[-1])),
    )
    path_outputs: list[dict[str, np.ndarray]] = []
    for bootstrap_id in range(n_bootstrap):
        lo = bootstrap_id * horizon
        hi = lo + horizon
        path_outputs.append({key: np.asarray(value[lo:hi]) for key, value in audited.items()})
    clip_fraction = np.asarray(n_clipped) / float(horizon * n_substeps)
    return path_outputs, clip_fraction


def _run_case(
    case_name: str,
    updates: dict[str, object],
    n_bootstrap: int,
    seed: int,
    out_dir: Path,
    rank_rel_tol: float,
    rank_abs_tol: float,
    pinv_rel_tol: float,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cfg = apply_config_updates(updates)
    _validate_supported_config(cfg)
    data = load_cached_data(cfg)
    model, model_path = _load_model(cfg, data)
    start_pos = _start_position(data, cfg)
    states, next_states, controls, end_dates, start_date = _observed_arrays(data, cfg, start_pos)
    country_idx = -1 if data.val_country_id is None else int(data.val_country_id)
    audit_batch = _make_audit_functions(
        model=model,
        cfg=cfg,
        country_idx=country_idx,
        rank_rel_tol=rank_rel_tol,
        rank_abs_tol=rank_abs_tol,
        pinv_rel_tol=pinv_rel_tol,
    )

    observed_raw = audit_batch(
        model,
        jnp.asarray(states),
        jnp.asarray(next_states),
        jnp.asarray(controls),
    )
    observed_path = {key: np.asarray(value) for key, value in observed_raw.items()}
    observed_metrics = _path_metrics(observed_path)

    null_paths, clip_fraction = _simulate_null_paths(
        model=model,
        cfg=cfg,
        data=data,
        start_pos=start_pos,
        n_bootstrap=n_bootstrap,
        seed=seed,
        audit_batch=audit_batch,
    )

    experiment = config_module.get_experiment_tag(cfg)
    draw_rows = []
    for bootstrap_id, path in enumerate(null_paths):
        draw_rows.append(
            {
                "case": case_name,
                "experiment": experiment,
                "start_date": start_date,
                "bootstrap_id": bootstrap_id,
                "clip_fraction": float(clip_fraction[bootstrap_id]),
                **_path_metrics(path),
            }
        )
    draws_df = pd.DataFrame(draw_rows)

    observed_df = pd.DataFrame(
        {
            "case": case_name,
            "experiment": experiment,
            "start_date": start_date,
            "date": [ts.date().isoformat() for ts in end_dates],
            **observed_path,
        }
    )

    summary_rows = []
    for metric in METRIC_TAILS:
        summary_rows.append(
            summarize_metric(
                case_name=case_name,
                experiment=experiment,
                start_date=start_date,
                n_bootstrap=n_bootstrap,
                metric=metric,
                observed=float(observed_metrics[metric]),
                null_values=draws_df[metric].values,
                metric_tails=METRIC_TAILS,
            )
        )

    metadata = {
        "case": case_name,
        "experiment": experiment,
        "start_date": start_date,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "model_path": str(model_path),
        "rank_rel_tol": rank_rel_tol,
        "rank_abs_tol": rank_abs_tol,
        "pinv_rel_tol": pinv_rel_tol,
        "transition_covariance_method": "tangent_linear_euler_substep",
        "instantaneous_projection_centerings": {
            "legacy_diff": "increment_minus_local_drift_times_observation_h",
            "diff_step": "increment_minus_propagated_finite_step_mean",
            "mean_gap": "propagated_finite_step_mean_minus_local_drift_times_observation_h",
        },
        "instantaneous_coordinate_normalization": "G_pinv_residual_divided_by_sqrt_observation_h",
        "observation_h": float(cfg.SDE_SUBSTEP_DT * cfg.SDE_NUM_SUBSTEPS),
        "observed_perp_identity_max_abs_error": float(
            observed_metrics["perp_identity_max_abs_error"]
        ),
        "max_null_perp_identity_max_abs_error": float(
            draws_df["perp_identity_max_abs_error"].max()
        ),
        "mean_clip_fraction": float(draws_df["clip_fraction"].mean()),
        "max_clip_fraction": float(draws_df["clip_fraction"].max()),
        "observed_metrics": observed_metrics,
    }
    return summary_rows, draws_df, observed_df, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        choices=sorted(CASES),
        help=f"Configurations to audit. Default: {', '.join(DEFAULT_CASES)}",
    )
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--rank-rel-tol", type=float, default=1e-6)
    parser.add_argument("--rank-abs-tol", type=float, default=1e-12)
    parser.add_argument("--pinv-rel-tol", type=float, default=1e-10)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2.")

    logging.getLogger().setLevel(logging.WARNING)
    selected_cases = args.cases or DEFAULT_CASES
    all_summary_rows: list[dict[str, object]] = []
    all_draws: list[pd.DataFrame] = []
    all_observed: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []

    for case_idx, case_name in enumerate(selected_cases):
        print(f"Running two-layer BI fitted-null audit: {case_name} ({args.n_bootstrap} paths)")
        summary_rows, draws_df, observed_df, case_metadata = _run_case(
            case_name=case_name,
            updates=CASES[case_name],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + case_idx,
            out_dir=args.out_dir,
            rank_rel_tol=args.rank_rel_tol,
            rank_abs_tol=args.rank_abs_tol,
            pinv_rel_tol=args.pinv_rel_tol,
        )
        all_summary_rows.extend(summary_rows)
        all_draws.append(draws_df)
        all_observed.append(observed_df)
        metadata.append(case_metadata)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_summary_rows)
    draws_df = pd.concat(all_draws, ignore_index=True)
    observed_df = pd.concat(all_observed, ignore_index=True)

    summary_path = args.out_dir / "sir_two_layer_bi_summary.csv"
    draws_path = args.out_dir / "sir_two_layer_bi_draws.csv"
    observed_path = args.out_dir / "sir_two_layer_bi_observed_path.csv"
    metadata_path = args.out_dir / "sir_two_layer_bi_metadata.json"

    summary_df.to_csv(summary_path, index=False)
    draws_df.to_csv(draws_path, index=False)
    observed_df.to_csv(observed_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    headline_metrics = [
        "step_z1_std",
        "step_z1_acf1",
        "step_z1_acf7",
        "step_zero_rms",
        "diff_z_std",
        "diff_z_acf1",
        "diff_z_acf7",
        "diff_perp_rms",
    ]
    headline = summary_df[summary_df["metric"].isin(headline_metrics)][
        [
            "case",
            "metric",
            "observed",
            "null_q025",
            "null_q500",
            "null_q975",
            "empirical_pvalue",
        ]
    ]
    print()
    print(headline.to_string(index=False))
    print()
    print(f"Saved summary:  {summary_path}")
    print(f"Saved draws:    {draws_path}")
    print(f"Saved observed: {observed_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
