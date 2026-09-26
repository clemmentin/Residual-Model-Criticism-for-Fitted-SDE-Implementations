"""Brownian-inversion residuals and numerical diagnostic statistics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import logging
import numpy as np
import pandas as pd

from config import Config, TrainingData, apply_control_ablation_array
from neural_sde import NeuralSDE
from scipy.stats import kstest, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from trajectory import JAXSignatureExtractor


def _evaluate_bi_residuals(
    model: NeuralSDE,
    start_date_str: str,
    data: TrainingData,
    config: Config,
) -> np.ndarray | None:
    """Return raw Brownian-inversion residuals on the validation path."""
    val_df = data.val_features_df
    start_date = pd.to_datetime(start_date_str)
    start_idx = val_df.index.get_indexer([start_date], method="nearest")[0]

    from sir_data import (
        apply_temporal_prewhiten_frame,
        create_sir_controls,
        get_sir_control_lookback,
    )

    max_lookback = get_sir_control_lookback(config)
    start_idx = max(start_idx, max_lookback)
    end_idx = min(start_idx + config.FORECAST_HORIZON, len(val_df) - 1)
    horizon = end_idx - start_idx

    if horizon < 10:
        logging.warning("Not enough future data for Brownian inversion validation.")
        return None

    history_start = start_idx - max_lookback + 1
    full_norm_history = jnp.array(val_df.iloc[history_start : end_idx + 1].values)
    full_val_history = jnp.array(val_df.values)
    control_history = full_val_history
    if data.ar_phi is not None:
        phi = jnp.array(data.ar_phi)
        control_history = control_history.at[1:].set(
            control_history[1:] - phi[jnp.newaxis, :] * control_history[:-1]
        )

    signature_extractors = {
        length: JAXSignatureExtractor(
            depth=config.SIGNATURE_DEPTH,
            augment_time=True,
            lead_lag=config.SIGNATURE_LEAD_LAG,
        )
        for length in config.SIGNATURE_PATH_LENGTHS
    }

    raw_control_bank = []
    for end_pos in range(max_lookback, len(val_df) + 1):
        signatures = []
        for length in config.SIGNATURE_PATH_LENGTHS:
            window = jax.lax.dynamic_slice(
                control_history,
                (end_pos - length, 0),
                (length, control_history.shape[1]),
            )
            sig = signature_extractors[length]._compute_single(window)
            signatures.append(sig)
        raw_control_bank.append(np.array(jnp.concatenate(signatures)))

    control_bank = apply_control_ablation_array(
        np.stack(raw_control_bank, axis=0),
        config,
        seed_offset=2000,
    )
    control_start = start_idx - (max_lookback - 1)
    controls_path = jnp.array(control_bank[control_start : control_start + horizon])
    control_source_df = apply_temporal_prewhiten_frame(val_df, data.ar_phi)
    control_frame = create_sir_controls(control_source_df, config)
    control_frame = pd.DataFrame(
        apply_control_ablation_array(
            control_frame.values,
            config,
            seed_offset=2000,
        ),
        index=control_frame.index,
    )
    controls_path = jnp.array(
        control_frame.loc[val_df.index[start_idx:end_idx]].values
    )
    country_idx = jnp.array(
        -1 if data.val_country_id is None else data.val_country_id,
        dtype=jnp.int32,
    )

    if controls_path.shape[0] < horizon:
        logging.warning("Not enough controls for Brownian inversion validation.")
        return None

    @eqx.filter_jit
    def evaluate_path():
        sub_dt = config.SDE_SUBSTEP_DT
        n_substeps = config.SDE_NUM_SUBSTEPS

        def scan_body(carry, t):
            _ = carry
            current_idx = max_lookback - 1 + t
            control_vec = controls_path[t]
            current_state = full_norm_history[current_idx]

            def micro_step(carry, _unused):
                y_curr, var_acc = carry
                d = model._calculate_drift(y_curr, control_vec, country_idx=country_idx)
                s = model._calculate_diffusion(y_curr, control_vec, country_idx=country_idx) + config.DIFFUSION_REG
                y_next = y_curr + d * sub_dt
                var_next = var_acc + (s ** 2) * sub_dt
                return (y_next, var_next), None

            (y_final, var_final), _ = jax.lax.scan(
                micro_step,
                (current_state, jnp.zeros_like(current_state)),
                jnp.arange(n_substeps),
            )

            effective_drift = y_final - current_state
            effective_sigma = jnp.sqrt(jnp.maximum(var_final, 1e-12))
            return None, (effective_drift, effective_sigma)

        _, (drifts, sigmas) = jax.lax.scan(scan_body, None, jnp.arange(horizon))
        return drifts, sigmas, full_norm_history

    drifts, sigmas, full_norm_history = evaluate_path()

    actual_norm_path = full_norm_history[max_lookback - 1 :]
    actual_increments = jnp.diff(actual_norm_path, axis=0)[:horizon]

    z_increments = actual_increments[:, 1]
    z_drifts = drifts[:, 1]
    z_sigmas = sigmas[:, 1]
    inverted_increments = (z_increments - z_drifts) / z_sigmas
    return np.array(inverted_increments)


def _lag_correlation(residuals: np.ndarray, lag: int) -> float:
    if len(residuals) <= lag:
        return float("nan")

    lhs = residuals[:-lag]
    rhs = residuals[lag:]
    if np.std(lhs) < 1e-12 or np.std(rhs) < 1e-12:
        return float("nan")

    return float(np.corrcoef(lhs, rhs)[0, 1])


def compute_brownian_inversion_metrics(
    model: NeuralSDE,
    start_date_str: str,
    data: TrainingData,
    config: Config,
) -> dict | None:
    """Compute SIR Brownian-inversion metrics on raw validation residuals."""
    residuals = _evaluate_bi_residuals(model, start_date_str, data, config)
    if residuals is None:
        return None

    return compute_brownian_inversion_metrics_from_residuals(residuals)


def compute_brownian_inversion_metrics_from_residuals(
    residuals: np.ndarray,
) -> dict:
    """Compute BI metrics from an already-inverted innovation sequence."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 10:
        raise ValueError("Need at least 10 finite residuals for BI metrics.")

    W_t = np.cumsum(residuals)
    horizon = len(residuals)
    ks_stat, ks_pvalue = kstest(residuals, "norm", args=(0, 1))

    try:
        lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_stat = float(lb_result["lb_stat"].iloc[0])
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
    except Exception:
        lb_stat = float("nan")
        lb_pvalue = float("nan")

    return {
        "bi_z_mean": float(np.mean(residuals)),
        "bi_z_std": float(np.std(residuals, ddof=1)),
        "bi_z_skew": float(skew(residuals)),
        "bi_z_kurt": float(kurtosis(residuals, fisher=True)),
        "bi_acf1": _lag_correlation(residuals, 1),
        "bi_acf7": _lag_correlation(residuals, 7),
        "bi_ks_stat": float(ks_stat),
        "bi_ks_pvalue": float(ks_pvalue),
        "bi_lb_stat": lb_stat,
        "bi_lb_pvalue": lb_pvalue,
        "bi_max_w_norm": float(np.max(np.abs(W_t)) / np.sqrt(horizon)),
    }
