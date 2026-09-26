"""Forecast simulation, rolling evaluation and result reporting."""

import equinox as eqx
import jax
import jax.numpy as jnp
import logging
import numpy as np
import pandas as pd

from brownian_inversion import (
    _evaluate_bi_residuals,
    compute_brownian_inversion_metrics_from_residuals,
)
from config import (
    Config,
    TrainingData,
    apply_control_ablation_array,
    apply_control_ablation_frame,
    get_experiment_tag,
    get_recon_control_size,
)
from neural_sde import NeuralSDE
from pathlib import Path
from sir_data import (
    _explicit_recon_control_vector_jax,
    apply_temporal_prewhiten_frame,
    create_sir_controls,
    get_sir_control_lookback,
)
from tqdm import tqdm
from trajectory import JAXSignatureExtractor, get_signature_size
from typing import Dict, List, Tuple
from visualization import (
    plot_brownian_inversion_validation,
    plot_rolling_validation_sir,
    plot_sir_forecast,
    plot_sir_parameters,
)


def forecast_sir(
    ensemble_models: List[NeuralSDE],
    val_features_df: pd.DataFrame,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    config: Config,
    key: jnp.ndarray,
    start_idx: int = None,
    ar_phi: np.ndarray = None,
    country_idx: int | None = None,
) -> Tuple[jnp.ndarray, "pd.DatetimeIndex"]:
    """
    Simulate SIR paths from a point in the validation country.

    Alignment convention:
    - The last point in initial_path is treated as time t-1.
    - After one day of integration, the first forecasted point is time t.
    """
    max_lookback = get_sir_control_lookback(config)
    raw_lookback = max_lookback + (1 if ar_phi is not None else 0)
    control_mode = str(config.CONTROL_ABLATION).lower()
    if start_idx is None:
        start_idx = len(val_features_df) // 2
    start_idx = max(start_idx, raw_lookback)

    initial_path = jnp.array(
        val_features_df.iloc[start_idx - raw_lookback : start_idx].values
    )

    sig_extractors = {
        l: JAXSignatureExtractor(depth=config.SIGNATURE_DEPTH, augment_time=True, lead_lag=config.SIGNATURE_LEAD_LAG)
        for l in config.SIGNATURE_PATH_LENGTHS
    }
    signature_dim = sum(
        get_signature_size(config.STATE_SIZE, config.SIGNATURE_DEPTH, augment_time=True, lead_lag=config.SIGNATURE_LEAD_LAG)
        for _ in config.SIGNATURE_PATH_LENGTHS
    )
    control_dim = signature_dim + get_recon_control_size(config)
    shuffled_control_bank = None
    if control_mode == "shuffle":
        control_source_df = apply_temporal_prewhiten_frame(val_features_df, ar_phi)
        raw_bank = create_sir_controls(control_source_df, config)
        shuffled_control_bank = jnp.array(
            apply_control_ablation_array(raw_bank.values, config, seed_offset=3000)
        )

    dt = config.SDE_SUBSTEP_DT
    num_substeps = config.SDE_NUM_SUBSTEPS
    _country_idx = jnp.array(-1 if country_idx is None else country_idx, dtype=jnp.int32)
    paths_per_m = max(1, config.NUM_SPAGHETTI_PATHS // len(ensemble_models))
    total_paths = paths_per_m * len(ensemble_models)

    # Generate noise
    key, ic_key, path_key = jax.random.split(key, 3)
    ic_noise = jax.random.normal(ic_key, shape=(total_paths, 2)) * 0.01

    # Process noise: (TotalPaths, Horizon, Substeps, State)
    path_noise = jax.random.normal(
        path_key, shape=(total_paths, config.FORECAST_HORIZON, num_substeps, 2)
    )

    # Physics-aware clip bounds in normalised space
    # S ∈ [1e-6, 1-1e-6],  Z = log(I) ∈ [-20, 0]
    _phys_lo = jnp.array([1e-6, -20.0])
    _phys_hi = jnp.array([1.0 - 1e-6, 0.0])
    _norm_mean_j = jnp.array(norm_mean)
    _norm_std_j  = jnp.array(norm_std)

    def _clip_physical(s):
        """De-normalise → clip to physical SIR bounds → re-normalise."""
        phys = s * _norm_std_j + _norm_mean_j
        phys = jnp.clip(phys, _phys_lo, _phys_hi)
        return (phys - _norm_mean_j) / _norm_std_j

    # AR(1) pre-whitening coefficient for signature input (None → no whitening)
    _ar_phi_j = jnp.array(ar_phi) if ar_phi is not None else None

    def _whiten_path(path):
        """Apply AR(1) filter to normalised sliding window for signature input.
        ε_t = ỹ_t − φ · ỹ_{t-1};  first row is left unchanged.
        path: (L, state_size)  →  returns same shape.
        """
        return jnp.concatenate([
            path[:1],
            path[1:] - _ar_phi_j[jnp.newaxis, :] * path[:-1],
        ], axis=0)

    def _control_history(path):
        """Match the global AR(1) filter used to build training controls."""
        if _ar_phi_j is None:
            return path
        return path[1:] - _ar_phi_j[jnp.newaxis, :] * path[:-1]

    def simulate_one_path(model, ic_n, p_n):
        """SDE simulation: Drift + Diffusion dW."""
        # p_n shape: (Horizon, Substeps, 2)
        perturbed_path = initial_path.at[-1].set(initial_path[-1] + ic_n)

        # Scan over day-wise Brownian increments
        def scan_step(carry, step_input):
            norm_path = carry
            day_idx, day_noise = step_input

            if control_mode == "none":
                sig_path = _control_history(norm_path)
                sigs_list = [
                    sig_extractors[l](sig_path[-l:])
                    for l in config.SIGNATURE_PATH_LENGTHS
                ]
                current_weekday = (
                    jnp.array(val_features_df.index[start_idx - 1].weekday())
                    + day_idx
                ) % 7
                explicit = _explicit_recon_control_vector_jax(
                    sig_path,
                    config,
                    current_weekday,
                )
                control_vec = jnp.concatenate((*sigs_list, explicit))
            elif control_mode == "zero":
                control_vec = jnp.zeros((control_dim,), dtype=norm_path.dtype)
            else:
                bank_idx = (start_idx - max_lookback + day_idx) % shuffled_control_bank.shape[0]
                control_vec = shuffled_control_bank[bank_idx]

            state = norm_path[-1]

            def substep(s, dw_unit):
                # dw_unit: (2,) ~ N(0,1)
                d = model._calculate_drift(s, control_vec, country_idx=_country_idx)

                # Diffusion: Diag(0, sigma)
                diff = model._calculate_diffusion(s, control_vec, country_idx=_country_idx)

                dw = dw_unit * jnp.sqrt(dt)
                # Physics-aware clip: enforce S∈(0,1), I≤1 after every micro-step
                return _clip_physical(s + d * dt + diff * dw), None

            # Sub-step scan (10 steps)
            next_s, _ = jax.lax.scan(substep, state, day_noise)

            # NaN guard: if substep produced NaN (e.g. from 0/0), fall back to
            # previous state so the path continues rather than propagating NaN.
            next_s = jnp.where(jnp.isfinite(next_s), next_s, state)
            next_s = _clip_physical(next_s)
            next_path = jnp.roll(norm_path, shift=-1, axis=0).at[-1].set(next_s)

            return next_path, next_s

        _, states = jax.lax.scan(
            scan_step,
            perturbed_path,
            (jnp.arange(config.FORECAST_HORIZON), p_n),
        )
        # Decode normalised Z → physical log(I) → physical I
        # Enforce Z ≤ 0 here too (I is a fraction ≤ 1) before exp()
        z_vals = jnp.minimum(states[:, 1] * norm_std[1] + norm_mean[1], 0.0)
        I = jnp.exp(z_vals)
        I = jnp.where(jnp.isfinite(I), I, 0.0)
        I = jnp.clip(I, 0.0, 1.0)
        return I

    # Build the vmap-ed JIT simulation ONCE outside the model loop to
    # guarantee JAX compilation cache reuse across ensemble models.
    batch_sim = eqx.filter_jit(jax.vmap(simulate_one_path, in_axes=(None, 0, 0)))

    all_paths = []
    offset = 0
    for model in tqdm(ensemble_models, desc="SIR Forecast"):
        chunk_ic = ic_noise[offset : offset + paths_per_m]
        chunk_pn = path_noise[offset : offset + paths_per_m]
        offset += paths_per_m

        all_paths.append(batch_sim(model, chunk_ic, chunk_pn))

    paths_arr = jnp.concatenate(all_paths, axis=0)  # (N_paths, Horizon)

    start_date = val_features_df.index[start_idx]
    forecast_dates = pd.date_range(
        start=start_date,
        periods=config.FORECAST_HORIZON,
        freq="D",
    )
    return paths_arr, forecast_dates


def run_rolling_validation_sir(
    ensemble_models: List[NeuralSDE],
    data: TrainingData,
    config: Config,
) -> pd.DataFrame:
    """
    SDE-based rolling 1-day-ahead validation with Monte Carlo confidence intervals.

    Vectorised implementation: pre-computes all signatures, then runs a single
    batched JIT call per ensemble model over all validation steps × MC paths.

    Returns a DataFrame: [Date, Actual_I, Pred_I, Lower_90, Upper_90].
    """
    val_df = data.val_features_df
    max_lookback = get_sir_control_lookback(config)
    sig_exts = {
        l: JAXSignatureExtractor(depth=config.SIGNATURE_DEPTH, augment_time=True, lead_lag=config.SIGNATURE_LEAD_LAG)
        for l in config.SIGNATURE_PATH_LENGTHS
    }
    dt, n_sub = config.SDE_SUBSTEP_DT, config.SDE_NUM_SUBSTEPS
    n_mc = config.NUM_ROLLING_MC_PATHS
    _country_idx = jnp.array(
        -1 if data.val_country_id is None else data.val_country_id,
        dtype=jnp.int32,
    )

    val_df_arr = jnp.array(val_df.values)  # (T, 2)
    state_size = val_df_arr.shape[1]
    n_steps = len(val_df) - max_lookback

    if n_steps <= 0:
        return pd.DataFrame()

    # ── Apply AR(1) pre-whitening to the signature-input array (if enabled) ──
    ar_phi = data.ar_phi
    if ar_phi is not None:
        _phi_j = jnp.array(ar_phi)
        sig_input_arr = jnp.concatenate([
            val_df_arr[:1],
            val_df_arr[1:] - _phi_j[jnp.newaxis, :] * val_df_arr[:-1],
        ], axis=0)
        logging.info("Rolling Val: AR(1) pre-whitening applied to signature input.")
    else:
        sig_input_arr = val_df_arr

    # ── 1. Pre-compute all signatures in one vectorised pass ─────────────────
    # For each step t∈[max_lookback, T), extract history[-l:] and compute sig.
    logging.info("Rolling Val: pre-computing signatures …")
    all_sigs_list = []
    for l in config.SIGNATURE_PATH_LENGTHS:
        # Build (n_steps, l, state_size) windows via dynamic_slice + vmap
        offsets = jnp.arange(n_steps)  # step_i → start of history = (max_lookback - l + step_i)

        @jax.jit
        def extract_and_sig(offset):
            start = max_lookback - l + offset
            chunk = jax.lax.dynamic_slice(sig_input_arr, (start, 0), (l, state_size))
            return sig_exts[l]._compute_single(chunk)

        sigs = jax.vmap(extract_and_sig)(offsets)  # (n_steps, sig_dim)
        all_sigs_list.append(sigs)

    all_controls = jnp.concatenate(all_sigs_list, axis=-1)  # (n_steps, total_sig_dim)
    all_controls = apply_control_ablation_array(all_controls, config, seed_offset=4000)

    # Use the same full control builder as training. This replaces the
    # signature-only provisional array above and includes explicit controls.
    control_source_df = apply_temporal_prewhiten_frame(val_df, data.ar_phi)
    control_frame = create_sir_controls(control_source_df, config)
    control_frame = apply_control_ablation_frame(
        control_frame, config, seed_offset=4000
    )
    all_controls = jnp.array(control_frame.iloc[:-1].values)

    # States at each validation step
    all_states = jnp.array(val_df.loc[control_frame.index[:-1]].values)

    # ── 2. Pre-generate all noise ────────────────────────────────────────────
    master_key = jax.random.PRNGKey(config.SEED + 999)
    all_noise = jax.random.normal(
        master_key, shape=(n_steps, n_mc, n_sub, state_size)
    )

    # Physics-aware clip bounds (same as forecast_sir)
    _rv_phys_lo = jnp.array([1e-6, -20.0])
    _rv_phys_hi = jnp.array([1.0 - 1e-6, 0.0])
    _rv_nm = jnp.array(data.norm_mean)
    _rv_ns = jnp.array(data.norm_std)

    def _rv_clip_physical(s):
        phys = s * _rv_ns + _rv_nm
        phys = jnp.clip(phys, _rv_phys_lo, _rv_phys_hi)
        return (phys - _rv_nm) / _rv_ns

    # ── 3. Batched MC simulation per model ───────────────────────────────────
    @eqx.filter_jit
    def predict_all_steps(model, states, controls, noise):
        """
        Vectorised over (n_steps, n_mc).

        states  : (n_steps, state_size)
        controls: (n_steps, sig_dim)
        noise   : (n_steps, n_mc, n_sub, state_size)
        Returns : (n_steps, n_mc, state_size)
        """
        def simulate_one_mc(state, ctrl, mc_noise):
            # mc_noise: (n_sub, state_size)
            def sub(s, dw_unit):
                d = model._calculate_drift(s, ctrl, country_idx=_country_idx)
                diff = model._calculate_diffusion(s, ctrl, country_idx=_country_idx)
                dw = dw_unit * jnp.sqrt(dt)
                return _rv_clip_physical(s + d * dt + diff * dw), None

            final, _ = jax.lax.scan(sub, state, mc_noise)
            return final

        def simulate_step(state, ctrl, step_noise):
            # step_noise: (n_mc, n_sub, state_size)
            return jax.vmap(simulate_one_mc, in_axes=(None, None, 0))(
                state, ctrl, step_noise
            )

        return jax.vmap(simulate_step)(states, controls, noise)

    # Accumulate MC predictions across ensemble models
    all_model_preds = []
    for m in tqdm(ensemble_models, desc="Rolling Val (SDE)"):
        preds = predict_all_steps(m, all_states, all_controls, all_noise)
        all_model_preds.append(preds)  # (n_steps, n_mc, state_size)

    # Pool: (n_steps, n_models * n_mc, state_size)
    pooled = jnp.concatenate(all_model_preds, axis=1)

    # ── 4. Decode and build results ──────────────────────────────────────────
    z_preds = pooled[:, :, 1] * data.norm_std[1] + data.norm_mean[1]  # (n_steps, pool)
    I_preds_phys = jnp.exp(z_preds)

    I_pred_mean = np.array(jnp.mean(I_preds_phys, axis=1))
    I_lower = np.array(jnp.percentile(I_preds_phys, 5, axis=1))
    I_upper = np.array(jnp.percentile(I_preds_phys, 95, axis=1))

    # Actual I: val_df at index [max_lookback, T)
    z_actual = val_df_arr[max_lookback:, 1] * data.norm_std[1] + data.norm_mean[1]
    I_actual = np.array(jnp.exp(z_actual))

    dates = val_df.index[max_lookback:]

    return pd.DataFrame({
        "Date": dates,
        "Actual_I": I_actual,
        "Pred_I": I_pred_mean,
        "Lower_90": I_lower,
        "Upper_90": I_upper,
    })


def summarize_experiment(
    config: Config,
    data: TrainingData,
    actual_I: np.ndarray,
    fc_paths: jnp.ndarray,
    roll_df: pd.DataFrame,
    bi_metrics: Dict[str, float] | None = None,
) -> Dict[str, float]:
    forecast_paths = np.array(fc_paths)
    forecast_median = np.median(forecast_paths, axis=0)
    n_actual = min(len(actual_I), len(forecast_median))

    forecast_mae = np.nan
    forecast_rmse = np.nan
    forecast_smape = np.nan
    if n_actual > 0:
        forecast_err = forecast_median[:n_actual] - actual_I[:n_actual]
        forecast_mae = float(np.mean(np.abs(forecast_err)))
        forecast_rmse = float(np.sqrt(np.mean(forecast_err ** 2)))
        denom = np.abs(forecast_median[:n_actual]) + np.abs(actual_I[:n_actual]) + 1e-8
        forecast_smape = float(np.mean(2.0 * np.abs(forecast_err) / denom))

    rolling_mae = np.nan
    rolling_rmse = np.nan
    rolling_coverage_90 = np.nan
    rolling_width_mean = np.nan
    if not roll_df.empty:
        roll_err = roll_df["Pred_I"].values - roll_df["Actual_I"].values
        rolling_mae = float(np.mean(np.abs(roll_err)))
        rolling_rmse = float(np.sqrt(np.mean(roll_err ** 2)))
        rolling_coverage_90 = float(
            ((roll_df["Actual_I"] >= roll_df["Lower_90"]) & (roll_df["Actual_I"] <= roll_df["Upper_90"])).mean()
        )
        rolling_width_mean = float(np.mean(roll_df["Upper_90"] - roll_df["Lower_90"]))

    bi_metrics = bi_metrics or {}

    return {
        "experiment": get_experiment_tag(config),
        "control_ablation": config.CONTROL_ABLATION,
        "incidence_source": config.SIR_INCIDENCE_SOURCE,
        "incidence_blend_weight": float(config.SIR_INCIDENCE_BLEND_WEIGHT),
        "recovery_days": float(config.SIR_RECOVERY_DAYS),
        "freeze_gamma": bool(config.FREEZE_GAMMA),
        "diffusion_state_dependent": bool(config.DIFFUSION_STATE_DEPENDENT),
        "sigma_country_scale": bool(config.SIGMA_COUNTRY_SCALE),
        "temporal_prewhiten": bool(config.USE_TEMPORAL_PREWHITEN),
        "explicit_recon_controls": config.EXPLICIT_RECON_CONTROLS,
        "explicit_recon_control_window": int(config.EXPLICIT_RECON_CONTROL_WINDOW),
        "zscore_wd_weight": float(config.ZSCORE_WD_WEIGHT),
        "train_paths": int(data.train_set.ys.shape[0]),
        "val_paths": int(data.val_set.ys.shape[0]) if data.val_set else 0,
        "forecast_mae": forecast_mae,
        "forecast_rmse": forecast_rmse,
        "forecast_smape": forecast_smape,
        "rolling_mae": rolling_mae,
        "rolling_rmse": rolling_rmse,
        "rolling_coverage_90": rolling_coverage_90,
        "rolling_width_mean": rolling_width_mean,
        "bi_z_mean": float(bi_metrics.get("bi_z_mean", np.nan)),
        "bi_z_std": float(bi_metrics.get("bi_z_std", np.nan)),
        "bi_z_skew": float(bi_metrics.get("bi_z_skew", np.nan)),
        "bi_z_kurt": float(bi_metrics.get("bi_z_kurt", np.nan)),
        "bi_acf1": float(bi_metrics.get("bi_acf1", np.nan)),
        "bi_acf7": float(bi_metrics.get("bi_acf7", np.nan)),
        "bi_ks_stat": float(bi_metrics.get("bi_ks_stat", np.nan)),
        "bi_ks_pvalue": float(bi_metrics.get("bi_ks_pvalue", np.nan)),
        "bi_lb_pvalue": float(bi_metrics.get("bi_lb_pvalue", np.nan)),
        "bi_max_w_norm": float(bi_metrics.get("bi_max_w_norm", np.nan)),
    }


def save_experiment_summary(summary: Dict[str, float], config: Config) -> Path:
    summary_df = pd.DataFrame([summary])
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.SUMMARY_DIR / f"summary_{get_experiment_tag(config)}.csv"
    summary_df.to_csv(save_path, index=False)
    logging.info("Saved experiment summary to %s", save_path)
    return save_path


def save_brownian_inversion_metrics(metrics: Dict[str, object], config: Config) -> Path:
    metrics_df = pd.DataFrame([metrics])
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.SUMMARY_DIR / f"bi_metrics_{get_experiment_tag(config)}.csv"
    metrics_df.to_csv(save_path, index=False)
    logging.info("Saved BI metrics to %s", save_path)
    return save_path


def evaluate_experiment(ensemble_models, data: TrainingData, config: Config, key):
    """Forecast, evaluate and write the experiment tables and figures."""
    # ── SIR forecast spaghetti ───────────────────────────────────────────────
    fc_key, key = jax.random.split(key)
    fc_paths, fc_dates = forecast_sir(
        ensemble_models,
        data.val_features_df,
        data.norm_mean,
        data.norm_std,
        config,
        fc_key,
        ar_phi=data.ar_phi,
        country_idx=data.val_country_id,
    )
    # Actual I for the same window (decoded from normalised physical I)
    val_I_norm = data.val_features_df["I"].values
    start_pos = max(len(val_I_norm) // 2, max(config.SIGNATURE_PATH_LENGTHS))

    # actual_I needs correct decoding from Log space
    # The models path 'fc_paths' is already physical I.
    # We must decode the validation data validation slice similarly.

    # 1. Un-normalize Z
    actual_z_slice = (
        val_I_norm[start_pos : start_pos + config.FORECAST_HORIZON] * data.norm_std[1]
        + data.norm_mean[1]
    )

    # 2. Exponentiate to get physical I
    actual_I = np.exp(actual_z_slice)

    plot_sir_forecast(fc_paths, fc_dates, actual_I, config)

    # ── Parameter decomposition: constant baseline vs neural residual ────────
    plot_sir_parameters(ensemble_models[0], data, config)

    # ── Rolling validation ────────────────────────────────────────────────────
    roll_df = run_rolling_validation_sir(ensemble_models, data, config)
    if not roll_df.empty:
        plot_rolling_validation_sir(roll_df, config)

    # ── Brownian inversion diagnostic ────────────────────────────────────────
    bi_key, key = jax.random.split(key)
    bi_start_date = str(data.val_features_df.index[start_pos].date())
    bi_residuals = _evaluate_bi_residuals(
        ensemble_models[0], bi_start_date, data, config
    )
    bi_metrics = (
        compute_brownian_inversion_metrics_from_residuals(bi_residuals)
        if bi_residuals is not None else None
    )
    if bi_metrics is not None:
        save_brownian_inversion_metrics(
            {
                "experiment": get_experiment_tag(config),
                "val_country": config.VAL_COUNTRY,
                "start_date": bi_start_date,
                "incidence_source": config.SIR_INCIDENCE_SOURCE,
                "incidence_blend_weight": float(config.SIR_INCIDENCE_BLEND_WEIGHT),
                **bi_metrics,
            },
            config,
        )
    if bi_residuals is not None:
        plot_brownian_inversion_validation(
            ensemble_models[0], bi_start_date, data, config, bi_key,
            residuals_raw=bi_residuals,
        )

    summary = summarize_experiment(
        config,
        data,
        actual_I,
        fc_paths,
        roll_df,
        bi_metrics=bi_metrics,
    )
    save_experiment_summary(summary, config)
    logging.info("Experiment summary: %s", summary)
    return summary
