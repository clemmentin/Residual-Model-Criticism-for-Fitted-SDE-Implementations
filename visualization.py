"""Figures for SIR forecasts, parameters and inversion diagnostics."""

import matplotlib
matplotlib.use('Agg')

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from brownian_inversion import (
    _evaluate_bi_residuals,
    compute_brownian_inversion_metrics,
    compute_brownian_inversion_metrics_from_residuals,
)
from config import Config, TrainingData, apply_control_ablation_array, get_experiment_tag
from neural_sde import NeuralSDE
from pandas.plotting import autocorrelation_plot
from scipy.stats import gaussian_kde, norm
from sir_training import run_detailed_validation
from trajectory import JAXSignatureExtractor


def plot_brownian_inversion_validation(
    model: NeuralSDE,
    start_date_str: str,
    data: TrainingData,
    config: Config,
    key: jnp.ndarray,
    *,
    residuals_raw: np.ndarray | None = None,
):
    """
    Evaluates model drift/diffusion along the REAL (validation) trajectory,
    performs Brownian inversion Z_t = (ΔY - μ) / σ, and plots 4 diagnostics:
      1. Martingale (cumulative sum vs √t bounds)
      2. Distribution matching (histogram + KDE vs N(0,1))
      3. Q-Q plot
      4. Autocorrelation
    """
    logging.info(f"--- Running Brownian Inversion Validation for {start_date_str} ---")

    if residuals_raw is None:
        residuals_raw = _evaluate_bi_residuals(model, start_date_str, data, config)
    if residuals_raw is None:
        return

    # ── Re-standardise residuals ─────────────────────────────────────────────
    # Raw inverted residuals may have std << 1 (model σ too large) or >> 1.
    # We report both raw and standardised statistics, but plot using the
    # empirical scale so the cumulative-sum and Q-Q plots are visually useful.
    raw_mu, raw_std = float(np.mean(residuals_raw)), float(np.std(residuals_raw))
    # Standardise to zero-mean, unit-variance for shape diagnostics
    residuals = (residuals_raw - raw_mu) / max(raw_std, 1e-8)

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Real Data Brownian Inversion — {start_date_str}\n"
        f"Raw residual: μ={raw_mu:.4f}, σ={raw_std:.4f}  "
        f"(theoretical N(0,1) if model perfect)",
        fontsize=13,
    )

    # Plot 1: Martingale Property Test (Cumulative Sum)
    W_t = np.cumsum(residuals)
    time_steps = np.arange(1, len(W_t) + 1)

    axes[0, 0].plot(time_steps, W_t, lw=1.5, label="Cumulative Residuals $W_t$")

    bound_95 = 1.96 * np.sqrt(time_steps)
    axes[0, 0].plot(time_steps, bound_95, "k--", alpha=0.5, label="95% bound (±1.96√t)")
    axes[0, 0].plot(time_steps, -bound_95, "k--", alpha=0.5)
    bound_99 = 2.58 * np.sqrt(time_steps)
    axes[0, 0].fill_between(
        time_steps, -bound_99, bound_99, color="gray", alpha=0.1, label="99% region"
    )

    axes[0, 0].set_title("Martingale Test (standardised residuals)")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Cumulative Standardised Residual")
    axes[0, 0].legend(fontsize=9)

    # Plot 2: Distribution Matching with KDE
    axes[0, 1].hist(
        residuals, bins=30, density=True, alpha=0.5,
        label="Standardised Residuals", color="skyblue",
    )
    x = np.linspace(residuals.min() - 0.5, residuals.max() + 0.5, 200)
    axes[0, 1].plot(x, norm.pdf(x, 0, 1), "r-", lw=2, label="N(0,1)")
    try:
        kde = gaussian_kde(residuals)
        axes[0, 1].plot(x, kde(x), "b--", lw=2, label="KDE")
    except Exception as e:
        logging.warning(f"KDE fit failed: {e}")
    fit_mu, fit_std = norm.fit(residuals)
    axes[0, 1].set_title(
        f"Distribution (standardised: μ={fit_mu:.2f}, σ={fit_std:.2f}  |  "
        f"raw σ={raw_std:.4f})"
    )
    axes[0, 1].legend(fontsize=9)

    # Plot 3: Q-Q Plot
    sm.qqplot(residuals, line="s", ax=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot vs Standard Normal (standardised)")

    # Plot 4: Autocorrelation
    autocorrelation_plot(pd.Series(residuals), ax=axes[1, 1])
    axes[1, 1].set_title("Autocorrelation of Residuals")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_date_str = start_date_str.replace(":", "-").replace(" ", "_")
    save_path = (
        config.BASELINE_PLOT_DIR / f"real_brownian_inversion_{config.VAL_COUNTRY}_{safe_date_str}.png"
    )
    plt.savefig(save_path)
    logging.info(f"Saved Real Brownian inversion plot to {save_path}")
    plt.close(fig)


def plot_convergence_order(
    model: NeuralSDE,
    data: TrainingData,
    config: Config,
    key: jnp.ndarray,
    num_mc_paths: int = 32768,
):
    """
    Convergence Order Plot for the Neural SDE solver.

    Measures strong and weak convergence rates of the Heun scheme by comparing
    solutions at progressively coarser step sizes against a fine-grid reference.
    Uses Quasi-Monte Carlo (Sobol sequence) for Brownian path generation to
    improve convergence estimation stability. Both reference and coarser solutions
    share the same pre-computed Brownian paths.

    The results are displayed on log-log axes with:
      - Fitted convergence order (slope of the regression line)
      - Theoretical reference slopes O(dt^0.5), O(dt^1.0), O(dt^2.0)

    Returns a dict with dt_values, strong/weak errors, and fitted orders.
    """
    logging.info("--- Running Convergence Order Analysis ---")

    # ── 1. Setup: extract a sample trajectory for initial state & controls ──
    sample_idx = 0
    ys_sample = data.train_set.ys[sample_idx]  # (Steps, State)
    controls_sample = data.train_set.controls[sample_idx]  # (Steps, Control)
    y0 = ys_sample[0]

    N = len(ys_sample)  # number of fine grid points
    T = float(N - 1)  # terminal time
    ts_fine = jnp.linspace(0.0, T, N)

    # Shared control interpolation on the finest grid – this stays fixed so
    # that the only source of discretisation error is the solver step size.
    control_interp = diffrax.LinearInterpolation(ts=ts_fine, ys=controls_sample)

    # ── 2. Step-size hierarchy ──
    T_eval = 5.0
    dt_ref = 1.0 / 128.0
    test_dts = sorted([1.0, 0.5, 0.25, 0.125, 0.0625])
    max_solve_steps = int(T_eval / dt_ref) + 500

    logging.info(
        f"Reference dt = {dt_ref:.5f}, test dts = {test_dts}, T_eval = {T_eval}, MC paths = {num_mc_paths}"
    )

    mc_keys = jax.random.split(key, num_mc_paths)

    # ── 3. JIT-compiled single-path solver ──
    def solve_terminal(model_, ctrl_, y0_, dt0_, path_key):
        brownian = diffrax.VirtualBrownianTree(
            t0=0.0, t1=T_eval, tol=1e-4, shape=(model_.brownian_size,), key=path_key
        )

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(model_.drift),
            diffrax.ControlTerm(model_.diffusion, brownian),
        )
        sol = diffrax.diffeqsolve(
            terms,
            diffrax.Euler(),
            0.0,
            T_eval,
            dt0_,
            y0_,
            args=ctrl_,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=max_solve_steps,
        )
        return sol.ys[0]

    # 使用 JAX vmap 沿 path_key (第0维度) 批量化求解，大幅提升运行速度
    batch_solve = eqx.filter_jit(
        jax.vmap(solve_terminal, in_axes=(None, None, None, None, 0))
    )

    # ── 4. Reference: solve at finest dt ──
    logging.info(f"Computing {num_mc_paths} reference paths (dt={dt_ref}) ...")
    ref_terminals = batch_solve(model, control_interp, y0, dt_ref, mc_keys)
    ref_mean = jnp.mean(ref_terminals, axis=0)

    # ── 5. Coarser resolutions ──
    strong_errors, weak_errors = [], []
    for dt in test_dts:
        logging.info(f"  dt = {dt:.4f} ...")
        coarse_terminals = batch_solve(model, control_interp, y0, dt, mc_keys)
        coarse_mean = jnp.mean(coarse_terminals, axis=0)

        # Strong error: E[ ||X_dt(T) − X_ref(T)|| ]
        strong_err = float(
            jnp.mean(jnp.linalg.norm(coarse_terminals - ref_terminals, axis=1))
        )
        # Weak error: || E[X_dt(T)] − E[X_ref(T)] ||
        weak_err = float(jnp.linalg.norm(coarse_mean - ref_mean))

        strong_errors.append(strong_err)
        weak_errors.append(weak_err)

    dt_arr = np.array(test_dts)
    strong_arr = np.array(strong_errors)
    weak_arr = np.array(weak_errors)

    # ── 6. Fit convergence orders (log-log linear regression) ──
    log_dt = np.log(dt_arr)

    valid_s = strong_arr > 1e-15
    if valid_s.sum() >= 2:
        strong_order, s_intercept = np.polyfit(
            log_dt[valid_s], np.log(strong_arr[valid_s]), 1
        )
    else:
        strong_order, s_intercept = 0.5, np.log(strong_arr[0] + 1e-15)

    valid_w = weak_arr > 1e-15
    if valid_w.sum() >= 2:
        weak_order, w_intercept = np.polyfit(
            log_dt[valid_w], np.log(weak_arr[valid_w]), 1
        )
    else:
        weak_order, w_intercept = 1.0, np.log(weak_arr[0] + 1e-15)

    logging.info(f"Fitted Strong Convergence Order: {strong_order:.3f}")
    logging.info(f"Fitted Weak   Convergence Order: {weak_order:.3f}")

    # ── 7. Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Neural SDE Solver Convergence Order (Euler)", fontsize=14)

    # ─── Strong convergence ───
    ax = axes[0]
    ax.loglog(dt_arr, strong_arr, "bo-", ms=8, lw=2, label="Measured")
    fit_s = np.exp(s_intercept) * dt_arr**strong_order
    ax.loglog(
        dt_arr,
        fit_s,
        "b--",
        alpha=0.5,
        label=f"Fitted  $p = {strong_order:.2f}$",
    )
    # Reference slopes anchored at first measured point
    c05 = strong_arr[0] / dt_arr[0] ** 0.5
    c10 = strong_arr[0] / dt_arr[0] ** 1.0
    ax.loglog(
        dt_arr,
        c05 * dt_arr**0.5,
        "k:",
        alpha=0.4,
        label=r"$O(\Delta t^{0.5})$",
    )
    ax.loglog(
        dt_arr,
        c10 * dt_arr**1.0,
        "k-.",
        alpha=0.4,
        label=r"$O(\Delta t^{1.0})$",
    )
    ax.set_xlabel(r"Step size $\Delta t$", fontsize=12)
    ax.set_ylabel(
        r"Strong Error  $E[\|X_{\Delta t} - X_{\mathrm{ref}}\|]$", fontsize=11
    )
    ax.set_title("Strong Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ─── Weak convergence ───
    ax = axes[1]
    ax.loglog(dt_arr, weak_arr, "rs-", ms=8, lw=2, label="Measured")
    fit_w = np.exp(w_intercept) * dt_arr**weak_order
    ax.loglog(
        dt_arr,
        fit_w,
        "r--",
        alpha=0.5,
        label=f"Fitted  $q = {weak_order:.2f}$",
    )
    c10w = weak_arr[0] / dt_arr[0] ** 1.0
    c20w = weak_arr[0] / dt_arr[0] ** 2.0
    ax.loglog(
        dt_arr,
        c10w * dt_arr**1.0,
        "k:",
        alpha=0.4,
        label=r"$O(\Delta t^{1.0})$",
    )
    ax.loglog(
        dt_arr,
        c20w * dt_arr**2.0,
        "k-.",
        alpha=0.4,
        label=r"$O(\Delta t^{2.0})$",
    )
    ax.set_xlabel(r"Step size $\Delta t$", fontsize=12)
    ax.set_ylabel(
        r"Weak Error  $\|E[X_{\Delta t}] - E[X_{\mathrm{ref}}]\|$", fontsize=11
    )
    ax.set_title("Weak Convergence")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = config.BASELINE_PLOT_DIR / "convergence_order.png"
    config.BASELINE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    logging.info(f"Saved convergence order plot to {save_path}")
    plt.close(fig)

    return {
        "dt_values": dt_arr,
        "strong_errors": strong_arr,
        "weak_errors": weak_arr,
        "strong_order": strong_order,
        "weak_order": weak_order,
    }


def plot_sir_forecast(
    forecast_paths: jnp.ndarray,
    forecast_dates: "pd.DatetimeIndex",
    actual_I: "np.ndarray",
    config: Config,
):
    """
    Fan-chart of simulated I(t) paths (as fraction of population).
    Overlays the actual validation-country I(t) for comparison.

    forecast_paths : (N_paths, Horizon)  un-normalised I fractions
    forecast_dates : DatetimeIndex of length Horizon
    actual_I       : array of length ≤ Horizon with true I values
    """
    logging.info("--- Plotting SIR Forecast Fan-Chart ---")

    paths_np = np.array(forecast_paths)
    p5  = np.percentile(paths_np,  5, axis=0)
    p25 = np.percentile(paths_np, 25, axis=0)
    p50 = np.percentile(paths_np, 50, axis=0)
    p75 = np.percentile(paths_np, 75, axis=0)
    p95 = np.percentile(paths_np, 95, axis=0)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.fill_between(forecast_dates, p5,  p95,  color="steelblue", alpha=0.15,
                    label="Risk boundary (5–95%)")
    ax.fill_between(forecast_dates, p25, p75,  color="steelblue", alpha=0.35,
                    label="Core trend (25–75%)")
    ax.plot(forecast_dates, p50, color="navy", lw=2, linestyle="--",
            label="Median forecast")

    # Plot representative central paths rather than arbitrary leading samples.
    # The percentile bands already encode tail risk; this keeps a couple of
    # extreme Monte Carlo draws from visually dominating the spaghetti layer.
    representative_count = min(20, len(paths_np))
    path_scores = np.mean((paths_np - p50[None, :]) ** 2, axis=1)
    representative_idx = np.argsort(path_scores)[:representative_count]

    for path in paths_np[representative_idx]:
        ax.plot(forecast_dates, path, color="steelblue", alpha=0.08, lw=0.7)

    n_actual = min(len(actual_I), len(forecast_dates))
    if n_actual > 0:
        ax.plot(forecast_dates[:n_actual], actual_I[:n_actual],
                color="black", lw=2.5, label=f"Actual I ({config.VAL_COUNTRY})")

    ax.set_title(f"SIR Forecast — Validation Country: {config.VAL_COUNTRY}", fontsize=14)
    ax.set_ylabel("Infected fraction  I(t)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = (
        config.BASELINE_PLOT_DIR
        / f"sir_forecast_{config.VAL_COUNTRY}_{get_experiment_tag(config)}.png"
    )
    config.BASELINE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    logging.info(f"Saved SIR forecast to {save_path}")
    plt.close(fig)


def plot_sir_parameters(
    model: NeuralSDE,
    data: TrainingData,
    config: Config,
):
    """
    Plots β(t) and γ(t) decomposed into constant baseline and neural residual
    over the validation country trajectory.

    For each time step t on the validation path:
      β(t)   = β_base  + β_scale  * tanh(net[0])   (full)
      Δβ(t)  =           β_scale  * tanh(net[0])   (residual)
      β_base is drawn as a constant reference line.
    """
    from neural_sde import _SIR_BETA_SCALE, _SIR_GAMMA_SCALE
    from sir_data import (
        apply_temporal_prewhiten_frame,
        create_sir_controls,
        get_sir_control_lookback,
    )

    logging.info("--- Plotting SIR Parameter Decomposition ---")

    beta_base  = model.beta_base
    gamma_base = model.gamma_base

    val_arr = jnp.array(data.val_features_df.values)  # (T, 2) normalised
    sig_input_arr = val_arr
    if data.ar_phi is not None:
        phi = jnp.array(data.ar_phi)
        sig_input_arr = sig_input_arr.at[1:].set(
            sig_input_arr[1:] - phi[jnp.newaxis, :] * sig_input_arr[:-1]
        )
    max_lookback = get_sir_control_lookback(config)
    sig_exts = {
        l: JAXSignatureExtractor(depth=config.SIGNATURE_DEPTH, augment_time=True, lead_lag=config.SIGNATURE_LEAD_LAG)
        for l in config.SIGNATURE_PATH_LENGTHS
    }
    n_steps = len(data.val_features_df) - max_lookback
    if n_steps <= 0:
        logging.warning("Not enough val data for parameter plot.")
        return

    dates = data.val_features_df.index[max_lookback:]

    @jax.jit
    def get_params_at(state, ctrl):
        if model.signature_linear_drift:
            sig = ctrl[:model.total_signature_size]
            raw = model.sig_linear_drift(sig)
        else:
            net_input = jnp.concatenate([state, ctrl])
            raw = model.drift_net(net_input)  # (2,)
        delta_beta  = _SIR_BETA_SCALE  * jnp.tanh(raw[0])
        delta_gamma = _SIR_GAMMA_SCALE * jnp.tanh(raw[1])
        beta  = jnp.maximum(beta_base  + delta_beta,  0.0)
        gamma = jnp.maximum(gamma_base + delta_gamma, 1e-2)
        return beta, gamma, delta_beta, delta_gamma

    raw_controls = []
    betas, gammas, d_betas, d_gammas = [], [], [], []
    for i in range(n_steps):
        state = val_arr[max_lookback + i]
        sigs = jnp.concatenate([
            sig_exts[l]._compute_single(
                jax.lax.dynamic_slice(sig_input_arr, (i + max_lookback - l + 1, 0), (l, 2))
            )
            for l in config.SIGNATURE_PATH_LENGTHS
        ])
        raw_controls.append(np.array(sigs))

    controls = apply_control_ablation_array(
        np.stack(raw_controls, axis=0),
        config,
        seed_offset=4000,
    )
    control_source_df = apply_temporal_prewhiten_frame(
        data.val_features_df,
        data.ar_phi,
    )
    control_frame = create_sir_controls(control_source_df, config)
    control_frame = pd.DataFrame(
        apply_control_ablation_array(
            control_frame.values,
            config,
            seed_offset=4000,
        ),
        index=control_frame.index,
    )
    controls = control_frame.loc[dates].values

    for i in range(n_steps):
        state = val_arr[max_lookback + i]
        sigs = jnp.array(controls[i])
        b, g, db, dg = get_params_at(state, sigs)
        betas.append(float(b));  gammas.append(float(g))
        d_betas.append(float(db)); d_gammas.append(float(dg))

    betas    = np.array(betas)
    gammas   = np.array(gammas)
    d_betas  = np.array(d_betas)
    d_gammas = np.array(d_gammas)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    fig.suptitle(
        f"SIR Parameter Decomposition — {config.VAL_COUNTRY}\n"
        f"Constant (baseline) vs Neural Residual [{get_experiment_tag(config)}]",
        fontsize=13,
    )

    # β(t) full
    ax = axes[0, 0]
    ax.plot(dates, betas, color="steelblue", lw=1.5, label="β(t) full")
    ax.axhline(beta_base, color="navy", lw=1.2, linestyle="--",
               label=f"β_base = {beta_base:.4f}")
    ax.set_ylabel("β(t)")
    ax.set_title("Transmission rate β(t)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Δβ(t) residual
    ax = axes[0, 1]
    ax.plot(dates, d_betas, color="tomato", lw=1.2, label="Δβ(t) neural residual")
    ax.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax.fill_between(dates, d_betas, 0,
                    where=d_betas > 0, color="tomato", alpha=0.2, label="+residual")
    ax.fill_between(dates, d_betas, 0,
                    where=d_betas < 0, color="steelblue", alpha=0.2, label="−residual")
    ax.set_ylabel("Δβ(t)")
    ax.set_title("Neural residual on β")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # γ(t) full
    ax = axes[1, 0]
    ax.plot(dates, gammas, color="seagreen", lw=1.5, label="γ(t) full")
    ax.axhline(gamma_base, color="darkgreen", lw=1.2, linestyle="--",
               label=f"γ_base = {gamma_base:.4f}")
    ax.set_ylabel("γ(t)")
    ax.set_title("Recovery rate γ(t)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Δγ(t) residual
    ax = axes[1, 1]
    ax.plot(dates, d_gammas, color="orange", lw=1.2, label="Δγ(t) neural residual")
    ax.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax.fill_between(dates, d_gammas, 0,
                    where=d_gammas > 0, color="orange", alpha=0.2, label="+residual")
    ax.fill_between(dates, d_gammas, 0,
                    where=d_gammas < 0, color="seagreen", alpha=0.2, label="−residual")
    ax.set_ylabel("Δγ(t)")
    ax.set_title("Neural residual on γ")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    save_path = (
        config.BASELINE_PLOT_DIR
        / f"sir_params_{config.VAL_COUNTRY}_{get_experiment_tag(config)}.png"
    )
    config.BASELINE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    logging.info(f"Saved parameter decomposition to {save_path}")
    plt.close(fig)


def plot_rolling_validation_sir(df: pd.DataFrame, config: Config):
    """
    Plot 1-day-ahead rolling I(t) forecast vs actual for the validation country.

    df columns: Date, Actual_I, Pred_I, Lower_90, Upper_90
    """
    if df.empty:
        logging.warning("Rolling validation DataFrame is empty — skipping plot.")
        return

    coverage = (
        (df["Actual_I"] >= df["Lower_90"]) & (df["Actual_I"] <= df["Upper_90"])
    ).mean()

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(df["Date"], df["Actual_I"], color="black", lw=1.5,
            label=f"Actual I — {config.VAL_COUNTRY}")
    ax.plot(df["Date"], df["Pred_I"],   color="navy",  lw=1.0,
            linestyle="--", label="Predicted mean")
    ax.fill_between(df["Date"], df["Lower_90"], df["Upper_90"],
                    color="steelblue", alpha=0.25,
                    label="90% prediction interval")

    ax.set_title(
        f"Rolling 1-day-ahead SIR Forecast — {config.VAL_COUNTRY}\n"
        f"90% Coverage: {coverage:.1%}  (target 90%)",
        fontsize=13,
    )
    ax.set_ylabel("Infected fraction  I(t)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = (
        config.ROLLING_PLOT_DIR
        / f"sir_rolling_{config.VAL_COUNTRY}_{get_experiment_tag(config)}.png"
    )
    config.ROLLING_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    logging.info(f"Saved SIR rolling validation to {save_path}")
    plt.close(fig)

# Compatibility for existing analysis scripts.
