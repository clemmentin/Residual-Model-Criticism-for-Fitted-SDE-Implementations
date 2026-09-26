"""OWID reconstruction, causal controls and cached training datasets."""

import jax
import jax.numpy as jnp
import logging
import numpy as np
import pandas as pd
import pickle
import requests

from config import (
    Config,
    TrainingData,
    TrainingDataSet,
    _validate_incidence_source,
    _validate_recon_control_mode,
    apply_control_ablation_frame,
    get_cache_path,
    get_recon_control_size,
)
from scipy.optimize import curve_fit as scipy_curve_fit
from trajectory import JAXSignatureExtractor, get_signature_size
from typing import Dict, List, Tuple


OWID_MIRROR = (
    "https://raw.githubusercontent.com/owid/covid-19-data"
    "/master/public/data/owid-covid-data.csv"
)

OWID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

def download_owid_data(config: Config, force_download: bool = False) -> pd.DataFrame:
    """Download (or load from cache) the full OWID COVID-19 dataset."""
    cache = config.OWID_CACHE_PATH
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not force_download:
        logging.info(f"Loading OWID data from cache: {cache}")
        return pd.read_csv(cache, parse_dates=["date"], low_memory=False)

    for url in (OWID_URL, OWID_MIRROR):
        logging.info(f"Trying to download OWID data from: {url}")
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            temp_cache = cache.with_suffix(cache.suffix + ".tmp")
            temp_cache.write_bytes(resp.content)
            temp_cache.replace(cache)
            logging.info(f"Download successful, cached to: {cache}")
            return pd.read_csv(cache, parse_dates=["date"], low_memory=False)
        except Exception as e:
            logging.warning(f"Download failed ({e}). Trying next URL…")

    raise RuntimeError(
        f"Cannot download OWID data from any URL.\n"
        f"Please manually download the CSV and place it at:\n"
        f"  {cache.resolve()}\n"
        f"Download from: {OWID_MIRROR}"
    )


def _weekday_adjust_incidence(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weekdays = pd.to_datetime(dates).dt.weekday.to_numpy()
    positive_mean = float(np.mean(values[values > 0])) if np.any(values > 0) else 0.0
    if positive_mean <= 0.0:
        return values.copy()

    factors = np.ones(7, dtype=np.float64)
    for weekday in range(7):
        mask = weekdays == weekday
        if np.any(mask):
            factors[weekday] = max(float(np.mean(values[mask])) / positive_mean, 1e-3)

    adjusted = values / factors[weekdays]
    raw_total = float(values.sum())
    adjusted_total = float(adjusted.sum())
    if adjusted_total > 1e-12:
        adjusted *= raw_total / adjusted_total
    return np.clip(adjusted, 0.0, None)


def _select_incidence_array(
    cdf: pd.DataFrame,
    incidence_source: str,
    blend_weight: float = 0.0,
) -> np.ndarray:
    incidence_source = _validate_incidence_source(incidence_source)
    smoothed = cdf["new_cases_smoothed"].fillna(0).clip(lower=0).values.astype(np.float64)
    if incidence_source == "smoothed":
        return smoothed

    raw = cdf["new_cases"].fillna(0).clip(lower=0).values.astype(np.float64)
    if incidence_source == "raw":
        return raw
    if incidence_source == "weekday_adjusted":
        return _weekday_adjust_incidence(raw, cdf["date"])
    if incidence_source == "blend_raw_smoothed":
        blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
        return (1.0 - blend_weight) * smoothed + blend_weight * raw
    raise AssertionError(f"Unhandled incidence source: {incidence_source}")


def prepare_sir_countries(
    raw_df: pd.DataFrame,
    countries: List[str],
    start_date: str,
    recovery_days: float = 14.0,
    active_threshold: float = 1e-5,
    incidence_source: str = "smoothed",
    incidence_blend_weight: float = 0.0,
    trim_to_active_period: bool = True,
    anchor_cutoff: str | pd.Timestamp | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    For each country ISO code, extract daily physical [S, I] fractions:

        S(t) = (population − cumulative_cases) / population
        I(t) = exponentially_decayed_active_cases / population

    where active cases are estimated via the recursive formula:
        active[t] = new_cases_smoothed[t] + alpha * active[t-1]
        alpha = exp(-1 / recovery_days)   (e.g. 14-day infectious period)

    By default, keep the contiguous active-period block. Fixed-calendar
    holdouts use ``trim_to_active_period=False`` to retain the full series
    and ``anchor_cutoff`` to restrict the cumulative-case anchor to data
    available by the evaluation start.

    Returns {iso_code → DataFrame(index=date, columns=[S, I])}.
    """
    start = pd.to_datetime(start_date)
    alpha = float(np.exp(-1.0 / recovery_days))
    result: Dict[str, pd.DataFrame] = {}

    for iso in countries:
        cdf = raw_df[raw_df["iso_code"] == iso].copy()
        cdf = cdf[cdf["date"] >= start].sort_values("date")

        pop_vals = cdf["population"].dropna()
        if pop_vals.empty:
            logging.warning(f"{iso}: no population data — skipping.")
            continue
        pop = float(pop_vals.iloc[0])

        total_cases_arr = (
            cdf["total_cases"].fillna(0).clip(lower=0).values.astype(np.float64)
        )
        new_cases_arr = _select_incidence_array(
            cdf,
            incidence_source,
            blend_weight=incidence_blend_weight,
        )

        # ── Reconstruct daily cumulative cases from the chosen incidence series
        # total_cases in OWID updates weekly (step function), breaking
        # the continuous SIR assumption dS/dt = -β·S·I.
        # Reconstruct daily S using cumsum of the selected incidence series,
        # anchored to the first available total_cases value.
        smooth_cumcases = np.cumsum(new_cases_arr)
        # Fixed-calendar holdouts must not choose an anchor after their start.
        anchor_mask = total_cases_arr > 0
        if anchor_cutoff is not None:
            anchor_mask &= (
                pd.to_datetime(cdf["date"]).to_numpy()
                <= pd.to_datetime(anchor_cutoff).to_datetime64()
            )
        if not np.any(anchor_mask):
            if anchor_cutoff is not None:
                logging.warning(f"{iso}: no positive total_cases by anchor cutoff — skipping.")
                continue
            anchor_idx = 0
        else:
            anchor_idx = int(np.flatnonzero(anchor_mask)[0])
        offset = total_cases_arr[anchor_idx] - smooth_cumcases[anchor_idx]
        smooth_cumcases = np.clip(smooth_cumcases + offset, 0.0, None)

        # ── Exponential-decay active cases ────────────────────────────────────
        # active[t] = new_cases[t] + alpha * active[t-1]
        # Each historical day's contribution decays as exp(-k/T_rec).
        active = np.zeros(len(new_cases_arr), dtype=np.float64)
        for t in range(len(new_cases_arr)):
            active[t] = new_cases_arr[t] + alpha * (active[t - 1] if t > 0 else 0.0)

        I_frac = active / pop  # active infected fraction

        # ── Active-period filter ───────────────────────────────────────────────
        # Keep only the contiguous window [first_active_day, last_active_day].
        if trim_to_active_period:
            active_mask = I_frac > active_threshold
            if not active_mask.any():
                logging.warning(
                    f"{iso}: no active epidemic period (threshold={active_threshold}) — skipping."
                )
                continue
            idx_on = np.where(active_mask)[0]
            t0, t1 = int(idx_on[0]), int(idx_on[-1]) + 1
        else:
            t0, t1 = 0, len(cdf)

        cdf_sl = cdf.iloc[t0:t1]
        smooth_cumcases_sl = smooth_cumcases[t0:t1]
        I_sl = I_frac[t0:t1]

        S_sl = np.clip((pop - smooth_cumcases_sl) / pop, 1e-6, 1.0 - 1e-6)
        I_sl = np.clip(I_sl, 1e-8, 1.0 - 1e-8)

        df = pd.DataFrame(
            {"S": S_sl, "I": I_sl},
            index=pd.to_datetime(cdf_sl["date"].values),
        )
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < 60:
            logging.warning(f"{iso}: only {len(df)} rows after filtering — skipping.")
            continue

        result[iso] = df

        i_med = float(np.median(I_sl))
        logging.info(
            f"  {iso}: {len(df)} rows  "
            f"({df.index[0].date()} – {df.index[-1].date()})  "
            f"I_active median={i_med:.2e}  (α={alpha:.4f} T_rec={recovery_days}d)"
        )

    return result


def compute_norm_stats(
    country_data: Dict[str, pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray]:
    """Global z-score stats from all supplied (training) country DataFrames."""
    all_data = pd.concat(list(country_data.values()))
    state_data = all_data[["S", "I"]]

    mean = state_data.mean().values.astype(np.float32)
    std = np.clip(state_data.std().values.astype(np.float32), 1e-8, None)
    return mean, std


def fit_sir_baseline(
    country_data: Dict[str, pd.DataFrame],
) -> Tuple[float, float]:
    """
    Fit globally optimal constant SIR parameters β̄, γ̄ via nonlinear least
    squares (scipy.optimize.curve_fit) on all training-country trajectories.

    The discrete-time SIR model (dt = 1 day) gives:
        ΔS(t) = −β · S(t) · I(t)
        ΔI(t) =  (β · S(t) − γ) · I(t)

    Both equations are stacked into a single NLS problem.
    Returns (beta_hat, gamma_hat) clipped to safe physical ranges.
    """
    S_all, I_all, dS_all, dI_all = [], [], [], []

    for iso, df in country_data.items():
        # Baseline fit uses physical units.
        # Note: If 'I' has been log-transformed in place elsewhere, this function will break.
        # But this function is called BEFORE log-transform in 'main'.
        # Wait, inside main:
        #   1. prepare_sir_countries (physical S, I)
        #   2. fit_sir_baseline (physical S, I) - OK
        #   3. Log Transform in place - OK

        S = df["S"].values.astype(np.float64)
        I = df["I"].values.astype(np.float64)
        if len(S) < 2:
            continue
        S_all.append(S[:-1])
        I_all.append(I[:-1])
        dS_all.append(np.diff(S))
        dI_all.append(np.diff(I))

    if not S_all:
        logging.warning("fit_sir_baseline: no usable data, returning defaults.")
        return 0.30, 0.07

    S_arr = np.concatenate(S_all)
    I_arr = np.concatenate(I_all)
    dS_arr = np.concatenate(dS_all)
    dI_arr = np.concatenate(dI_all)

    # X = (2, N) feature matrix;  Y = [dS..., dI...] concatenated
    X = np.stack([S_arr, I_arr], axis=0)
    Y = np.concatenate([dS_arr, dI_arr])

    def sir_model(X: np.ndarray, beta: float, gamma: float) -> np.ndarray:
        S, I = X
        return np.concatenate([-beta * S * I, (beta * S - gamma) * I])

    try:
        params, _ = scipy_curve_fit(
            sir_model,
            X,
            Y,
            p0=[0.30, 0.07],
            bounds=([0.01, 0.005], [2.0, 1.0]),
            maxfev=20_000,
        )
        beta_hat, gamma_hat = float(params[0]), float(params[1])
    except RuntimeError as exc:
        logging.warning(f"curve_fit did not converge ({exc}), using p0 defaults.")
        beta_hat, gamma_hat = 0.30, 0.07

    beta_hat = float(np.clip(beta_hat, 0.01, 2.0))
    gamma_hat = float(np.clip(gamma_hat, 0.005, 1.0))

    logging.info(
        f"NLS SIR baseline — β̄ = {beta_hat:.4f}  γ̄ = {gamma_hat:.4f}"
        f"  R₀ = {beta_hat / gamma_hat:.2f}  (N = {len(S_arr):,} obs)"
    )
    return beta_hat, gamma_hat


def _safe_lag_corr(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if lag <= 0 or values.size <= lag:
        return 0.0
    a = values[:-lag]
    b = values[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def get_sir_control_lookback(config: Config) -> int:
    """Maximum history required by every enabled causal control block."""
    explicit_window = (
        int(config.EXPLICIT_RECON_CONTROL_WINDOW)
        if _validate_recon_control_mode(config.EXPLICIT_RECON_CONTROLS) != "none"
        else 0
    )
    return max(max(config.SIGNATURE_PATH_LENGTHS), explicit_window)


def _safe_lag_corr_jax(values: jnp.ndarray, lag: int) -> jnp.ndarray:
    if lag <= 0 or values.shape[0] <= lag:
        return jnp.array(0.0, dtype=values.dtype)
    a = values[:-lag]
    b = values[lag:]
    a_centered = a - jnp.mean(a)
    b_centered = b - jnp.mean(b)
    denominator = jnp.sqrt(
        jnp.sum(a_centered**2) * jnp.sum(b_centered**2)
    )
    return jnp.where(
        denominator > 1e-12,
        jnp.sum(a_centered * b_centered) / denominator,
        0.0,
    )


def _explicit_recon_control_vector_jax(
    control_history: jnp.ndarray,
    config: Config,
    weekday: jnp.ndarray,
) -> jnp.ndarray:
    """JAX equivalent of the explicit causal control frame for simulation."""
    mode = _validate_recon_control_mode(config.EXPLICIT_RECON_CONTROLS)
    if mode == "none":
        return jnp.zeros((0,), dtype=control_history.dtype)
    if mode != "acf_weekday":
        raise AssertionError(f"Unhandled EXPLICIT_RECON_CONTROLS={mode!r}")
    window = max(int(config.EXPLICIT_RECON_CONTROL_WINDOW), 9)
    increments = jnp.diff(control_history[-window:, 1])
    energy = jnp.log1p(jnp.mean(increments**2))
    angle = 2.0 * jnp.pi * weekday.astype(control_history.dtype) / 7.0
    return jnp.stack(
        (
            _safe_lag_corr_jax(increments, 1),
            _safe_lag_corr_jax(increments, 7),
            energy,
            jnp.sin(angle),
            jnp.cos(angle),
        )
    )


def _explicit_recon_control_frame(
    features_normalized: pd.DataFrame,
    config: Config,
    valid_ts: pd.Index,
) -> pd.DataFrame:
    """Causal reconstruction-history controls appended after signatures."""
    mode = _validate_recon_control_mode(config.EXPLICIT_RECON_CONTROLS)
    if mode == "none":
        return pd.DataFrame(index=valid_ts)

    if mode != "acf_weekday":
        raise AssertionError(f"Unhandled EXPLICIT_RECON_CONTROLS={mode!r}")

    window = max(int(config.EXPLICIT_RECON_CONTROL_WINDOW), 9)
    i_values = features_normalized["I"].to_numpy(dtype=float)
    rows: list[dict[str, float]] = []
    for ts in valid_ts:
        end = int(features_normalized.index.get_loc(ts)) + 1
        start = max(0, end - window)
        i_window = i_values[start:end]
        increments = np.diff(i_window)
        if increments.size:
            energy = float(np.log1p(np.mean(increments**2)))
        else:
            energy = 0.0

        if isinstance(ts, pd.Timestamp):
            weekday = int(ts.weekday())
        else:
            weekday = 0
        angle = 2.0 * np.pi * weekday / 7.0
        rows.append(
            {
                "xctrl_acf1_dlogI": _safe_lag_corr(increments, 1),
                "xctrl_acf7_dlogI": _safe_lag_corr(increments, 7),
                "xctrl_energy_dlogI": energy,
                "xctrl_weekday_sin": float(np.sin(angle)),
                "xctrl_weekday_cos": float(np.cos(angle)),
            }
        )

    return pd.DataFrame(rows, index=valid_ts)


def create_sir_controls(
    features_normalized: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Compute causal controls from normalised [S, I].

    The first block is always a path signature.  Optional explicit
    reconstruction-history controls are appended after it; these are causal
    functions of the same reconstructed state path and the current calendar
    date, not functions of BI residuals.
    """
    sig_extractors = {
        length: JAXSignatureExtractor(depth=config.SIGNATURE_DEPTH, augment_time=True, lead_lag=config.SIGNATURE_LEAD_LAG)
        for length in config.SIGNATURE_PATH_LENGTHS
    }
    max_lookback = get_sir_control_lookback(config)
    valid_ts = features_normalized.index[max_lookback - 1 :]

    # Signatures from S, I
    state_arr = jnp.array(features_normalized.values)  # (Time, 2)

    # Direct integer indices — no per-timestamp Pandas .get_loc() call
    end_indices = jnp.arange(max_lookback, len(features_normalized) + 1)

    @jax.jit
    def compute_row(end_idx):
        # Signature-only control vector (no macro branch)
        sigs = []
        for length in config.SIGNATURE_PATH_LENGTHS:
            chunk = jax.lax.dynamic_slice(
                state_arr, (end_idx - length, 0), (length, state_arr.shape[1])
            )
            sigs.append(sig_extractors[length](chunk))

        return jnp.concatenate(sigs)

    ctrl_vectors = jax.lax.map(compute_row, end_indices)
    sig_df = pd.DataFrame(np.array(ctrl_vectors), index=valid_ts)
    explicit_df = _explicit_recon_control_frame(features_normalized, config, valid_ts)
    if explicit_df.empty:
        return sig_df
    return pd.concat([sig_df, explicit_df], axis=1)


def estimate_temporal_prewhiten_phi(norm_dfs: Dict[str, pd.DataFrame]) -> np.ndarray:
    """Estimate per-state AR(1) coefficients from normalized training paths."""
    numer = None
    denom = None
    for df in norm_dfs.values():
        arr = df[["S", "I"]].values.astype(float)
        if arr.shape[0] < 2:
            continue
        x_prev = arr[:-1]
        x_next = arr[1:]
        if numer is None:
            numer = np.zeros(arr.shape[1], dtype=float)
            denom = np.zeros(arr.shape[1], dtype=float)
        numer += np.sum(x_prev * x_next, axis=0)
        denom += np.sum(x_prev * x_prev, axis=0)

    if numer is None or denom is None:
        return np.zeros(2, dtype=float)

    phi = numer / np.maximum(denom, 1e-8)
    return np.clip(phi, -0.99, 0.99).astype(float)


def apply_temporal_prewhiten_frame(
    features_normalized: pd.DataFrame,
    ar_phi: np.ndarray | None,
) -> pd.DataFrame:
    """Apply AR(1) pre-whitening to signature inputs only."""
    if ar_phi is None:
        return features_normalized

    arr = features_normalized[["S", "I"]].values.astype(float)
    if arr.shape[0] < 2:
        return features_normalized.copy()

    phi = np.asarray(ar_phi, dtype=float)
    whitened = arr.copy()
    whitened[1:] = arr[1:] - phi[None, :] * arr[:-1]
    return pd.DataFrame(whitened, index=features_normalized.index, columns=features_normalized.columns)


def get_paths_from_dataframe(df: pd.DataFrame, path_length: int) -> np.ndarray:
    """Overlapping sliding-window paths.  Shape: (N, path_length, State)."""
    data = df.values.astype(np.float32)
    n = len(data) - path_length + 1
    if n <= 0:
        return np.empty((0, path_length, data.shape[1]), dtype=np.float32)
    # Zero-copy strided view — O(1) vs O(N·path_length) Python loop
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(data, (path_length, data.shape[1]))
    # sliding_window_view returns (N, 1, path_length, State); squeeze axis=1
    return np.ascontiguousarray(windows[:, 0])


def get_oracle_targets(ys: jnp.ndarray, window_k: int) -> jnp.ndarray:
    """
    Forward-smoothed drift targets for *all* state dimensions.
    target[t] = (y[t+k] − y[t]) / k
    Shape: (Batch, Time, State)  — matches ys.
    """
    vals_padded = jnp.pad(ys, ((0, 0), (0, window_k), (0, 0)), mode="edge")
    return (vals_padded[:, window_k:] - ys) / window_k


def get_realized_drift_targets(ys: jnp.ndarray) -> jnp.ndarray:
    """1-step realised drift targets for all state dims."""
    targets = jnp.zeros_like(ys)
    return targets.at[:, :-1, :].set(ys[:, 1:, :] - ys[:, :-1, :])


def load_or_create_training_data(
    config: Config, force_refresh: bool = False
) -> TrainingData:
    """Load from pickle cache or build from OWID download."""
    cache_path = get_cache_path(config)

    if not force_refresh and cache_path.exists():
        logging.info(f"Loading dataset from cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("config") == config:
                logging.info("Cache valid.")
                return saved["data"]
            logging.warning("Config mismatch — rebuilding.")
        except Exception as e:
            logging.warning(f"Cache load failed ({e}) — rebuilding.")

    logging.info("Building SIR training data from OWID …")

    raw_df = download_owid_data(config, force_download=force_refresh)

    all_isos = config.TRAIN_COUNTRIES + [config.VAL_COUNTRY]
    country_data = prepare_sir_countries(
        raw_df,
        all_isos,
        config.TRAINING_START_DATE,
        recovery_days=config.SIR_RECOVERY_DAYS,
        active_threshold=config.SIR_ACTIVE_THRESHOLD,
        incidence_source=config.SIR_INCIDENCE_SOURCE,
        incidence_blend_weight=config.SIR_INCIDENCE_BLEND_WEIGHT,
    )

    train_data = {
        k: v.copy() for k, v in country_data.items() if k in config.TRAIN_COUNTRIES
    }
    if not train_data:
        raise ValueError("No training country data available after filtering.")

    val_df_physical = country_data.get(config.VAL_COUNTRY)
    if val_df_physical is None:
        raise ValueError(f"Validation country {config.VAL_COUNTRY} not available.")

    # Use a separate copy for model input (Log-I).
    # Validations against physical data will use val_df_physical if needed,
    # but the pipeline expects val_df_raw to be compatible with model input.
    val_df_raw = val_df_physical.copy()

    # ── NEW: Log-Transform I for Log-Normal Dynamics ──────────────────────────
    # The model expects [S, log(I)]. We transform the COPIES in-place.
    for iso, df in train_data.items():
        df["I"] = np.log(df["I"])

    val_df_raw["I"] = np.log(val_df_raw["I"])

    # ── normalisation stats from training countries only ──────────────────────
    norm_mean, norm_std = compute_norm_stats(train_data)
    logging.info(f"Norm  mean={norm_mean}  std={norm_std}")

    config.NORMALIZATION_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.NORMALIZATION_STATS_PATH, mean=norm_mean, std=norm_std)

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        return (df - norm_mean) / (norm_std + 1e-8)

    train_norm_map = {
        iso: normalize(df[["S", "I"]])
        for iso, df in train_data.items()
    }
    ar_phi = None
    if config.USE_TEMPORAL_PREWHITEN:
        ar_phi = estimate_temporal_prewhiten_phi(train_norm_map)
        logging.info("Temporal pre-whitening enabled: ar_phi=%s", ar_phi)

    sig_sizes = [
        get_signature_size(config.STATE_SIZE, config.SIGNATURE_DEPTH, augment_time=True, lead_lag=config.SIGNATURE_LEAD_LAG)
        for _ in config.SIGNATURE_PATH_LENGTHS
    ]
    macro_size = get_recon_control_size(config)

    # ── per-country path arrays (no cross-country windows) ───────────────────
    train_ys_list: List[np.ndarray] = []
    train_ctrl_list: List[np.ndarray] = []
    train_norm_dfs: List[pd.DataFrame] = []
    train_cids_list: List[np.ndarray] = []

    # Iterate in TRAIN_COUNTRIES order so iso_idx matches beta_country_bases index in main().
    for iso_idx, iso in enumerate(config.TRAIN_COUNTRIES):
        if iso not in train_data:
            continue
        df = train_data[iso]
        # norm_df now contains only S, I (normalized)
        # df contains S, I (log)
        norm_df = train_norm_map[iso]

        ctrl_input_df = apply_temporal_prewhiten_frame(norm_df, ar_phi)
        ctrl_df = create_sir_controls(ctrl_input_df, config)
        ctrl_df = apply_control_ablation_frame(ctrl_df, config, seed_offset=100 + iso_idx)

        feat_al = norm_df.loc[norm_df.index >= ctrl_df.index[0]]
        ctrl_al = ctrl_df

        ys_arr = get_paths_from_dataframe(feat_al, config.TRAINING_PATH_LENGTH)
        ctrl_arr = get_paths_from_dataframe(ctrl_al, config.TRAINING_PATH_LENGTH)

        if ys_arr.shape[0] > 0:
            train_ys_list.append(ys_arr)
            train_ctrl_list.append(ctrl_arr)
            train_norm_dfs.append(feat_al)
            train_cids_list.append(np.full(ys_arr.shape[0], iso_idx, dtype=np.int32))
            logging.info(f"  {iso}: {ys_arr.shape[0]} training paths")

    if not train_ys_list:
        raise ValueError("No training paths created — check date range / path length.")

    train_ys = np.concatenate(train_ys_list, axis=0)
    train_ctrl = np.concatenate(train_ctrl_list, axis=0)
    train_country_ids = np.concatenate(train_cids_list) if train_cids_list else np.zeros(0, dtype=np.int32)
    n_countries = len(config.TRAIN_COUNTRIES)
    val_country_id = (
        config.TRAIN_COUNTRIES.index(config.VAL_COUNTRY)
        if config.VAL_COUNTRY in config.TRAIN_COUNTRIES
        else -1
    )

    # ── validation country ────────────────────────────────────────────────────
    val_norm_df = normalize(val_df_raw[["S", "I"]])
    val_ctrl_input_df = apply_temporal_prewhiten_frame(val_norm_df, ar_phi)
    val_ctrl_df = create_sir_controls(val_ctrl_input_df, config)
    val_ctrl_df = apply_control_ablation_frame(val_ctrl_df, config, seed_offset=2000)

    val_feat_al = val_norm_df.loc[val_norm_df.index >= val_ctrl_df.index[0]]
    val_ctrl_al = val_ctrl_df

    val_ys_arr = get_paths_from_dataframe(val_feat_al, config.TRAINING_PATH_LENGTH)
    val_ctrl_arr = get_paths_from_dataframe(val_ctrl_al, config.TRAINING_PATH_LENGTH)
    logging.info(f"  {config.VAL_COUNTRY} (val): {val_ys_arr.shape[0]} paths")

    train_set = TrainingDataSet(ys=jnp.array(train_ys), controls=jnp.array(train_ctrl))
    val_set = (
        TrainingDataSet(ys=jnp.array(val_ys_arr), controls=jnp.array(val_ctrl_arr))
        if val_ys_arr.shape[0] > 0
        else None
    )

    # ── combined raw S/I (un-normalised, for plotting) ──────────────────
    raw_pieces = []
    for iso, df in country_data.items():
        tmp = df.copy()
        tmp["country"] = iso
        raw_pieces.append(tmp)
    raw_sir_df = pd.concat(raw_pieces)

    features_df = pd.concat(train_norm_dfs)

    data = TrainingData(
        train_set=train_set,
        val_set=val_set,
        features_df=features_df,
        val_features_df=val_feat_al,
        macro_df=pd.DataFrame(),  # Can't concat df easily, better to leave empty for now
        raw_sir_df=raw_sir_df,
        norm_mean=norm_mean,
        norm_std=norm_std,
        signature_sizes=sig_sizes,
        macro_size=macro_size,
        train_country_ids=train_country_ids,
        val_country_id=val_country_id,
        n_countries=n_countries,
        ar_phi=ar_phi,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"data": data, "config": config}, f)
    logging.info(f"Saved cache to {cache_path}")
    return data
