from dataclasses import dataclass, field
from typing import List, Optional, NamedTuple
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pandas as pd


class TrainingDataSet(NamedTuple):
    ys: jnp.ndarray
    controls: jnp.ndarray


class TrainingData(NamedTuple):
    train_set: TrainingDataSet
    val_set: Optional[TrainingDataSet]
    features_df: pd.DataFrame  # normalized [S, I] — training countries combined
    val_features_df: pd.DataFrame  # normalized [S, I] — val country
    macro_df: pd.DataFrame  # empty for SIR (kept for API compat)
    raw_sir_df: pd.DataFrame  # all countries raw S, I (date index, country col)
    norm_mean: np.ndarray  # global normalization mean
    norm_std: np.ndarray  # global normalization std
    signature_sizes: List[int]
    macro_size: int
    train_country_ids: Optional[np.ndarray] = None  # (N_train,) int country index per path
    val_country_id: Optional[int] = None             # country index for validation country
    n_countries: int = 0                              # total number of countries
    ar_phi: Optional[np.ndarray] = None              # AR(1) pre-whitening coefficients (state_size,)


_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    """Hyperparameters for the SIR-domain Neural SDE training."""

    # --- Paths ---
    CACHE_DIR: Path = _ROOT / "cache/"
    MODEL_SAVE_PATH: Path = _ROOT / "models/neural_sde_sir.eqx"
    SNAPSHOT_DIR: Path = _ROOT / "models/snapshots/"
    NORMALIZATION_STATS_PATH: Path = _ROOT / "models/sde_sir_normalization_stats.npz"
    BASELINE_PLOT_DIR: Path = _ROOT / "docs/images/baseline/"
    ROLLING_PLOT_DIR: Path = _ROOT / "docs/images/rolling/"
    SUMMARY_DIR: Path = _ROOT / "cache/summaries/"
    OWID_CACHE_PATH: Path = _ROOT / "cache/owid_covid_data.csv"

    # --- SIR Domain: Countries ---
    # Training countries: similar epidemiological profile (Western Europe)
    TRAIN_COUNTRIES: List[str] = field(
        default_factory=lambda: ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE"]
    )
    VAL_COUNTRY: str = "GBR"  # withheld for out-of-sample validation
    TRAINING_START_DATE: str = "2020-03-15"

    # --- SIR Physics Prior ---
    # DriftNet baseline: β_base=0.30, γ_base=0.07 (hardcoded in neural_sde.py).
    # Network learns small perturbations: β(t) = β_base ± 0.05, γ(t) = γ_base ± 0.02.
    # Exponential decay for active-cases estimation: I(t) = Σ new_cases(t-k) * exp(-k/T_rec)
    SIR_RECOVERY_DAYS: float = 14.0
    # Incidence input for reconstructing cumulative/active cases:
    # smoothed | raw | weekday_adjusted | blend_raw_smoothed.
    SIR_INCIDENCE_SOURCE: str = "smoothed"
    SIR_INCIDENCE_BLEND_WEIGHT: float = 0.0
    # If set, overrides gamma_base = 1/T_rec_dyn in SDE dynamics (decouples obs kernel from γ prior)
    SIR_RECOVERY_DAYS_DYN: Optional[float] = None
    # Drop rows whose active fraction < threshold (keeps only epidemic-active windows)
    SIR_ACTIVE_THRESHOLD: float = 1e-5

    # --- Model Architecture ---
    # State: [S, I] physical fractions, z-score normalised for network input.
    STATE_SIZE: int = 2  # [S, I] physical susceptible / infected fractions
    MACRO_SIZE: int = 0
    SIGNATURE_DEPTH: int = 3
    SIGNATURE_PATH_LENGTHS: List[int] = field(default_factory=lambda: [20])
    SIGNATURE_LEAD_LAG: bool = True  # Lead-Lag transform doubles channels before signature
    # Optional explicit causal controls appended after path signatures.
    # none | acf_weekday
    EXPLICIT_RECON_CONTROLS: str = "none"
    EXPLICIT_RECON_CONTROL_WINDOW: int = 20
    DRIFT_WIDTH: int = 32  # DriftNet hidden layer width
    SIGNATURE_LINEAR_DRIFT: bool = False  # True → β,γ = linear(sig); False → MLP(state, sig)
    DIFFUSION_STATE_DEPENDENT: bool = False  # True → σ(y, controls) via DiffusionNet; False → constant σ
    LOG_SIGMA_INIT: float = -2.5  # initial log_sigma for constant σ mode; σ₀ = exp(this)
    SIGMA_POSTHOC_RESCALE: bool = False   # Level 1: post-hoc per-country σ rescale at inference
    SIGMA_COUNTRY_SCALE: bool = False     # Level 2: learnable per-country σ embedding
    PER_COUNTRY_BETA_GAMMA: bool = False  # Per-country NLS β_base/γ_base (country-specific physical prior)
    FREEZE_GAMMA: bool = True  # True → γ_scale=0 (DriftNet outputs β residual only, γ fixed to γ_base)
    CONTROL_ABLATION: str = "none"  # one of: none | zero | shuffle

    # --- Data Parameters ---
    TRAINING_PATH_LENGTH: int = 60  # ~2 months of daily data per path

    # --- Training Parameters ---
    SEED: int = 42
    NUM_EPOCHS: int = 60
    BATCH_SIZE: int = 512
    BASE_LEARNING_RATE: float = 5e-3
    WEIGHT_DECAY: float = 1e-4
    GRADIENT_CLIP_NORM: float = 1.0
    VALIDATION_FREQ: int = 5
    SNAPSHOT_FREQ: int = 60

    # --- Ensemble ---
    NUM_ENSEMBLE_MODELS: int = 1

    # --- Forecast / Spaghetti ---
    FORECAST_HORIZON: int = 60
    NUM_SPAGHETTI_PATHS: int = 100
    NUM_ROLLING_MC_PATHS: int = 64  # SDE Monte Carlo paths per step for rolling CI

    # --- Loss Integration ---
    # fast_euler=False: SDE_NUM_SUBSTEPS micro-steps at SDE_SUBSTEP_DT per day (consistent with inference)
    # fast_euler=True : single dt=1 Euler step in loss (10x faster, O(dt²) error)
    FAST_EULER_LOSS: bool = False
    # Sub-step Euler parameters used in both inference (forecast / rolling) and loss (fast_euler=False)
    SDE_SUBSTEP_DT: float = 0.1     # integration sub-step size (days)
    SDE_NUM_SUBSTEPS: int = 10      # sub-steps per day; must equal round(1 / SDE_SUBSTEP_DT)
    # S state scale: daily ΔS ~ 1e-4 in norm space, Huber ≈ 5e-9 vs NLL O(1)
    S_LOSS_SCALE: float = 1e4
    # Optional 1D Wasserstein regularizer on standardized Brownian innovations z_t.
    ZSCORE_WD_WEIGHT: float = 0.0

    # Diffusion regularisation floor (prevents division by zero in inversion)
    DIFFUSION_REG: float = 1e-6

    # --- Temporal Pre-Whitening ---
    # AR(1) pre-whitening applied to the path fed into path-signature controls.
    # The model state (ys) is unchanged; only the signature input is filtered.
    # ar_phi is estimated from training data and stored in TrainingData.
    USE_TEMPORAL_PREWHITEN: bool = False


def get_cache_path(config: Config) -> Path:
    """Generates a unique cache path based on critical config parameters."""
    path_lengths_str = "_".join(map(str, config.SIGNATURE_PATH_LENGTHS))
    countries_str = "_".join(sorted(config.TRAIN_COUNTRIES))
    control_tag = _validate_control_ablation_mode(config.CONTROL_ABLATION)
    incidence_tag = _validate_incidence_source(config.SIR_INCIDENCE_SOURCE)
    recon_control_tag = _validate_recon_control_mode(config.EXPLICIT_RECON_CONTROLS)
    cache_tag = (
        f"sir_"
        f"train{countries_str}_"
        f"val{config.VAL_COUNTRY}_"
        f"start{config.TRAINING_START_DATE}_"
        f"pathlen{config.TRAINING_PATH_LENGTH}_"
        f"sigdepth{config.SIGNATURE_DEPTH}_"
        f"sigpaths{path_lengths_str}_"
        f"leadlag{int(config.SIGNATURE_LEAD_LAG)}_"
        f"diffnet{int(config.DIFFUSION_STATE_DEPENDENT)}_"
        f"state{config.STATE_SIZE}_"
        f"ctrlabl{control_tag}_"
        f"physical_"
        f"logI_"
        f"inc{incidence_tag}_"
        f"blend{_format_float_tag(config.SIR_INCIDENCE_BLEND_WEIGHT)}_"
        f"recov{config.SIR_RECOVERY_DAYS}_"
        f"xctrl{recon_control_tag}_"
        f"xctrlw{config.EXPLICIT_RECON_CONTROL_WINDOW}_"
        f"thresh{config.SIR_ACTIVE_THRESHOLD}_"
        f"prewhiten{int(config.USE_TEMPORAL_PREWHITEN)}"
    )
    safe_cache_tag = "".join(
        c if c.isalnum() or c in ("_", "-") else "-" for c in cache_tag
    )
    return config.CACHE_DIR / f"{safe_cache_tag}.pkl"


def _validate_control_ablation_mode(mode: str) -> str:
    mode = str(mode).lower()
    valid_modes = {"none", "zero", "shuffle"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown CONTROL_ABLATION={mode!r}. Expected one of {sorted(valid_modes)}."
        )
    return mode


def _validate_incidence_source(source: str) -> str:
    source = str(source).lower()
    valid_sources = {"smoothed", "raw", "weekday_adjusted", "blend_raw_smoothed"}
    if source not in valid_sources:
        raise ValueError(
            f"Unknown SIR_INCIDENCE_SOURCE={source!r}. Expected one of {sorted(valid_sources)}."
        )
    return source


def _validate_recon_control_mode(mode: str) -> str:
    mode = str(mode).lower()
    valid_modes = {"none", "acf_weekday"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown EXPLICIT_RECON_CONTROLS={mode!r}. Expected one of {sorted(valid_modes)}."
        )
    return mode


def get_recon_control_size(config: Config) -> int:
    mode = _validate_recon_control_mode(config.EXPLICIT_RECON_CONTROLS)
    if mode == "none":
        return 0
    if mode == "acf_weekday":
        # rolling ACF(1), rolling ACF(7), rolling increment energy,
        # weekday sin, weekday cos.
        return 5
    raise AssertionError(f"Unhandled reconstruction control mode: {mode}")


def _format_recovery_days_tag(recovery_days: float) -> str:
    recovery_days = float(recovery_days)
    if recovery_days.is_integer():
        return str(int(recovery_days))
    return f"{recovery_days:g}"


def _format_float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def get_experiment_tag(config: Config) -> str:
    mode = _validate_control_ablation_mode(config.CONTROL_ABLATION)
    incidence_source = _validate_incidence_source(config.SIR_INCIDENCE_SOURCE)
    recon_control_mode = _validate_recon_control_mode(config.EXPLICIT_RECON_CONTROLS)
    ctrl_part = "baseline" if mode == "none" else f"ctrl_{mode}"
    # Include signature params when they deviate from defaults
    parts = [ctrl_part]
    if incidence_source != "smoothed":
        parts.append(f"inc{incidence_source}")
    if incidence_source == "blend_raw_smoothed":
        parts.append(f"b{_format_float_tag(config.SIR_INCIDENCE_BLEND_WEIGHT)}")
    if config.SIR_RECOVERY_DAYS != 14.0:
        parts.append(f"recov{_format_recovery_days_tag(config.SIR_RECOVERY_DAYS)}")
    if config.SIR_RECOVERY_DAYS_DYN is not None:
        parts.append(f"dyn{_format_recovery_days_tag(config.SIR_RECOVERY_DAYS_DYN)}")
    if config.SIGNATURE_DEPTH != 3:
        parts.append(f"d{config.SIGNATURE_DEPTH}")
    if config.SIGNATURE_PATH_LENGTHS != [20]:
        parts.append("w" + "_".join(map(str, config.SIGNATURE_PATH_LENGTHS)))
    if not config.SIGNATURE_LEAD_LAG:
        parts.append("noll")
    if recon_control_mode != "none":
        parts.append(f"xctrl{recon_control_mode}")
        if config.EXPLICIT_RECON_CONTROL_WINDOW != max(config.SIGNATURE_PATH_LENGTHS):
            parts.append(f"xw{config.EXPLICIT_RECON_CONTROL_WINDOW}")
    if config.DRIFT_WIDTH != 32:
        parts.append(f"w{config.DRIFT_WIDTH}")
    if config.SIGNATURE_LINEAR_DRIFT:
        parts.append("siglinear")
    if config.DIFFUSION_STATE_DEPENDENT:
        parts.append("sigmastate")
    if config.USE_TEMPORAL_PREWHITEN:
        parts.append("tprewhite")
    if config.LOG_SIGMA_INIT != -2.5:
        parts.append(f"ls{config.LOG_SIGMA_INIT:.1f}")
    if config.SIGMA_COUNTRY_SCALE:
        parts.append("csigma")
    if config.PER_COUNTRY_BETA_GAMMA:
        parts.append("pcbeta")
    if config.ZSCORE_WD_WEIGHT != 0.0:
        parts.append(f"zwd{_format_float_tag(config.ZSCORE_WD_WEIGHT)}")
    if config.FREEZE_GAMMA:
        parts.append("frzgamma")
    return "_".join(parts)


def get_model_save_path(config: Config) -> Path:
    base = config.MODEL_SAVE_PATH.with_suffix(".eqx")
    tag = get_experiment_tag(config)
    return base.with_name(f"{base.stem}_{tag}{base.suffix}")


def apply_control_ablation_array(controls, config: Config, seed_offset: int = 0):
    mode = _validate_control_ablation_mode(config.CONTROL_ABLATION)
    if mode == "none":
        return controls
    if mode == "zero":
        return controls * 0

    perm = np.random.default_rng(config.SEED + seed_offset).permutation(
        int(controls.shape[0])
    )
    return controls[perm]


def apply_control_ablation_frame(
    controls_df: pd.DataFrame,
    config: Config,
    seed_offset: int = 0,
) -> pd.DataFrame:
    arr = apply_control_ablation_array(controls_df.values, config, seed_offset=seed_offset)
    return pd.DataFrame(arr, index=controls_df.index, columns=controls_df.columns)
