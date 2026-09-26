"""Experiment settings, single-run orchestration and ablation sweeps."""

import equinox as eqx
import jax
import jax.numpy as jnp
import logging
import pandas as pd

from config import Config, get_experiment_tag, get_model_save_path
from neural_sde import NeuralSDE
from sir_data import fit_sir_baseline, load_or_create_training_data
from sir_evaluation import evaluate_experiment
from sir_training import train_ensemble, validate_training_config
from typing import Dict, List


def run_ablation_suite(
    base_config: Config,
    force_data_refresh: bool = False,
    force_train: bool = False,
    num_models_override: int = None,
) -> pd.DataFrame:
    summaries = []
    for mode in ("none", "zero", "shuffle"):
        cfg = Config(**vars(base_config))
        cfg.CONTROL_ABLATION = mode
        summary = main(
            cfg,
            force_data_refresh=force_data_refresh,
            force_train=force_train,
            num_models_override=num_models_override,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    base_config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    save_path = base_config.SUMMARY_DIR / "ablation_suite_summary.csv"
    summary_df.to_csv(save_path, index=False)
    logging.info("Saved ablation suite summary to %s", save_path)
    logging.info("\n%s", summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return summary_df


def run_sigma_ablation_suite(
    base_config: Config,
    force_data_refresh: bool = False,
    force_train: bool = False,
    num_models_override: int = None,
) -> pd.DataFrame:
    """Sweep log_sigma initialization values to find optimal constant σ."""
    log_sigma_values = [-4.0, -3.5, -3.0, -2.5, -2.0]
    summaries = []

    for ls_init in log_sigma_values:
        cfg = Config(**vars(base_config))
        cfg.LOG_SIGMA_INIT = ls_init
        cfg.CONTROL_ABLATION = "none"
        sigma_init = float(jnp.exp(jnp.array(ls_init)))
        logging.info(
            "=== Sigma ablation: log_sigma_init=%.1f (σ₀=%.4f) ===",
            ls_init, sigma_init,
        )
        summary = main(
            cfg,
            force_data_refresh=force_data_refresh,
            force_train=True,
            num_models_override=num_models_override,
        )
        summary["ablation_type"] = "log_sigma_init"
        summary["ablation_value"] = str(ls_init)

        # Record learned σ after training
        key = jax.random.PRNGKey(cfg.SEED)
        init_keys = jax.random.split(key, cfg.NUM_ENSEMBLE_MODELS)
        data = load_or_create_training_data(cfg, force_refresh=False)
        train_raw_map = {
            iso: data.raw_sir_df[data.raw_sir_df["country"] == iso].drop(columns=["country"])
            for iso in cfg.TRAIN_COUNTRIES
            if iso in data.raw_sir_df["country"].values
        }
        beta_hat, gamma_hat = fit_sir_baseline(train_raw_map)
        template = [
            NeuralSDE(
                signature_sizes=data.signature_sizes,
                macro_size=data.macro_size,
                state_size=cfg.STATE_SIZE,
                key=init_keys[i],
                norm_mean=data.norm_mean,
                norm_std=data.norm_std,
                beta_base=beta_hat,
                gamma_base=gamma_hat,
                log_sigma_init=cfg.LOG_SIGMA_INIT,
                use_diffusion_net=cfg.DIFFUSION_STATE_DEPENDENT,
                drift_width=cfg.DRIFT_WIDTH,
                signature_linear_drift=cfg.SIGNATURE_LINEAR_DRIFT,
                freeze_gamma=cfg.FREEZE_GAMMA,
                sigma_country_scale=cfg.SIGMA_COUNTRY_SCALE,
                n_countries=data.n_countries,
                use_per_country_beta=cfg.PER_COUNTRY_BETA_GAMMA,
            )
            for i in range(cfg.NUM_ENSEMBLE_MODELS)
        ]
        model_path = get_model_save_path(cfg)
        models = eqx.tree_deserialise_leaves(model_path, like=template)
        m = models[0] if isinstance(models, list) else models
        summary["sigma_init"] = sigma_init
        summary["sigma_learned"] = float(jnp.exp(m.log_sigma))
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    base_config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = base_config.SUMMARY_DIR / "sigma_ablation_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logging.info("Saved sigma ablation summary to %s", out_path)
    logging.info(
        "\n%s",
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x),
        ),
    )
    return summary_df


def run_recovery_days_ablation(
    base_config: Config,
    recovery_days_values: List[float],
    force_data_refresh: bool = False,
    force_train: bool = False,
    num_models_override: int = None,
) -> pd.DataFrame:
    summaries = []

    for recovery_days in recovery_days_values:
        cfg = Config(**vars(base_config))
        cfg.SIR_RECOVERY_DAYS = float(recovery_days)
        cfg.CONTROL_ABLATION = "none"
        logging.info("=== Recovery-days ablation: recovery_days=%s ===", recovery_days)
        summary = main(
            cfg,
            force_data_refresh=force_data_refresh,
            force_train=force_train,
            num_models_override=num_models_override,
        )
        summary["ablation_type"] = "recovery_days"
        summary["ablation_value"] = str(recovery_days)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    base_config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = base_config.SUMMARY_DIR / "recovery_days_ablation_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logging.info("Saved recovery-days ablation summary to %s", out_path)
    logging.info(
        "\n%s",
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x),
        ),
    )
    return summary_df


def run_special_experiments(
    base_config: Config,
    force_data_refresh: bool = False,
    force_train: bool = False,
    num_models_override: int = None,
) -> pd.DataFrame:
    """Run two targeted experiments:
    1. Freeze-γ + T_rec=28: γ_scale=0, only β has neural residual. Tests whether γ channel is redundant.
    2. T_rec=28 + d4_w64: combine the two best individual improvements.
    """
    summaries = []

    # --- Experiment 1: Freeze γ + T_rec=28 ---
    cfg1 = Config(**vars(base_config))
    cfg1.SIR_RECOVERY_DAYS = 28.0
    cfg1.FREEZE_GAMMA = True
    cfg1.CONTROL_ABLATION = "none"
    logging.info("=== Special: freeze_gamma=True, T_rec=28 ===")
    s1 = main(
        cfg1,
        force_data_refresh=force_data_refresh,
        force_train=force_train,
        num_models_override=num_models_override,
    )
    s1["ablation_type"] = "special"
    s1["ablation_value"] = "frzgamma_rec28"
    summaries.append(s1)

    # --- Experiment 2: T_rec=28 + d4_w64 ---
    cfg2 = Config(**vars(base_config))
    cfg2.SIR_RECOVERY_DAYS = 28.0
    cfg2.SIGNATURE_DEPTH = 4
    cfg2.DRIFT_WIDTH = 64
    cfg2.CONTROL_ABLATION = "none"
    logging.info("=== Special: T_rec=28 + depth=4 + width=64 ===")
    s2 = main(
        cfg2,
        force_data_refresh=force_data_refresh,
        force_train=True,  # always retrain: new architecture combo
        num_models_override=num_models_override,
    )
    s2["ablation_type"] = "special"
    s2["ablation_value"] = "rec28_d4_w64"
    summaries.append(s2)

    summary_df = pd.DataFrame(summaries)
    base_config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = base_config.SUMMARY_DIR / "special_experiments_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logging.info("Saved special experiments summary to %s", out_path)
    logging.info(
        "\n%s",
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x),
        ),
    )
    return summary_df


def run_depth_ablation(
    base_config: Config,
    force_data_refresh: bool = False,
    num_models_override: int = None,
) -> pd.DataFrame:
    """Run depth-only ablation: depth in {1, 2, 3, 4, 5}."""
    summaries = []
    for depth in (1, 2, 3, 4, 5):
        cfg = Config(**vars(base_config))
        cfg.SIGNATURE_DEPTH = depth
        cfg.CONTROL_ABLATION = "none"
        logging.info("=== Depth ablation: depth=%d ===", depth)
        summary = main(
            cfg,
            force_data_refresh=force_data_refresh,
            force_train=True,
            num_models_override=num_models_override,
        )
        summary["ablation_type"] = "depth"
        summary["ablation_value"] = str(depth)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    base_config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    save_path = base_config.SUMMARY_DIR / "depth_ablation_summary.csv"
    summary_df.to_csv(save_path, index=False)
    logging.info("Saved depth ablation summary to %s", save_path)
    logging.info(
        "\n%s",
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x),
        ),
    )
    return summary_df


def run_depth_width_ablation(
    base_config: Config,
    force_data_refresh: bool = False,
    num_models_override: int = None,
) -> pd.DataFrame:
    """Factorial depth × width ablation: {3, 4} × {32, 64, 128}."""
    summaries = []
    for depth in (3, 4):
        for width in (32, 64, 128):
            cfg = Config(**vars(base_config))
            cfg.SIGNATURE_DEPTH = depth
            cfg.DRIFT_WIDTH = width
            cfg.CONTROL_ABLATION = "none"
            logging.info("=== Depth×Width ablation: depth=%d, width=%d ===", depth, width)
            summary = main(
                cfg,
                force_data_refresh=force_data_refresh,
                force_train=True,
                num_models_override=num_models_override,
            )
            summary["ablation_type"] = "depth_width"
            summary["ablation_value"] = f"d{depth}_w{width}"
            summary["depth"] = depth
            summary["width"] = width
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    base_config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    save_path = base_config.SUMMARY_DIR / "depth_width_ablation_summary.csv"
    summary_df.to_csv(save_path, index=False)
    logging.info("Saved depth×width ablation summary to %s", save_path)
    logging.info(
        "\n%s",
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x),
        ),
    )
    return summary_df


def run_signature_ablation_suite(
    base_config: Config,
    force_data_refresh: bool = False,
    force_train: bool = False,
    num_models_override: int = None,
) -> pd.DataFrame:
    """Run signature-related ablation experiments: depth, lead-lag, window lengths."""
    summaries = []

    # 1. Depth ablation: {1, 2, 3, 4, 5}
    for depth in (1, 2, 3, 4, 5):
        cfg = Config(**vars(base_config))
        cfg.SIGNATURE_DEPTH = depth
        cfg.CONTROL_ABLATION = "none"
        logging.info("=== Signature depth ablation: depth=%d ===", depth)
        summary = main(
            cfg,
            force_data_refresh=force_data_refresh,
            force_train=True,  # architecture changes → must retrain
            num_models_override=num_models_override,
        )
        summary["ablation_type"] = "depth"
        summary["ablation_value"] = str(depth)
        summaries.append(summary)

    # 2. Lead-lag ablation: on vs off
    for ll in (True, False):
        cfg = Config(**vars(base_config))
        cfg.SIGNATURE_LEAD_LAG = ll
        cfg.CONTROL_ABLATION = "none"
        logging.info("=== Signature lead-lag ablation: lead_lag=%s ===", ll)
        summary = main(
            cfg,
            force_data_refresh=force_data_refresh,
            force_train=True,
            num_models_override=num_models_override,
        )
        summary["ablation_type"] = "lead_lag"
        summary["ablation_value"] = str(ll)
        summaries.append(summary)

    # 3. Window length ablation: [10], [20], [40], [60]
    for wlen in ([10], [20], [40], [60]):
        cfg = Config(**vars(base_config))
        cfg.SIGNATURE_PATH_LENGTHS = wlen
        cfg.CONTROL_ABLATION = "none"
        logging.info("=== Signature window ablation: window=%s ===", wlen)
        summary = main(
            cfg,
            force_data_refresh=force_data_refresh,
            force_train=True,
            num_models_override=num_models_override,
        )
        summary["ablation_type"] = "window"
        summary["ablation_value"] = str(wlen[0])
        summaries.append(summary)

    # 4. Diffusion mode ablation: constant vs state-dependent σ
    for sd in (False, True):
        cfg = Config(**vars(base_config))
        cfg.DIFFUSION_STATE_DEPENDENT = sd
        cfg.CONTROL_ABLATION = "none"
        logging.info("=== Diffusion mode ablation: state_dependent=%s ===", sd)
        summary = main(
            cfg,
            force_data_refresh=force_data_refresh,
            force_train=True,
            num_models_override=num_models_override,
        )
        summary["ablation_type"] = "diffusion"
        summary["ablation_value"] = str(sd)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    base_config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    save_path = base_config.SUMMARY_DIR / "signature_ablation_summary.csv"
    summary_df.to_csv(save_path, index=False)
    logging.info("Saved signature ablation summary to %s", save_path)
    logging.info("\n%s", summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))
    return summary_df


def main(
    config: Config,
    force_data_refresh: bool = False,
    force_train: bool = False,
    num_models_override: int = None,
) -> Dict[str, float]:
    logging.info("=== SIR Neural ODE ===")
    logging.info(
        "Experiment tag: %s | control ablation: %s",
        get_experiment_tag(config),
        config.CONTROL_ABLATION,
    )
    key = jax.random.PRNGKey(config.SEED)

    if num_models_override is not None:
        config.NUM_ENSEMBLE_MODELS = max(1, num_models_override)

    validate_training_config(config)
    data = load_or_create_training_data(config, force_refresh=force_data_refresh)
    n_train = data.train_set.ys.shape[0]
    n_val = data.val_set.ys.shape[0] if data.val_set else 0
    validate_training_config(config, n_train=n_train)
    logging.info(f"Train paths: {n_train}   Val paths: {n_val}")

    ensemble_models = train_ensemble(config, data, key, force_train=force_train)
    return evaluate_experiment(ensemble_models, data, config, key)
