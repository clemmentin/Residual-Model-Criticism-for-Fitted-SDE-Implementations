"""Rebuild the historical SWD comparison from retained fitted models."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import brownian_inversion
import config as config_module
import losses
import sir_evaluation

from experiments._bi_audit_common import (
    apply_config_updates as _apply_updates,
    load_cached_data as _load_cached_data,
)
from experiments._sir_model_loader import build_like_models, deserialize_models


OUT_PATH = ROOT / "cache" / "summaries" / "sir_swd_kernel_diagnostic.csv"

CASES = [
    # --- coupled: T_obs == T_dyn ---
    ("baseline",           {"FREEZE_GAMMA": False}),
    ("baseline_frzgamma",  {"FREEZE_GAMMA": True}),
    # --- distributional regularization: NLL + lambda * W1(z, N(0, 1)) ---
    ("baseline_zwd0p1_frzgamma", {"FREEZE_GAMMA": True, "ZSCORE_WD_WEIGHT": 0.1}),
    ("baseline_zwd1_frzgamma",   {"FREEZE_GAMMA": True, "ZSCORE_WD_WEIGHT": 1.0}),
    ("recov28",            {"SIR_RECOVERY_DAYS": 28.0, "FREEZE_GAMMA": False}),
    ("recov28_frzgamma",   {"SIR_RECOVERY_DAYS": 28.0, "FREEZE_GAMMA": True}),
    # --- decoupled: T_obs=28 (observation kernel), T_dyn=14 (latent gamma prior) ---
    ("recov28_dyn14",      {"SIR_RECOVERY_DAYS": 28.0, "SIR_RECOVERY_DAYS_DYN": 14.0,
                             "FREEZE_GAMMA": False}),
]


def _build_like_model(cfg: config_module.Config, data: config_module.TrainingData):
    return build_like_models(cfg, data)[0]


def _load_models(model_path: Path, like_model):
    return deserialize_models(model_path, [like_model])


def _actual_log_i_path(data: config_module.TrainingData, start_date, horizon: int) -> np.ndarray:
    start_idx = data.val_features_df.index.get_indexer([pd.to_datetime(start_date)], method="nearest")[0]
    z_norm = data.val_features_df["I"].values[start_idx : start_idx + horizon]
    return np.asarray(z_norm * data.norm_std[1] + data.norm_mean[1], dtype=float)


def _swd_to_single_path(
    simulated_paths: np.ndarray,
    observed_path: np.ndarray,
    n_projections: int = 512,
    seed: int = 123,
) -> float:
    simulated_paths = np.asarray(simulated_paths, dtype=float)
    observed_path = np.asarray(observed_path, dtype=float)
    if simulated_paths.ndim != 2:
        raise ValueError("simulated_paths must have shape (n_paths, horizon)")

    horizon = simulated_paths.shape[1]
    if observed_path.shape[0] != horizon:
        raise ValueError("observed_path horizon mismatch")

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, horizon))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

    sim_proj = simulated_paths @ directions.T
    obs_proj = observed_path @ directions.T
    return float(np.mean(np.abs(sim_proj - obs_proj[None, :])))


def _path_scores(simulated_i_paths: np.ndarray, observed_log_i: np.ndarray) -> dict:
    simulated_log_i = np.log(np.clip(np.asarray(simulated_i_paths, dtype=float), 1e-12, 1.0))
    observed_log_i = np.asarray(observed_log_i, dtype=float)

    sim_dlog_i = np.diff(simulated_log_i, axis=1)
    obs_dlog_i = np.diff(observed_log_i)

    return {
        "swd_logI_path": _swd_to_single_path(simulated_log_i, observed_log_i),
        "swd_dlogI_path": _swd_to_single_path(sim_dlog_i, obs_dlog_i),
        "w1_terminal_logI": float(np.mean(np.abs(simulated_log_i[:, -1] - observed_log_i[-1]))),
        "median_path_mae_logI": float(np.mean(np.abs(np.median(simulated_log_i, axis=0) - observed_log_i))),
    }


def _validation_nll(model, cfg: config_module.Config, data: config_module.TrainingData) -> dict:
    ts = jnp.linspace(0, cfg.TRAINING_PATH_LENGTH - 1, cfg.TRAINING_PATH_LENGTH)
    keep = cfg.TRAINING_PATH_LENGTH - 1
    total_loss, diagnostics = losses.loss(
        model,
        ts,
        data.val_set.ys,
        data.val_set.controls,
        data.val_set.ys,
        keep_steps=keep,
        # The archived comparison reports a one-day Euler validation loss.
        # Keep that reporting convention independent of the current training
        # default; forecast_sir below still uses projected substep simulation.
        fast_euler=True,
        s_loss_scale=cfg.S_LOSS_SCALE,
        substep_dt=cfg.SDE_SUBSTEP_DT,
        num_substeps=cfg.SDE_NUM_SUBSTEPS,
    )
    return {
        "val_total_loss": float(total_loss),
        "val_transition_nll": float(diagnostics["loss_nll"]),
        "val_loss_mse": float(diagnostics["loss_mse"]),
        "val_loss_bi_z_std": float(diagnostics["bi_z_std"]),
    }


def _run_case(name: str, updates: dict, reference_data: config_module.TrainingData) -> dict:
    cfg = _apply_updates(updates)
    cfg.NUM_SPAGHETTI_PATHS = 256

    data = _load_cached_data(cfg)
    like_model = _build_like_model(cfg, data)
    model_path = config_module.get_model_save_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing cached model: {model_path}")

    model = _load_models(model_path, like_model)[0]
    start_idx = max(len(data.val_features_df) // 2, max(cfg.SIGNATURE_PATH_LENGTHS))
    start_date = data.val_features_df.index[start_idx]

    key = jax.random.PRNGKey(cfg.SEED + 20260526)
    forecast_paths, _ = sir_evaluation.forecast_sir(
        [model],
        data.val_features_df,
        data.norm_mean,
        data.norm_std,
        cfg,
        key,
        start_idx=start_idx,
        ar_phi=data.ar_phi,
        country_idx=data.val_country_id,
    )
    forecast_paths = np.asarray(forecast_paths)

    own_log_i = _actual_log_i_path(data, start_date, cfg.FORECAST_HORIZON)
    ref_log_i = _actual_log_i_path(reference_data, start_date, cfg.FORECAST_HORIZON)

    bi_metrics = brownian_inversion.compute_brownian_inversion_metrics(model, str(start_date.date()), data, cfg)
    row = {
        "case": name,
        "experiment": config_module.get_experiment_tag(cfg),
        "recovery_days": float(cfg.SIR_RECOVERY_DAYS),
        "freeze_gamma": bool(cfg.FREEZE_GAMMA),
        "start_date": str(start_date.date()),
        **_validation_nll(model, cfg, data),
        **{f"own_{k}": v for k, v in _path_scores(forecast_paths, own_log_i).items()},
        **{f"ref28_{k}": v for k, v in _path_scores(forecast_paths, ref_log_i).items()},
        "bi_z_std": float(bi_metrics["bi_z_std"]),
        "bi_acf1": float(bi_metrics["bi_acf1"]),
        "bi_acf7": float(bi_metrics["bi_acf7"]),
        "bi_ks_pvalue": float(bi_metrics["bi_ks_pvalue"]),
        "bi_lb_pvalue": float(bi_metrics["bi_lb_pvalue"]),
    }
    return row


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    sir_evaluation.tqdm = lambda iterable, **kwargs: iterable

    reference_cfg = _apply_updates({"SIR_RECOVERY_DAYS": 28.0, "FREEZE_GAMMA": False})
    reference_data = _load_cached_data(reference_cfg)

    selected = set(sys.argv[1:])
    cases = [case for case in CASES if not selected or case[0] in selected]
    if not cases:
        raise ValueError(f"Unknown cases: {sorted(selected)}")

    rows = [_run_case(name, updates, reference_data) for name, updates in cases]
    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    cols = [
        "case",
        "val_transition_nll",
        "own_swd_logI_path",
        "own_swd_dlogI_path",
        "ref28_swd_logI_path",
        "ref28_swd_dlogI_path",
        "bi_z_std",
        "bi_acf1",
        "bi_acf7",
    ]
    print(f"Saved SIR SWD kernel diagnostic to {OUT_PATH}")
    print(df[cols].to_string(index=False))
    print()
    print(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
