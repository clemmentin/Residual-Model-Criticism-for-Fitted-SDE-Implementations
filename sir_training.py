"""Ensemble fitting, optimizer settings and training diagnostics."""

import config as config_module
import equinox as eqx
import jax
import jax.numpy as jnp
import logging
import numpy as np
import optax
import pandas as pd
import sir_data

from config import Config, TrainingData, _validate_recon_control_mode, get_model_save_path
from losses import loss as sde_loss
from neural_sde import NeuralSDE
from tqdm import tqdm


def validate_training_config(config: Config, n_train: int | None = None) -> None:
    """Fail early when training and one-day integration semantics disagree."""
    if config.SDE_SUBSTEP_DT <= 0:
        raise ValueError("SDE_SUBSTEP_DT must be positive.")
    if config.SDE_NUM_SUBSTEPS <= 0:
        raise ValueError("SDE_NUM_SUBSTEPS must be positive.")
    integrated_day = float(config.SDE_SUBSTEP_DT) * int(config.SDE_NUM_SUBSTEPS)
    if not np.isclose(integrated_day, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(
            "SDE_SUBSTEP_DT * SDE_NUM_SUBSTEPS must equal one day; "
            f"got {integrated_day:.12g}."
        )
    if config.BATCH_SIZE <= 0 or config.NUM_EPOCHS <= 0:
        raise ValueError("BATCH_SIZE and NUM_EPOCHS must be positive.")
    if not config.SIGNATURE_PATH_LENGTHS or min(config.SIGNATURE_PATH_LENGTHS) <= 0:
        raise ValueError("SIGNATURE_PATH_LENGTHS must contain positive integers.")
    if config.TRAINING_PATH_LENGTH <= 1:
        raise ValueError("TRAINING_PATH_LENGTH must be greater than one.")
    if (
        _validate_recon_control_mode(config.EXPLICIT_RECON_CONTROLS) != "none"
        and config.EXPLICIT_RECON_CONTROL_WINDOW <= 0
    ):
        raise ValueError("EXPLICIT_RECON_CONTROL_WINDOW must be positive.")
    if n_train is not None and n_train < config.BATCH_SIZE:
        raise ValueError(
            f"Training set has {n_train} paths, fewer than BATCH_SIZE={config.BATCH_SIZE}."
        )


def make_trainable_filter_spec(model: NeuralSDE):
    """Select trainable inexact arrays while excluding coordinate constants."""
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    return eqx.tree_at(
        lambda item: (
            item.norm_mean,
            item.norm_std,
            item.norm_whiten_matrix,
            item.norm_dewhiten_matrix,
        ),
        spec,
        replace=(False, False, False, False),
    )


def assert_model_normalization(model: NeuralSDE, data: TrainingData) -> None:
    """Verify that the frozen model implements the dataset's z-score map."""
    expected_mean = np.asarray(data.norm_mean)
    expected_std = np.asarray(data.norm_std)
    expected_whiten = np.diag(1.0 / expected_std)
    expected_dewhiten = np.diag(expected_std)
    checks = {
        "norm_mean": (np.asarray(model.norm_mean), expected_mean),
        "norm_std": (np.asarray(model.norm_std), expected_std),
        "norm_whiten_matrix": (
            np.asarray(model.norm_whiten_matrix),
            expected_whiten,
        ),
        "norm_dewhiten_matrix": (
            np.asarray(model.norm_dewhiten_matrix),
            expected_dewhiten,
        ),
    }
    failures = []
    for name, (actual, expected) in checks.items():
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-7):
            failures.append(
                f"{name} max_abs_error={float(np.max(np.abs(actual - expected))):.6g}"
            )
    inverse_error = float(
        np.max(
            np.abs(
                np.asarray(model.norm_whiten_matrix)
                @ np.asarray(model.norm_dewhiten_matrix)
                - np.eye(expected_mean.size)
            )
        )
    )
    if inverse_error > 1e-6:
        failures.append(f"normalization inverse error={inverse_error:.6g}")
    if failures:
        raise ValueError("Invalid model normalization: " + "; ".join(failures))


def get_sir_optimizer(
    model: NeuralSDE,
    total_steps: int,
    base_lr: float,
    weight_decay: float,
    config: Config,
) -> optax.GradientTransformation:
    """Single-phase AdamW with cosine-decay schedule."""
    schedule = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps,
        alpha=0.1,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.GRADIENT_CLIP_NORM),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )


def run_detailed_validation(
    model: NeuralSDE,
    start_date_str,       # unused for SIR — kept for API compat
    data,
    config: Config,
    epoch: int = None,
):
    """
    Runs a detailed validation pass and prints statistics to the log.
    Works for any STATE_SIZE — shows per-dimension drift / sigma.
    """
    # accept old 4-arg call (model, data, config, epoch) as well as
    # new 5-arg call (model, start_date_str, data, config, epoch)
    if not isinstance(start_date_str, str):
        # 4-arg form: positional args are (model, data, config, epoch)
        epoch           = config
        config          = data
        data            = start_date_str
        start_date_str  = None

    if data.val_set is None or data.val_set.ys.shape[0] == 0:
        return

    epoch_tag = f"Epoch {epoch}" if epoch is not None else ""
    logging.info(f"--- Detailed Validation {epoch_tag} ---")

    sample_idx    = 0
    ys_path       = data.val_set.ys[sample_idx]       # (Steps, State)
    controls_path = data.val_set.controls[sample_idx]  # (Steps, Control)

    @jax.jit
    def get_path_diagnostics(y_seq, c_seq):
        return jax.vmap(lambda y, c: model.get_diagnostics(0.0, y, c))(y_seq, c_seq)

    diag     = get_path_diagnostics(ys_path, controls_path)
    # diag["drift"]: (Steps, State),  diag["sigma"]: (Steps, State)
    state_size = ys_path.shape[-1]

    actual_changes = ys_path[1:] - ys_path[:-1]   # (Steps-1, State)
    pred_drift     = diag["drift"][:-1]             # (Steps-1, State)
    pred_sigma     = diag["sigma"][:-1]             # (Steps-1, State)
    gate_values    = diag["gate"][:-1]              # (Steps-1,)

    steps = np.arange(len(actual_changes))
    cols  = {
        "Step": steps,
        "Gate": np.array(gate_values).flatten(),
    }
    state_labels = ["S", "I"] if state_size == 2 else [f"s{d}" for d in range(state_size)]
    for d, lbl in enumerate(state_labels):
        cols[f"Act_{lbl}"]   = np.array(actual_changes[:, d])
        cols[f"Drift_{lbl}"] = np.array(pred_drift[:, d])
        cols[f"Sigma_{lbl}"] = np.array(pred_sigma[:, d])

    df_diag = pd.DataFrame(cols)
    logging.info(f"\n{epoch_tag} Validation Statistics:\n{df_diag.describe().to_string()}")


def train_ensemble(config: Config, data: TrainingData, key, force_train: bool = False):
    """Initialize, load or fit the ensemble; preserve the supplied random key."""
    n_train = data.train_set.ys.shape[0]
    ensemble_models = build_like_models(config, data)

    save_path = get_model_save_path(config)
    should_train = force_train

    if not force_train and save_path.exists():
        logging.info(f"Loading model from {save_path} …")
        try:
            ensemble_models = eqx.tree_deserialise_leaves(
                save_path, like=ensemble_models
            )
            for loaded_model in ensemble_models:
                assert_model_normalization(loaded_model, data)
            logging.info(f"Loaded {len(ensemble_models)} model(s).")
            should_train = False
        except Exception as e:
            logging.error(f"Load failed ({e}). Retraining.")
            should_train = True
    elif not force_train:
        should_train = True

    if should_train:
        ts = jnp.linspace(
            0, config.TRAINING_PATH_LENGTH - 1, config.TRAINING_PATH_LENGTH
        )
        # steps_per_epoch (ceiling) only used for LR schedule total_steps count.
        steps_per_epoch = (n_train + config.BATCH_SIZE - 1) // config.BATCH_SIZE
        total_steps = config.NUM_EPOCHS * steps_per_epoch
        # n_full: largest multiple of BATCH_SIZE ≤ n_train, used for scan batching.
        # Dropped samples (< BATCH_SIZE per epoch) rotate each epoch due to shuffling.
        n_full = (n_train // config.BATCH_SIZE) * config.BATCH_SIZE

        trained = []
        for i, model in enumerate(ensemble_models):
            logging.info(f"--- Training model {i+1}/{config.NUM_ENSEMBLE_MODELS} ---")
            assert_model_normalization(model, data)
            optimizer = get_sir_optimizer(
                model,
                total_steps,
                config.BASE_LEARNING_RATE,
                config.WEIGHT_DECAY,
                config,
            )
            trainable_spec = make_trainable_filter_spec(model)
            trainable_model, frozen_model = eqx.partition(model, trainable_spec)
            opt_state = optimizer.init(trainable_model)

            keep = config.TRAINING_PATH_LENGTH - 1

            @eqx.filter_jit
            def train_epoch(trainable_model, opt_state, ys_batched, ctrl_batched, cids_batched):
                """Process all batches in one JIT-compiled scan.

                Coordinate constants remain in the closed-over frozen tree and
                therefore cannot be changed by AdamW or weight decay.
                """
                _fast_euler = config.FAST_EULER_LOSS
                _s_loss_scale = config.S_LOSS_SCALE
                _zscore_wd_weight = config.ZSCORE_WD_WEIGHT
                _substep_dt = config.SDE_SUBSTEP_DT
                _num_substeps = config.SDE_NUM_SUBSTEPS

                def batch_step(carry, xs):
                    current_trainable, os = carry
                    y_b, c_b, cid_b = xs

                    def trainable_loss(candidate):
                        mdl = eqx.combine(candidate, frozen_model)
                        return sde_loss(
                            mdl,
                            ts,
                            y_b,
                            c_b,
                            y_b,
                            keep_steps=keep,
                            fast_euler=_fast_euler,
                            s_loss_scale=_s_loss_scale,
                            zscore_wd_weight=_zscore_wd_weight,
                            substep_dt=_substep_dt,
                            num_substeps=_num_substeps,
                            country_ids=cid_b,
                        )

                    (lv, _), grads = eqx.filter_value_and_grad(
                        trainable_loss, has_aux=True
                    )(current_trainable)
                    updates, new_os = optimizer.update(grads, os, current_trainable)
                    next_trainable = eqx.apply_updates(current_trainable, updates)
                    return (next_trainable, new_os), lv

                (trainable_model, opt_state), losses = jax.lax.scan(
                    batch_step, (trainable_model, opt_state),
                    (ys_batched, ctrl_batched, cids_batched),
                )
                return trainable_model, opt_state, jnp.mean(losses)

            @eqx.filter_jit
            def make_eval(model, ts, y_b, c_b, t_b):
                return sde_loss(
                    model, ts, y_b, c_b, t_b,
                    fast_euler=config.FAST_EULER_LOSS,
                    s_loss_scale=config.S_LOSS_SCALE,
                    zscore_wd_weight=config.ZSCORE_WD_WEIGHT,
                    substep_dt=config.SDE_SUBSTEP_DT,
                    num_substeps=config.SDE_NUM_SUBSTEPS,
                )

            pbar = tqdm(range(config.NUM_EPOCHS), desc=f"Model {i+1}")
            for epoch in pbar:
                epoch_key = jax.random.fold_in(key, i * config.NUM_EPOCHS + epoch)
                perm = jax.random.permutation(epoch_key, n_train)

                # Shuffle and reshape into (n_steps, BATCH_SIZE, ...) for scan
                ys_shuf   = data.train_set.ys[perm[:n_full]]
                ctrl_shuf = data.train_set.controls[perm[:n_full]]
                _cids_all = (
                    jnp.array(data.train_country_ids)
                    if data.train_country_ids is not None
                    else jnp.full(n_train, -1, dtype=jnp.int32)
                )
                cids_shuf = _cids_all[perm[:n_full]]
                n_steps   = n_full // config.BATCH_SIZE
                ys_batched   = ys_shuf.reshape(
                    n_steps, config.BATCH_SIZE, *ys_shuf.shape[1:]
                )
                ctrl_batched = ctrl_shuf.reshape(
                    n_steps, config.BATCH_SIZE, *ctrl_shuf.shape[1:]
                )
                cids_batched = cids_shuf.reshape(n_steps, config.BATCH_SIZE)

                trainable_model, opt_state, avg_loss_jax = train_epoch(
                    trainable_model, opt_state, ys_batched, ctrl_batched, cids_batched
                )
                model = eqx.combine(trainable_model, frozen_model)
                assert_model_normalization(model, data)
                avg_loss = avg_loss_jax.item()
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                if (epoch + 1) % config.VALIDATION_FREQ == 0 and data.val_set:
                    vt = data.val_set.ys
                    vl, _ = make_eval(
                        model,
                        ts,
                        data.val_set.ys,
                        data.val_set.controls,
                        vt,
                    )
                    logging.info(
                        f"Epoch {epoch+1:3d} | Train {avg_loss:.4f} | Val {vl.item():.4f}"
                    )
                    run_detailed_validation(model, data, config, epoch + 1)
                else:
                    logging.info(f"Epoch {epoch+1:3d} | Train {avg_loss:.4f}")

                if (epoch + 1) % config.SNAPSHOT_FREQ == 0:
                    snap = config.SNAPSHOT_DIR / f"sir_m{i}_ep{epoch+1}.eqx"
                    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                    eqx.tree_serialise_leaves(snap, model)

            trained.append(model)

        ensemble_models = trained
        save_path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(save_path, ensemble_models)
        logging.info(f"Saved ensemble → {save_path}")

    return ensemble_models


def _training_raw_map(data, cfg: config_module.Config) -> dict:
    return {
        iso: data.raw_sir_df[data.raw_sir_df["country"] == iso].drop(columns=["country"])
        for iso in cfg.TRAIN_COUNTRIES
        if iso in data.raw_sir_df["country"].values
    }


def fit_sir_priors(cfg: config_module.Config, data) -> tuple[float, float, list[float] | None, list[float] | None]:
    """Fit the shared baseline priors for training and checkpoint reconstruction."""
    train_raw_map = _training_raw_map(data, cfg)
    beta_hat, gamma_hat = sir_data.fit_sir_baseline(train_raw_map)

    if cfg.SIR_RECOVERY_DAYS_DYN is not None:
        gamma_hat = float(np.clip(1.0 / cfg.SIR_RECOVERY_DAYS_DYN, 0.005, 1.0))

    if not cfg.PER_COUNTRY_BETA_GAMMA:
        return beta_hat, gamma_hat, None, None

    per_country_params = {}
    for iso in cfg.TRAIN_COUNTRIES:
        if iso in train_raw_map:
            per_country_params[iso] = sir_data.fit_sir_baseline({iso: train_raw_map[iso]})

    beta_country_bases = [
        per_country_params.get(iso, (beta_hat, gamma_hat))[0]
        for iso in cfg.TRAIN_COUNTRIES
    ]
    gamma_country_bases = [
        per_country_params.get(iso, (beta_hat, gamma_hat))[1]
        for iso in cfg.TRAIN_COUNTRIES
    ]
    if cfg.SIR_RECOVERY_DAYS_DYN is not None:
        gamma_country_bases = [gamma_hat] * len(gamma_country_bases)

    return beta_hat, gamma_hat, beta_country_bases, gamma_country_bases


def build_like_models(cfg: config_module.Config, data) -> list[NeuralSDE]:
    """Initialize the same ensemble structure for fitting and checkpoint loading."""
    beta_hat, gamma_hat, beta_country_bases, gamma_country_bases = fit_sir_priors(cfg, data)
    init_keys = jax.random.split(jax.random.PRNGKey(cfg.SEED), cfg.NUM_ENSEMBLE_MODELS)
    return [
        NeuralSDE(
            signature_sizes=data.signature_sizes,
            macro_size=data.macro_size,
            state_size=cfg.STATE_SIZE,
            key=init_keys[i],
            norm_mean=data.norm_mean,
            norm_std=data.norm_std,
            beta_base=beta_hat,
            gamma_base=gamma_hat,
            use_diffusion_net=cfg.DIFFUSION_STATE_DEPENDENT,
            log_sigma_init=cfg.LOG_SIGMA_INIT,
            drift_width=cfg.DRIFT_WIDTH,
            signature_linear_drift=cfg.SIGNATURE_LINEAR_DRIFT,
            freeze_gamma=cfg.FREEZE_GAMMA,
            sigma_country_scale=cfg.SIGMA_COUNTRY_SCALE,
            n_countries=data.n_countries,
            use_per_country_beta=cfg.PER_COUNTRY_BETA_GAMMA,
            beta_country_bases=beta_country_bases,
            gamma_country_bases=gamma_country_bases,
        )
        for i in range(cfg.NUM_ENSEMBLE_MODELS)
    ]
