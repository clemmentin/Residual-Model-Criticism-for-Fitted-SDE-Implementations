"""Paired map-only comparison for the corrected SIR fitted-null audit.

This runner deliberately does not retrain a model and does not change the
fitted-null simulator.  It generates one common bank of stochastic paths and
evaluates two audit maps on those same paths:

* the existing deterministic plug-in accumulated local scale; and
* a scalar I-coordinate scale obtained from the finite-step tangent covariance
  recursion

      Sigma_{l+1} = F_l Sigma_l F_l.T + h g_l g_l.T,
      F_l = I + h J_b(y_l, c).

The score family remains the registered six-component family.  The purpose is
an implementation-level paired comparison, not a new confirmatory analysis.
Run from the sde directory, for example:

    python experiments/run_sir_tangent_map_comparison.py \
        --val FIN NOR SWE --n-bootstrap 500
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import config as config_module
import sir_data
from trajectory import JAXSignatureExtractor

from experiments import run_sir_bi_bootstrap as bi
from experiments._bi_audit_common import apply_config_updates, load_cached_data, summarize_metric
from experiments.country_bi_scoring import build_country_rows


BASE_COUNTRIES = ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE", "GBR"]
CORRECTED_MODEL_ROOT = Path(
    "models/country_cv/corrected_fixednorm_matchedsolver"
)
DEFAULT_OUT_DIR = ROOT / "cache" / "summaries" / "tangent_map_comparison"


def _config(val_country: str, model_root: Path) -> config_module.Config:
    updates = {
        **bi.CASES["recov28_frzgamma"],
        "VAL_COUNTRY": val_country,
        "TRAIN_COUNTRIES": BASE_COUNTRIES,
        "MODEL_SAVE_PATH": model_root
        / f"neural_sde_sir_val{val_country}.eqx",
    }
    return apply_config_updates(updates)


def _make_context(model, data: config_module.TrainingData, cfg: config_module.Config):
    max_lookback = sir_data.get_sir_control_lookback(cfg)
    raw_lookback = max_lookback + (1 if data.ar_phi is not None else 0)
    dt = cfg.SDE_SUBSTEP_DT
    n_substeps = cfg.SDE_NUM_SUBSTEPS
    identity = jnp.eye(cfg.STATE_SIZE)
    country_idx = jnp.array(
        -1 if data.val_country_id is None else data.val_country_id,
        dtype=jnp.int32,
    )
    norm_mean = jnp.asarray(data.norm_mean)
    norm_std = jnp.asarray(data.norm_std)
    ar_phi = None if data.ar_phi is None else jnp.asarray(data.ar_phi)
    signature_extractors = {
        length: JAXSignatureExtractor(
            depth=cfg.SIGNATURE_DEPTH,
            augment_time=True,
            lead_lag=cfg.SIGNATURE_LEAD_LAG,
        )
        for length in cfg.SIGNATURE_PATH_LENGTHS
    }

    def control_vector(history, weekday):
        control_history = history
        if ar_phi is not None:
            control_history = history[1:] - ar_phi[jnp.newaxis, :] * history[:-1]
        signatures = [
            signature_extractors[length](control_history[-length:])
            for length in cfg.SIGNATURE_PATH_LENGTHS
        ]
        explicit = sir_data._explicit_recon_control_vector_jax(
            control_history, cfg, weekday
        )
        return jnp.concatenate((*signatures, explicit))

    def transition_moments(state, control_vec):
        """Return deterministic drift and old/new I scales for one day."""

        def micro_step(carry, _unused):
            y_curr, cov = carry
            drift = model._calculate_drift(
                y_curr, control_vec, country_idx=country_idx
            )
            diffusion = model._calculate_diffusion(
                y_curr, control_vec, country_idx=country_idx
            ) + cfg.DIFFUSION_REG
            jacobian = jax.jacfwd(
                lambda y: model._calculate_drift(
                    y, control_vec, country_idx=country_idx
                )
            )(y_curr)
            F = identity + dt * jacobian
            next_cov = (
                F @ cov @ F.T
                + dt * jnp.outer(diffusion, diffusion)
            )
            return (y_curr + drift * dt, next_cov), None

        (final_state, tangent_cov), _ = jax.lax.scan(
            micro_step,
            (state, jnp.zeros((cfg.STATE_SIZE, cfg.STATE_SIZE), dtype=state.dtype)),
            jnp.arange(n_substeps),
        )
        plugin_sigma_sq = jnp.zeros((cfg.STATE_SIZE,), dtype=state.dtype)

        def plugin_step(carry, _unused):
            y_curr, var_acc = carry
            diffusion = model._calculate_diffusion(
                y_curr, control_vec, country_idx=country_idx
            ) + cfg.DIFFUSION_REG
            return (
                y_curr
                + model._calculate_drift(
                    y_curr, control_vec, country_idx=country_idx
                )
                * dt,
                var_acc + diffusion**2 * dt,
            ), None

        (_, plugin_sigma_sq), _ = jax.lax.scan(
            plugin_step,
            (state, plugin_sigma_sq),
            jnp.arange(n_substeps),
        )
        return (
            final_state - state,
            jnp.sqrt(jnp.maximum(plugin_sigma_sq, 1e-12)),
            jnp.sqrt(jnp.maximum(jnp.diag(tangent_cov), 1e-12)),
            tangent_cov,
        )

    return raw_lookback, control_vector, transition_moments


def _observed_residuals(model, data, cfg, start_pos: int):
    raw_lookback, control_vector, transition_moments = _make_context(model, data, cfg)
    horizon = min(cfg.FORECAST_HORIZON, len(data.val_features_df) - 1 - start_pos)
    initial_history = jnp.asarray(
        data.val_features_df.iloc[start_pos - raw_lookback + 1 : start_pos + 1].values
    )
    future_states = jnp.asarray(
        data.val_features_df.iloc[start_pos + 1 : start_pos + horizon + 1].values
    )
    start_weekday = int(data.val_features_df.index[start_pos].weekday())

    # Weekday changes each day; use a Python loop to preserve the exact
    # observed-path control alignment while keeping the transition itself JITed.
    history = initial_history
    old_rows, tangent_rows, cov01_rows, cov11_rows = [], [], [], []
    for offset, next_state in enumerate(future_states):
        control_vec = control_vector(history, (start_weekday + offset) % 7)
        drift, plugin_sigma, tangent_sigma, tangent_cov = transition_moments(
            history[-1], control_vec
        )
        increment = next_state - history[-1]
        old_rows.append((increment[1] - drift[1]) / plugin_sigma[1])
        tangent_rows.append((increment[1] - drift[1]) / tangent_sigma[1])
        cov01_rows.append(tangent_cov[0, 1])
        cov11_rows.append(tangent_cov[1, 1])
        history = jnp.roll(history, shift=-1, axis=0).at[-1].set(next_state)
    return (
        np.asarray(old_rows),
        np.asarray(tangent_rows),
        np.asarray(cov01_rows),
        np.asarray(cov11_rows),
    )


def _simulated_residuals(model, data, cfg, start_pos: int, n_bootstrap: int, seed: int):
    raw_lookback, control_vector, transition_moments = _make_context(model, data, cfg)
    horizon = min(cfg.FORECAST_HORIZON, len(data.val_features_df) - 1 - start_pos)
    dt = cfg.SDE_SUBSTEP_DT
    n_substeps = cfg.SDE_NUM_SUBSTEPS
    sqrt_dt = jnp.sqrt(dt)
    initial_history = jnp.asarray(
        data.val_features_df.iloc[start_pos - raw_lookback + 1 : start_pos + 1].values
    )
    norm_mean = jnp.asarray(data.norm_mean)
    norm_std = jnp.asarray(data.norm_std)
    country_idx = jnp.array(
        -1 if data.val_country_id is None else data.val_country_id,
        dtype=jnp.int32,
    )

    def clip_physical(state):
        physical = state * norm_std + norm_mean
        clipped = jnp.clip(
            physical,
            jnp.array([1e-6, -20.0]),
            jnp.array([1.0 - 1e-6, 0.0]),
        )
        was_clipped = jnp.any(jnp.abs(clipped - physical) > 1e-12)
        return (clipped - norm_mean) / norm_std, was_clipped

    start_weekday = int(data.val_features_df.index[start_pos].weekday())

    def simulate_one(path_noise):
        def day_step(history, day_input):
            day_noise, offset = day_input
            control_vec = control_vector(
                history, (start_weekday + offset) % 7
            )
            state = history[-1]
            drift, plugin_sigma, tangent_sigma, tangent_cov = transition_moments(
                state, control_vec
            )

            def stochastic_micro_step(carry, dw_unit):
                current_state, n_clipped = carry
                drift_now = model._calculate_drift(
                    current_state, control_vec, country_idx=country_idx
                )
                diffusion_now = model._calculate_diffusion(
                    current_state, control_vec, country_idx=country_idx
                )
                proposed = current_state + drift_now * dt + diffusion_now * dw_unit * sqrt_dt
                next_state, was_clipped = clip_physical(proposed)
                return (next_state, n_clipped + was_clipped.astype(jnp.int32)), None

            (next_state, n_clipped), _ = jax.lax.scan(
                stochastic_micro_step,
                (state, jnp.zeros((), dtype=jnp.int32)),
                day_noise,
            )
            increment = next_state - state
            old_residual = (increment[1] - drift[1]) / plugin_sigma[1]
            tangent_residual = (increment[1] - drift[1]) / tangent_sigma[1]
            next_history = jnp.roll(history, shift=-1, axis=0).at[-1].set(next_state)
            return next_history, (
                old_residual,
                tangent_residual,
                tangent_cov[0, 1],
                tangent_cov[1, 1],
                n_clipped,
            )

        _, outputs = jax.lax.scan(
            day_step,
            initial_history,
            (path_noise, jnp.arange(horizon)),
        )
        return outputs

    @eqx.filter_jit
    def simulate_batch(path_noise):
        return jax.vmap(simulate_one)(path_noise)

    noise = jax.random.normal(
        jax.random.PRNGKey(seed), shape=(n_bootstrap, horizon, n_substeps)
    )
    old, tangent, cov01, cov11, n_clipped = simulate_batch(noise)
    n_clipped = jnp.sum(n_clipped, axis=1)
    clip_fraction = n_clipped / float(horizon * n_substeps)
    return (
        np.asarray(old),
        np.asarray(tangent),
        np.asarray(cov01),
        np.asarray(cov11),
        np.asarray(clip_fraction),
    )


def _metric_rows(case, val_country, experiment, start_date, observed, draws):
    rows = []
    for metric in bi.METRIC_TAILS:
        rows.append(
            summarize_metric(
                case_name=case,
                experiment=experiment,
                start_date=start_date,
                n_bootstrap=len(draws),
                metric=metric,
                observed=float(observed[metric]),
                null_values=draws[metric].values,
                metric_tails=bi.METRIC_TAILS,
            )
        )
    summary = pd.DataFrame(rows)
    summary["val_country"] = val_country
    return summary


def _metric_draws(case, val_country, experiment, start_date, residuals, clip_fraction):
    rows = []
    for idx, values in enumerate(residuals):
        rows.append(
            {
                "case": case,
                "experiment": experiment,
                "start_date": start_date,
                "bootstrap_id": idx,
                "clip_fraction": float(clip_fraction[idx]),
                "val_country": val_country,
                "fold_train_countries": ",".join(BASE_COUNTRIES),
                **bi.compute_brownian_inversion_metrics_from_residuals(values),
            }
        )
    return pd.DataFrame(rows)


def run_country(val_country: str, n_bootstrap: int, seed: int, out_root: Path, model_root: Path):
    cfg = _config(val_country, model_root)
    bi._validate_supported_config(cfg)
    data = load_cached_data(cfg)
    model, model_path = bi._load_model(cfg, data)
    start_pos = bi._start_position(data, cfg)
    start_date = str(data.val_features_df.index[start_pos].date())
    experiment = config_module.get_experiment_tag(cfg)

    observed_old, observed_tangent, obs_cov01, obs_cov11 = _observed_residuals(
        model, data, cfg, start_pos
    )
    old_draws_raw, tangent_draws_raw, cov01, cov11, clip_fraction = _simulated_residuals(
        model, data, cfg, start_pos, n_bootstrap, seed
    )

    observed_old_metrics = bi.compute_brownian_inversion_metrics_from_residuals(observed_old)
    observed_tangent_metrics = bi.compute_brownian_inversion_metrics_from_residuals(
        observed_tangent
    )
    old_draws = _metric_draws(
        "recov28_frzgamma_plugin", val_country, experiment, start_date, old_draws_raw, clip_fraction
    )
    tangent_draws = _metric_draws(
        "recov28_frzgamma_tangentI", val_country, experiment, start_date, tangent_draws_raw, clip_fraction
    )
    old_summary = _metric_rows(
        "recov28_frzgamma_plugin", val_country, experiment, start_date,
        observed_old_metrics, old_draws
    )
    tangent_summary = _metric_rows(
        "recov28_frzgamma_tangentI", val_country, experiment, start_date,
        observed_tangent_metrics, tangent_draws
    )
    old_global, old_profile = build_country_rows(old_summary, old_draws)
    tangent_global, tangent_profile = build_country_rows(tangent_summary, tangent_draws)

    out_dir = out_root / f"recov28_frzgamma_val{val_country}"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_draws.to_csv(out_dir / "plugin_draws.csv", index=False)
    tangent_draws.to_csv(out_dir / "tangentI_draws.csv", index=False)
    old_summary.to_csv(out_dir / "plugin_summary.csv", index=False)
    tangent_summary.to_csv(out_dir / "tangentI_summary.csv", index=False)
    pd.DataFrame([old_global]).to_csv(out_dir / "plugin_global_score.csv", index=False)
    pd.DataFrame([tangent_global]).to_csv(out_dir / "tangentI_global_score.csv", index=False)
    pd.DataFrame([old_profile]).to_csv(out_dir / "plugin_max_adjusted_profile.csv", index=False)
    pd.DataFrame([tangent_profile]).to_csv(out_dir / "tangentI_max_adjusted_profile.csv", index=False)
    np.savez(
        out_dir / "paired_raw_diagnostics.npz",
        observed_plugin=observed_old,
        observed_tangentI=observed_tangent,
        observed_cov01=obs_cov01,
        observed_cov11=obs_cov11,
        simulated_cov01=cov01,
        simulated_cov11=cov11,
    )

    comparison = {
        "country": val_country,
        "model_path": str(model_path),
        "start_date": start_date,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "plugin_global_p": float(old_global["smax_empirical_pvalue"]),
        "tangentI_global_p": float(tangent_global["smax_empirical_pvalue"]),
        "plugin_observed_smax": float(old_global["observed_smax"]),
        "tangentI_observed_smax": float(tangent_global["observed_smax"]),
        "mean_tangent_cov_II": float(np.mean(cov11)),
        "mean_tangent_cov_SI": float(np.mean(cov01)),
        "observed_mean_tangent_cov_II": float(np.mean(obs_cov11)),
        "observed_mean_tangent_cov_SI": float(np.mean(obs_cov01)),
        "mean_clip_fraction": float(np.mean(clip_fraction)),
        "max_clip_fraction": float(np.max(clip_fraction)),
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", nargs="+", default=["FIN", "NOR", "SWE"])
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-root", type=Path, default=CORRECTED_MODEL_ROOT)
    args = parser.parse_args()

    comparisons = [
        run_country(
            val_country=val.upper(),
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            out_root=args.out_dir,
            model_root=args.model_root,
        )
        for val in args.val
    ]
    pd.DataFrame(comparisons).to_csv(args.out_dir / "paired_comparison.csv", index=False)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
