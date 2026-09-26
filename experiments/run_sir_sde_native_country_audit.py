"""Frozen-development SDE-native country audit for the corrected SIR model.

The protocol has two deliberately separate phases:

1. ``develop`` evaluates and numerically checks the endpoint family only on
   the original nine-country development cohort.
2. ``evaluate`` reads the development result summary and then evaluates the
   FIN/NOR/SWE trajectories exactly once.

The endpoint family adds three SDE-native layers without replacing the
registered six-component BI analysis:

* a finite-step I-coordinate martingale map using projected tangent covariance;
* predictable-bracket functionals of that coordinate; and
* a nonlinear Dynkin martingale diagnostic for S*I.

All endpoint ranks are calibrated with paths from the fitted projected Euler
simulator.  No iid-Normal or chi-square reference law is used.  A separate
finite-step alignment table compares the training plug-in variance, projected
tangent covariance, and direct conditional Monte Carlo moments; alignment
numbers are implementation diagnostics and never enter the country score.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

import config as config_module
import sir_data
import sir_training
from trajectory import JAXSignatureExtractor

from experiments import run_sir_bi_bootstrap as bi
from experiments._bi_audit_common import apply_config_updates, load_cached_data
from experiments._sir_model_loader import build_like_models, deserialize_models
from experiments.audit_statistics import departure as _departure, upper_rank


DEVELOPMENT_COUNTRIES = (
    "DEU",
    "FRA",
    "ITA",
    "ESP",
    "NLD",
    "BEL",
    "AUT",
    "CHE",
    "GBR",
)
CONFIRMATORY_COUNTRIES = ("FIN", "NOR", "SWE")

MODEL_ROOT = ROOT / "models" / "country_cv" / "corrected_fixednorm_matchedsolver"
COMMON_MODEL_PATH = (
    MODEL_ROOT / "neural_sde_sir_valFIN_baseline_recov28_frzgamma.eqx"
)
OWID_CACHE_PATH = ROOT / "cache" / "owid_covid_data.csv"
DEFAULT_OUT_ROOT = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_country_audit"
    / "original9_frozen"
)
PHI_NAMES = ("S_times_I",)

# Dict insertion order is part of the fixed score definition.
METRIC_SPECS: dict[str, dict[str, str]] = {
    "martingale_I_mean": {
        "layer": "martingale",
        "tail": "centered",
        "score_role": "primary",
    },
    "martingale_I_max_abs_cum": {
        "layer": "martingale",
        "tail": "upper",
        "score_role": "primary",
    },
    "bracket_I_mean_z2": {
        "layer": "bracket",
        "tail": "centered",
        "score_role": "primary",
    },
    "bracket_I_max_abs_cum": {
        "layer": "bracket",
        "tail": "upper",
        "score_role": "primary",
    },
    "bracket_I_energy_acf1": {
        "layer": "bracket",
        "tail": "centered",
        "score_role": "primary",
    },
    "bracket_I_energy_acf7": {
        "layer": "bracket",
        "tail": "centered",
        "score_role": "primary",
    },
    "dynkin_S_times_I_mean": {
        "layer": "generator",
        "tail": "centered",
        "score_role": "diagnostic",
    },
    "dynkin_S_times_I_max_abs_cum": {
        "layer": "generator",
        "tail": "upper",
        "score_role": "diagnostic",
    },
}

LAYER_ORDER = ("martingale", "bracket", "generator")
PRIMARY_COMPONENTS = tuple(
    metric
    for metric, spec in METRIC_SPECS.items()
    if spec["score_role"] == "primary"
)
DEVELOPMENT_ENDPOINT_DECISION = {
    "exploratory_bank": "100 fitted-null paths per original-nine country",
    "candidate_family": (
        "martingale I, bracket I, Dynkin I, and Dynkin S_times_I path functionals"
    ),
    "pooled_null_correlations": {
        "dynkin_I_mean_vs_dynkin_S_times_I_mean": 0.999995,
        "martingale_I_mean_vs_dynkin_S_times_I_mean": 0.999708,
        "martingale_I_max_vs_dynkin_S_times_I_max": 0.997654,
    },
    "decision": (
        "Use the six martingale/bracket components as the primary global score. "
        "Retain one nonlinear S_times_I Dynkin layer as a calibrated structural "
        "diagnostic, but exclude it from the global maximum to avoid counting "
        "the same scalar-driver innovation repeatedly."
    ),
    "reason": (
        "For a one-dimensional Brownian driver, standardized nondegenerate "
        "test-function martingales share the same leading innovation."
    ),
}


def _country_seed(base_seed: int, country: str, purpose: str) -> int:
    # This digest is a deterministic seed derivation, not an integrity check.
    # Changing it changes the random stream and therefore the experiment.
    import hashlib

    token = f"{int(base_seed)}:{country.upper()}:{purpose}".encode("ascii")
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return int((int(base_seed) + offset) % (2**31 - 1))


def _config(val_country: str) -> config_module.Config:
    updates = {
        **bi.CASES["recov28_frzgamma"],
        "VAL_COUNTRY": val_country.upper(),
        "TRAIN_COUNTRIES": list(DEVELOPMENT_COUNTRIES),
        # This base path is not used to claim a country-specific development
        # fit.  The common byte-identical original-nine fit is loaded below.
        "MODEL_SAVE_PATH": MODEL_ROOT / "neural_sde_sir_valFIN.eqx",
    }
    return apply_config_updates(updates)


def _endpoint_settings(*, n_bootstrap: int, seed: int) -> dict[str, object]:
    return {
        "case": "recov28_frzgamma",
        "development_countries": list(DEVELOPMENT_COUNTRIES),
        "confirmatory_countries": list(CONFIRMATORY_COUNTRIES),
        "audit_horizon_days": 60,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "null_bank_split": {
            "pilot": "even bootstrap_id",
            "evaluation": "odd bootstrap_id",
            "pilot_role": "component median and sample-standard-deviation only",
            "evaluation_role": "finite-sample upper ranks only",
            "plus_one_rank_rule": True,
        },
        "map": {
            "solver": "projected Euler-Maruyama",
            "substep_dt": 0.1,
            "num_substeps": 10,
            "scalar_common_driver": True,
            "controls": "causal and recomputed daily on every path",
            "martingale_coordinate": (
                "(Y_I,next - projected deterministic Y_I,next) / "
                "sqrt(projected tangent Sigma_II)"
            ),
            "tangent_recursion": (
                "Sigma[l+1]=A[l] Sigma[l] A[l]^T + h q[l]q[l]^T; "
                "A=D(P o EulerDrift), q=DP*g"
            ),
            "generator_rule": "left-point sum of grad(phi).b + 0.5*g^T Hess(phi) g",
            "generator_qv_rule": "left-point sum of (grad(phi).g)^2",
            "test_functions": list(PHI_NAMES),
            "diffusion_regularization_added_to_simulator_or_map": False,
            "projection_non_gaussianity_handled_by": "matched fitted-null simulation",
            "tangent_covariance_claimed_exact": False,
        },
        "metric_specs": METRIC_SPECS,
        "primary_global_components": list(PRIMARY_COMPONENTS),
        "development_endpoint_decision": DEVELOPMENT_ENDPOINT_DECISION,
        "layer_order": list(LAYER_ORDER),
        "score_rule": {
            "component_departure": "pilot-null standardized, tail transformed",
            "layer_score": "maximum component departure within layer",
            "global_score": "maximum departure across the six primary components",
            "multiplicity_calibration": "same maximum applied to every odd null path",
            "generator_role": (
                "separately ranked structural diagnostic; excluded from global score "
                "because original-nine development showed scalar-driver redundancy"
            ),
        },
        "alignment": {
            "role": "implementation diagnostic; excluded from endpoint score",
            "objects": [
                "training projected deterministic mean and diagonal plug-in variance",
                "projected tangent mean and full covariance",
                "conditional Monte Carlo mean and covariance from the fitted simulator",
            ],
        },
    }


def _load_common_original9_model(
    cfg: config_module.Config,
    data: config_module.TrainingData,
):
    """Load the common nine-country fit without reading a Nordic trajectory.

    The on-disk file name contains ``valFIN`` for historical reasons.  All
    three corrected checkpoint files have identical bytes.  For development
    we deserialize against a template built solely from the requested
    development-country data.  The training cohort and solver settings are
    explicit in ``DEVELOPMENT_COUNTRIES`` and ``_config``; no checkpoint
    sidecar is required to run the analysis.
    """

    if not COMMON_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing corrected common model: {COMMON_MODEL_PATH}")

    template = build_like_models(cfg, data)
    model = deserialize_models(COMMON_MODEL_PATH, template)[0]
    sir_training.assert_model_normalization(model, data)
    return model


def _make_context(model, data: config_module.TrainingData, cfg: config_module.Config):
    raw_lookback = sir_data.get_sir_control_lookback(cfg) + (
        1 if data.ar_phi is not None else 0
    )
    dt = float(cfg.SDE_SUBSTEP_DT)
    n_substeps = int(cfg.SDE_NUM_SUBSTEPS)
    sqrt_dt = jnp.sqrt(dt)
    identity = jnp.eye(cfg.STATE_SIZE)
    norm_mean = jnp.asarray(data.norm_mean)
    norm_std = jnp.asarray(data.norm_std)
    lower = jnp.array([1e-6, -20.0])
    upper = jnp.array([1.0 - 1e-6, 0.0])
    ar_phi = None if data.ar_phi is None else jnp.asarray(data.ar_phi)
    country_idx = jnp.array(
        -1 if data.val_country_id is None else data.val_country_id,
        dtype=jnp.int32,
    )
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

    def project(state):
        physical = state * norm_std + norm_mean
        clipped = jnp.clip(physical, lower, upper)
        was_clipped = jnp.any(jnp.abs(clipped - physical) > 1e-12)
        return (clipped - norm_mean) / norm_std, was_clipped

    def phi_terms(state):
        s = state[0] * norm_std[0] + norm_mean[0]
        z = state[1] * norm_std[1] + norm_mean[1]
        infectious = jnp.exp(jnp.clip(z, -20.0, 5.0))
        grad_i = jnp.array([0.0, infectious * norm_std[1]])
        hess_i = jnp.array(
            [[0.0, 0.0], [0.0, infectious * norm_std[1] ** 2]]
        )
        grad_si = jnp.array(
            [norm_std[0] * infectious, s * infectious * norm_std[1]]
        )
        cross = norm_std[0] * infectious * norm_std[1]
        hess_si = jnp.array(
            [[0.0, cross], [cross, s * infectious * norm_std[1] ** 2]]
        )
        del grad_i, hess_i
        values = jnp.array([s * infectious])
        grads = jnp.stack([grad_si])
        hessians = jnp.stack([hess_si])
        return values, grads, hessians

    def transition_terms(state, control_vec):
        """Projected deterministic location and two finite-step variance maps."""

        def micro_step(carry, _unused):
            y_curr, tangent_cov, plugin_var, gen_int, gen_qv = carry
            drift = model._calculate_drift(
                y_curr, control_vec, country_idx=country_idx
            )
            diffusion = model._calculate_diffusion(
                y_curr, control_vec, country_idx=country_idx
            )
            drift_jac = jax.jacfwd(
                lambda y: model._calculate_drift(
                    y, control_vec, country_idx=country_idx
                )
            )(y_curr)
            proposed = y_curr + drift * dt
            proposed_phys = proposed * norm_std + norm_mean
            projection_derivative = (
                (proposed_phys > lower) & (proposed_phys < upper)
            ).astype(y_curr.dtype)
            projection_jac = jnp.diag(projection_derivative)
            step_jac = projection_jac @ (identity + dt * drift_jac)
            projected_diffusion = projection_derivative * diffusion
            next_cov = (
                step_jac @ tangent_cov @ step_jac.T
                + dt * jnp.outer(projected_diffusion, projected_diffusion)
            )
            next_state, _ = project(proposed)

            _phi, grads, hessians = phi_terms(y_curr)
            generator = grads @ drift + 0.5 * jnp.einsum(
                "i,pij,j->p", diffusion, hessians, diffusion
            )
            qv_rate = (grads @ diffusion) ** 2
            return (
                next_state,
                0.5 * (next_cov + next_cov.T),
                plugin_var + dt * diffusion**2,
                gen_int + dt * generator,
                gen_qv + dt * qv_rate,
            ), None

        initial = (
            state,
            jnp.zeros((cfg.STATE_SIZE, cfg.STATE_SIZE), dtype=state.dtype),
            jnp.zeros((cfg.STATE_SIZE,), dtype=state.dtype),
            jnp.zeros((len(PHI_NAMES),), dtype=state.dtype),
            jnp.zeros((len(PHI_NAMES),), dtype=state.dtype),
        )
        final, _ = jax.lax.scan(micro_step, initial, jnp.arange(n_substeps))
        return final

    def evaluate_future(initial_history, future_states, start_weekday):
        def day_step(history, day_input):
            next_state, offset = day_input
            state = history[-1]
            control_vec = control_vector(history, (start_weekday + offset) % 7)
            predicted, tangent_cov, plugin_var, gen_int, gen_qv = transition_terms(
                state, control_vec
            )
            tangent_var_i = jnp.maximum(tangent_cov[1, 1], 1e-12)
            z_i = (next_state[1] - predicted[1]) / jnp.sqrt(tangent_var_i)
            phi0, _grad0, _hess0 = phi_terms(state)
            phi1, _grad1, _hess1 = phi_terms(next_state)
            dynkin_increment = phi1 - phi0 - gen_int
            dynkin_z = dynkin_increment / jnp.sqrt(jnp.maximum(gen_qv, 1e-20))
            next_history = jnp.roll(history, shift=-1, axis=0).at[-1].set(
                next_state
            )
            return next_history, (
                z_i,
                dynkin_z,
                predicted,
                tangent_cov,
                plugin_var,
                gen_int,
                gen_qv,
            )

        _, outputs = jax.lax.scan(
            day_step,
            initial_history,
            (future_states, jnp.arange(future_states.shape[0])),
        )
        return outputs

    def simulate_one(initial_history, path_noise, start_weekday):
        def day_step(history, day_input):
            day_noise, offset = day_input
            control_vec = control_vector(history, (start_weekday + offset) % 7)

            def stochastic_step(carry, noise):
                state, n_clipped = carry
                drift = model._calculate_drift(
                    state, control_vec, country_idx=country_idx
                )
                diffusion = model._calculate_diffusion(
                    state, control_vec, country_idx=country_idx
                )
                proposed = state + drift * dt + diffusion * noise * sqrt_dt
                next_state, was_clipped = project(proposed)
                return (
                    next_state,
                    n_clipped + was_clipped.astype(jnp.int32),
                ), None

            (next_state, n_clipped), _ = jax.lax.scan(
                stochastic_step,
                (history[-1], jnp.zeros((), dtype=jnp.int32)),
                day_noise,
            )
            next_history = jnp.roll(history, shift=-1, axis=0).at[-1].set(
                next_state
            )
            return next_history, (next_state, n_clipped)

        _, outputs = jax.lax.scan(
            day_step,
            initial_history,
            (path_noise, jnp.arange(path_noise.shape[0])),
        )
        future, clipped = outputs
        return future, jnp.sum(clipped)

    def simulate_one_day(state, control_vec, day_noise):
        def stochastic_step(carry, noise):
            current, n_clipped = carry
            drift = model._calculate_drift(
                current, control_vec, country_idx=country_idx
            )
            diffusion = model._calculate_diffusion(
                current, control_vec, country_idx=country_idx
            )
            proposed = current + drift * dt + diffusion * noise * sqrt_dt
            next_state, was_clipped = project(proposed)
            return (
                next_state,
                n_clipped + was_clipped.astype(jnp.int32),
            ), None

        return jax.lax.scan(
            stochastic_step,
            (state, jnp.zeros((), dtype=jnp.int32)),
            day_noise,
        )[0]

    return {
        "raw_lookback": raw_lookback,
        "control_vector": control_vector,
        "transition_terms": transition_terms,
        "evaluate_future": evaluate_future,
        "simulate_one": simulate_one,
        "simulate_one_day": simulate_one_day,
        "n_substeps": n_substeps,
    }


def _lag_corr(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Lag correlation requires a finite one-dimensional array.")
    if lag <= 0:
        raise ValueError("Correlation lag must be positive.")
    if values.size - lag < 2:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    correlation = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("Non-finite lag correlation.")
    return correlation


def compute_path_metrics(z_i: np.ndarray, dynkin_z: np.ndarray) -> dict[str, float]:
    z_i = np.asarray(z_i, dtype=float)
    dynkin_z = np.asarray(dynkin_z, dtype=float)
    if z_i.ndim != 1 or dynkin_z.shape != (z_i.size, len(PHI_NAMES)):
        raise ValueError("Unexpected audit-path array shapes.")
    if not np.isfinite(z_i).all() or not np.isfinite(dynkin_z).all():
        raise ValueError("Audit path contains non-finite standardized increments.")
    n = z_i.size
    root_n = np.sqrt(max(n, 1))
    energy = z_i**2 - 1.0
    out = {
        "martingale_I_mean": float(np.mean(z_i)),
        "martingale_I_max_abs_cum": float(
            np.max(np.abs(np.cumsum(z_i))) / root_n
        ),
        "bracket_I_mean_z2": float(np.mean(z_i**2)),
        "bracket_I_max_abs_cum": float(
            np.max(np.abs(np.cumsum(energy))) / root_n
        ),
        "bracket_I_energy_acf1": _lag_corr(energy, 1),
        "bracket_I_energy_acf7": _lag_corr(energy, 7),
    }
    for phi_idx, phi in enumerate(PHI_NAMES):
        values = dynkin_z[:, phi_idx]
        out[f"dynkin_{phi}_mean"] = float(np.mean(values))
        out[f"dynkin_{phi}_max_abs_cum"] = float(
            np.max(np.abs(np.cumsum(values))) / root_n
        )
    if tuple(out) != tuple(METRIC_SPECS):
        raise AssertionError("Metric implementation and frozen specification diverged.")
    return out


def _upper_rank(observed: float, reference: np.ndarray) -> float:
    return upper_rank(observed, reference)


def build_split_scores(
    observed: dict[str, float], draws: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame]:
    ids = pd.to_numeric(draws["bootstrap_id"], errors="raise").astype(int)
    pilot = draws.loc[ids.mod(2).eq(0)].copy()
    evaluation = draws.loc[ids.mod(2).eq(1)].copy()
    if pilot.empty or evaluation.empty:
        raise ValueError("Both even pilot and odd evaluation banks are required.")

    observed_departure: dict[str, float] = {}
    evaluation_departure: dict[str, np.ndarray] = {}
    rows = []
    for metric, spec in METRIC_SPECS.items():
        if not np.isfinite(observed[metric]):
            raise ValueError(f"Non-finite observed metric value for {metric}.")
        pilot_values = pd.to_numeric(pilot[metric], errors="coerce").to_numpy(float)
        eval_values = pd.to_numeric(evaluation[metric], errors="coerce").to_numpy(float)
        if not np.isfinite(pilot_values).all() or not np.isfinite(eval_values).all():
            raise ValueError(f"Non-finite null metric values for {metric}.")
        center = float(np.median(pilot_values))
        scale = float(np.std(pilot_values, ddof=1))
        if not np.isfinite(scale) or scale <= 1e-10:
            raise ValueError(f"Degenerate pilot scale for {metric}: {scale}")
        obs_dep = float(
            _departure(np.array([observed[metric]]), center, scale, spec["tail"])[0]
        )
        eval_dep = _departure(eval_values, center, scale, spec["tail"])
        observed_departure[metric] = obs_dep
        evaluation_departure[metric] = eval_dep
        rows.append(
            {
                "metric": metric,
                "layer": spec["layer"],
                "tail": spec["tail"],
                "score_role": spec["score_role"],
                "observed": float(observed[metric]),
                "pilot_n": int(len(pilot_values)),
                "pilot_median": center,
                "pilot_std": scale,
                "evaluation_n": int(len(eval_values)),
                "evaluation_q025": float(np.quantile(eval_values, 0.025)),
                "evaluation_q500": float(np.quantile(eval_values, 0.500)),
                "evaluation_q975": float(np.quantile(eval_values, 0.975)),
                "observed_departure": obs_dep,
                "component_rank_pvalue": _upper_rank(obs_dep, eval_dep),
            }
        )

    eval_matrix = np.column_stack(
        [evaluation_departure[metric] for metric in METRIC_SPECS]
    )
    obs_vector = np.array([observed_departure[m] for m in METRIC_SPECS])
    primary_indices = [
        tuple(METRIC_SPECS).index(metric) for metric in PRIMARY_COMPONENTS
    ]
    global_null = np.max(eval_matrix[:, primary_indices], axis=1)
    global_observed = float(np.max(obs_vector[primary_indices]))
    dominant = PRIMARY_COMPONENTS[int(np.argmax(obs_vector[primary_indices]))]
    score: dict[str, object] = {
        "score_rule": "split_calibrated_max_null_standardized_departure",
        "primary_component_metrics": ",".join(PRIMARY_COMPONENTS),
        "diagnostic_component_metrics": ",".join(
            metric
            for metric, spec in METRIC_SPECS.items()
            if spec["score_role"] == "diagnostic"
        ),
        "pilot_n": int(len(pilot)),
        "evaluation_n": int(len(evaluation)),
        "observed_global_score": global_observed,
        "global_rank_pvalue": _upper_rank(global_observed, global_null),
        "global_null_q500": float(np.quantile(global_null, 0.5)),
        "global_null_q950": float(np.quantile(global_null, 0.95)),
        "dominant_metric": dominant,
        "dominant_departure": float(observed_departure[dominant]),
    }

    summary = pd.DataFrame(rows)
    summary["primary_global_max_adjusted_pvalue"] = [
        _upper_rank(float(row.observed_departure), global_null)
        if row.score_role == "primary"
        else float("nan")
        for row in summary.itertuples(index=False)
    ]
    for layer in LAYER_ORDER:
        layer_metrics = [
            metric
            for metric, spec in METRIC_SPECS.items()
            if spec["layer"] == layer
        ]
        indices = [tuple(METRIC_SPECS).index(metric) for metric in layer_metrics]
        obs_layer = float(np.max(obs_vector[indices]))
        null_layer = np.max(eval_matrix[:, indices], axis=1)
        score[f"{layer}_observed_score"] = obs_layer
        score[f"{layer}_rank_pvalue"] = _upper_rank(obs_layer, null_layer)
        score[f"{layer}_dominant_metric"] = layer_metrics[
            int(np.argmax(obs_vector[indices]))
        ]
    return score, summary


def _evaluate_country_paths(
    model,
    data: config_module.TrainingData,
    cfg: config_module.Config,
    *,
    n_bootstrap: int,
    seed: int,
    retain_null_z: bool = False,
    start_pos: int | None = None,
    horizon: int | None = None,
):
    context = _make_context(model, data, cfg)
    if start_pos is None:
        start_pos = bi._start_position(data, cfg)
    else:
        start_pos = int(start_pos)
    raw_lookback = int(context["raw_lookback"])
    if start_pos < raw_lookback - 1 or start_pos >= len(data.val_features_df) - 1:
        raise ValueError(f"Invalid audit start position or insufficient history: {start_pos}")
    available = len(data.val_features_df) - 1 - start_pos
    if horizon is None:
        horizon = min(cfg.FORECAST_HORIZON, available)
    else:
        horizon = int(horizon)
        if horizon > available:
            raise ValueError("Requested audit window extends beyond the available trajectory.")
    if horizon < 10:
        raise ValueError("Insufficient country trajectory for SDE-native audit.")
    initial_history = jnp.asarray(
        data.val_features_df.iloc[
            start_pos - raw_lookback + 1 : start_pos + 1
        ].values
    )
    observed_future = jnp.asarray(
        data.val_features_df.iloc[start_pos + 1 : start_pos + horizon + 1].values
    )
    start_weekday = jnp.array(
        int(data.val_features_df.index[start_pos].weekday()), dtype=jnp.int32
    )

    evaluate_future: Callable = context["evaluate_future"]
    simulate_one: Callable = context["simulate_one"]

    @eqx.filter_jit
    def evaluate_observed(future):
        return evaluate_future(initial_history, future, start_weekday)

    @eqx.filter_jit
    def simulate_batch(noise):
        return jax.vmap(
            lambda one_noise: simulate_one(
                initial_history, one_noise, start_weekday
            )
        )(noise)

    @eqx.filter_jit
    def evaluate_batch(futures):
        return jax.vmap(
            lambda future: evaluate_future(
                initial_history, future, start_weekday
            )
        )(futures)

    observed_outputs = evaluate_observed(observed_future)
    noise = jax.random.normal(
        jax.random.PRNGKey(seed),
        shape=(n_bootstrap, horizon, int(context["n_substeps"])),
    )
    simulated_future, clipped = simulate_batch(noise)
    null_outputs = evaluate_batch(simulated_future)

    observed_z = np.asarray(observed_outputs[0])
    observed_dynkin = np.asarray(observed_outputs[1])
    null_z = np.asarray(null_outputs[0])
    null_dynkin = np.asarray(null_outputs[1])
    clip_fraction = np.asarray(clipped, dtype=float) / float(
        horizon * int(context["n_substeps"])
    )
    observed_metrics = compute_path_metrics(observed_z, observed_dynkin)
    draw_rows = []
    for bootstrap_id in range(n_bootstrap):
        draw_rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "clip_fraction": float(clip_fraction[bootstrap_id]),
                **compute_path_metrics(
                    null_z[bootstrap_id], null_dynkin[bootstrap_id]
                ),
            }
        )
    draws = pd.DataFrame(draw_rows)
    score, component_summary = build_split_scores(observed_metrics, draws)
    raw = {
        "observed_z_I": observed_z,
        "observed_dynkin_z": observed_dynkin,
        "observed_predicted_state": np.asarray(observed_outputs[2]),
        "observed_tangent_cov": np.asarray(observed_outputs[3]),
        "observed_training_plugin_var": np.asarray(observed_outputs[4]),
        "observed_generator_integral": np.asarray(observed_outputs[5]),
        "observed_generator_qv": np.asarray(observed_outputs[6]),
    }
    if retain_null_z:
        raw["null_z_I"] = null_z
    dates = data.val_features_df.index[start_pos + 1 : start_pos + horizon + 1]
    return {
        "start_pos": start_pos,
        "start_date": str(data.val_features_df.index[start_pos].date()),
        "end_date": str(data.val_features_df.index[start_pos + horizon].date()),
        "horizon": horizon,
        "dates": dates,
        "observed_metrics": observed_metrics,
        "draws": draws,
        "score": score,
        "component_summary": component_summary,
        "raw": raw,
    }


def _alignment_rows(
    model,
    data: config_module.TrainingData,
    cfg: config_module.Config,
    *,
    start_pos: int,
    horizon: int,
    n_anchors: int,
    n_mc: int,
    seed: int,
) -> pd.DataFrame:
    context = _make_context(model, data, cfg)
    raw_lookback = int(context["raw_lookback"])
    anchors = np.unique(
        np.linspace(0, max(horizon - 1, 0), num=n_anchors, dtype=int)
    )
    transition_terms: Callable = context["transition_terms"]
    control_vector: Callable = context["control_vector"]
    simulate_one_day: Callable = context["simulate_one_day"]

    @eqx.filter_jit
    def one_day_batch(state, control, noise):
        return jax.vmap(
            lambda one_noise: simulate_one_day(state, control, one_noise)
        )(noise)

    rows = []
    for anchor_number, offset in enumerate(anchors):
        pos = start_pos + int(offset)
        history = jnp.asarray(
            data.val_features_df.iloc[pos - raw_lookback + 1 : pos + 1].values
        )
        weekday = int(data.val_features_df.index[pos].weekday())
        control = control_vector(history, weekday)
        predicted, tangent_cov, plugin_var, _gen, _qv = transition_terms(
            history[-1], control
        )
        anchor_seed = _country_seed(seed + anchor_number, cfg.VAL_COUNTRY, "alignment")
        noise = jax.random.normal(
            jax.random.PRNGKey(anchor_seed),
            shape=(n_mc, int(context["n_substeps"])),
        )
        mc_states, mc_clipped = one_day_batch(history[-1], control, noise)
        mc_np = np.asarray(mc_states, dtype=float)
        mc_mean = np.mean(mc_np, axis=0)
        mc_cov = np.cov(mc_np, rowvar=False, ddof=1)
        tangent_np = np.asarray(tangent_cov, dtype=float)
        plugin_cov = np.diag(np.asarray(plugin_var, dtype=float))
        predicted_np = np.asarray(predicted, dtype=float)
        mc_scale = max(float(np.linalg.norm(mc_cov, ord="fro")), 1e-12)
        mc_var_i = max(float(mc_cov[1, 1]), 1e-12)
        pinv_cov = np.linalg.pinv(mc_cov, rcond=1e-10)
        mean_delta = predicted_np - mc_mean
        rows.append(
            {
                "country": cfg.VAL_COUNTRY,
                "anchor_number": int(anchor_number),
                "offset": int(offset),
                "date": str(data.val_features_df.index[pos].date()),
                "n_mc": int(n_mc),
                "mc_clip_fraction": float(
                    np.sum(np.asarray(mc_clipped))
                    / (n_mc * int(context["n_substeps"]))
                ),
                "mean_bias_I_in_mc_sd": float(mean_delta[1] / np.sqrt(mc_var_i)),
                "mean_bias_mahalanobis": float(
                    np.sqrt(max(mean_delta @ pinv_cov @ mean_delta, 0.0))
                ),
                "tangent_cov_relative_frobenius_error": float(
                    np.linalg.norm(tangent_np - mc_cov, ord="fro") / mc_scale
                ),
                "training_plugin_cov_relative_frobenius_error": float(
                    np.linalg.norm(plugin_cov - mc_cov, ord="fro") / mc_scale
                ),
                "tangent_I_variance_ratio_to_mc": float(
                    tangent_np[1, 1] / mc_var_i
                ),
                "training_plugin_I_variance_ratio_to_mc": float(
                    plugin_cov[1, 1] / mc_var_i
                ),
                "mc_cov_SI": float(mc_cov[0, 1]),
                "tangent_cov_SI": float(tangent_np[0, 1]),
                "training_plugin_cov_SI": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _write_country_artifacts(
    country: str,
    phase_dir: Path,
    result: dict,
    alignment: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    n_alignment_mc: int,
) -> dict[str, object]:
    country_dir = phase_dir / country
    country_dir.mkdir(parents=True, exist_ok=True)
    draws = result["draws"].copy()
    draws.insert(0, "country", country)
    draws.to_csv(country_dir / "null_metric_draws.csv", index=False)
    component = result["component_summary"].copy()
    component.insert(0, "country", country)
    component.to_csv(country_dir / "component_summary.csv", index=False)
    score = {"country": country, **result["score"]}
    pd.DataFrame([score]).to_csv(country_dir / "layer_and_global_scores.csv", index=False)
    alignment.to_csv(country_dir / "finite_step_alignment.csv", index=False)
    np.savez(country_dir / "observed_path_diagnostics.npz", **result["raw"])

    metadata = {
        "country": country,
        "case": "recov28_frzgamma",
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "horizon": int(result["horizon"]),
        "n_bootstrap": int(n_bootstrap),
        "country_seed": int(seed),
        "n_alignment_mc": int(n_alignment_mc),
        "model_path": str(COMMON_MODEL_PATH.relative_to(ROOT)),
        "model_training_countries": list(DEVELOPMENT_COUNTRIES),
        "model_label_caveat": (
            "The source filename says valFIN, but its bytes are the common "
            "original-nine-trained fit and are identical to the NOR/SWE copies. "
            "Training ran for a fixed 60 epochs; validation was diagnostic and "
            "did not select optimizer updates or a best checkpoint."
        ),
        "observed_metrics": result["observed_metrics"],
        "score": score,
        "mean_null_clip_fraction": float(draws["clip_fraction"].mean()),
        "max_null_clip_fraction": float(draws["clip_fraction"].max()),
    }
    (country_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def run_countries(
    countries: tuple[str, ...],
    phase_dir: Path,
    *,
    n_bootstrap: int,
    seed: int,
    n_alignment_anchors: int,
    n_alignment_mc: int,
) -> None:
    metadata_rows = []
    all_scores = []
    all_components = []
    all_alignment = []

    for country in countries:
        print(f"=== SDE-native audit: {country} ===", flush=True)
        cfg = _config(country)
        bi._validate_supported_config(cfg)
        data = load_cached_data(cfg)
        model = _load_common_original9_model(cfg, data)
        country_null_seed = _country_seed(seed, country, "fitted-null")
        result = _evaluate_country_paths(
            model,
            data,
            cfg,
            n_bootstrap=n_bootstrap,
            seed=country_null_seed,
        )
        alignment = _alignment_rows(
            model,
            data,
            cfg,
            start_pos=int(result["start_pos"]),
            horizon=int(result["horizon"]),
            n_anchors=n_alignment_anchors,
            n_mc=n_alignment_mc,
            seed=seed,
        )
        metadata = _write_country_artifacts(
            country,
            phase_dir,
            result,
            alignment,
            n_bootstrap=n_bootstrap,
            seed=country_null_seed,
            n_alignment_mc=n_alignment_mc,
        )
        metadata_rows.append(metadata)
        all_scores.append({"country": country, **result["score"]})
        all_components.append(
            result["component_summary"].assign(country=country)
        )
        all_alignment.append(alignment)
        print(
            pd.Series(
                {
                    "global_p": result["score"]["global_rank_pvalue"],
                    "martingale_p": result["score"]["martingale_rank_pvalue"],
                    "bracket_p": result["score"]["bracket_rank_pvalue"],
                    "generator_p": result["score"]["generator_rank_pvalue"],
                    "dominant": result["score"]["dominant_metric"],
                }
            ).to_string(),
            flush=True,
        )

    phase_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_scores).to_csv(phase_dir / "country_scores.csv", index=False)
    pd.concat(all_components, ignore_index=True).to_csv(
        phase_dir / "component_summaries.csv", index=False
    )
    alignment_all = pd.concat(all_alignment, ignore_index=True)
    alignment_all.to_csv(phase_dir / "finite_step_alignment_all.csv", index=False)
    alignment_summary = (
        alignment_all.groupby("country", sort=False)
        .agg(
            n_anchors=("anchor_number", "size"),
            median_abs_mean_bias_I_in_mc_sd=(
                "mean_bias_I_in_mc_sd",
                lambda x: float(np.median(np.abs(x))),
            ),
            median_tangent_cov_relative_error=(
                "tangent_cov_relative_frobenius_error",
                "median",
            ),
            median_training_plugin_cov_relative_error=(
                "training_plugin_cov_relative_frobenius_error",
                "median",
            ),
            median_tangent_I_variance_ratio=(
                "tangent_I_variance_ratio_to_mc",
                "median",
            ),
            median_training_plugin_I_variance_ratio=(
                "training_plugin_I_variance_ratio_to_mc",
                "median",
            ),
            max_mc_clip_fraction=("mc_clip_fraction", "max"),
        )
        .reset_index()
    )
    alignment_summary.to_csv(
        phase_dir / "finite_step_alignment_summary.csv", index=False
    )
    run_summary = {
        "countries": list(countries),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "n_alignment_anchors": int(n_alignment_anchors),
        "n_alignment_mc": int(n_alignment_mc),
        "country_metadata": metadata_rows,
    }
    (phase_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )


def _check_development_outputs(
    out_root: Path,
    *,
    n_bootstrap: int,
    seed: int,
    n_alignment_anchors: int,
    n_alignment_mc: int,
) -> Path:
    development_dir = out_root / "development_original9"
    required_top = (
        "country_scores.csv",
        "component_summaries.csv",
        "finite_step_alignment_all.csv",
        "finite_step_alignment_summary.csv",
        "run_summary.json",
    )
    missing = [name for name in required_top if not (development_dir / name).exists()]
    missing.extend(
        f"{country}/metadata.json"
        for country in DEVELOPMENT_COUNTRIES
        if not (development_dir / country / "metadata.json").exists()
    )
    if missing:
        raise FileNotFoundError(
            "Development artifacts are incomplete: " + ", ".join(missing)
        )

    run_summary = json.loads(
        (development_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    expected = {
        "countries": list(DEVELOPMENT_COUNTRIES),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "n_alignment_anchors": int(n_alignment_anchors),
        "n_alignment_mc": int(n_alignment_mc),
    }
    for key, value in expected.items():
        if run_summary.get(key) != value:
            raise ValueError(
                f"Development run summary mismatch for {key}: "
                f"{run_summary.get(key)!r} != {value!r}"
            )
    return development_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--n-alignment-anchors", type=int, default=8)
    parser.add_argument("--n-alignment-mc", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 4 or args.n_bootstrap % 2:
        raise ValueError("--n-bootstrap must be an even integer of at least four.")
    if args.n_alignment_anchors < 1 or args.n_alignment_mc < 8:
        raise ValueError("Alignment requires at least one anchor and eight MC paths.")

    if args.phase == "develop":
        run_countries(
            DEVELOPMENT_COUNTRIES,
            args.out_root / "development_original9",
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            n_alignment_anchors=args.n_alignment_anchors,
            n_alignment_mc=args.n_alignment_mc,
        )
        return

    _check_development_outputs(
        args.out_root,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        n_alignment_anchors=args.n_alignment_anchors,
        n_alignment_mc=args.n_alignment_mc,
    )
    confirmatory_dir = args.out_root / "confirmatory_nordic"
    staging_dir = args.out_root / "_confirmatory_nordic_staging"
    if confirmatory_dir.exists():
        raise FileExistsError(
            "Nordic SDE-native outputs already exist; one-time evaluation will not overwrite them."
        )
    if staging_dir.exists():
        raise FileExistsError(
            "A prior Nordic staging directory exists; inspect it before retrying evaluation."
        )
    run_countries(
        CONFIRMATORY_COUNTRIES,
        staging_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        n_alignment_anchors=args.n_alignment_anchors,
        n_alignment_mc=args.n_alignment_mc,
    )
    staging_dir.replace(confirmatory_dir)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
