"""Read-only model/bank replay for the beta positive-part question.

Writes only new diagnostic reports, never frozen evidence, models or paper text.
Observed data have no observed microsteps: their microsteps below are the
zero-noise paths used to construct the score. Reference paths have both actual
stochastic microsteps and these deterministic score-construction microsteps.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from neural_sde import _SIR_BETA_SCALE

from experiments import run_sir_sde_native_country_audit as native
import config as config_module
import sir_data
import trajectory


NORDIC = ROOT / "cache/summaries/sde_native_country_audit/original9_frozen/confirmatory_nordic"
BATCH1 = ROOT / "cache/summaries/random_country_5000_reference_draws"
BATCH2 = ROOT / "cache/summaries/sde_native_random_country_replication/random_country_batch2_frozen/evaluation"


@lru_cache(maxsize=1)
def raw_snapshot():
    # Never call the download or training-cache builder: the latter overwrites
    # shared normalization files. Only the archived CSV is read here.
    return pd.read_csv(native.OWID_CACHE_PATH, parse_dates=["date"], low_memory=False)


def load_audit_data(cfg):
    """Same preprocessing as load_or_create_training_data, without train tensors/writes."""
    if cfg.USE_TEMPORAL_PREWHITEN or cfg.CONTROL_ABLATION != "none":
        raise ValueError("Unsupported preprocessing")
    country_data = sir_data.prepare_sir_countries(
        raw_snapshot(), cfg.TRAIN_COUNTRIES + [cfg.VAL_COUNTRY],
        cfg.TRAINING_START_DATE, recovery_days=cfg.SIR_RECOVERY_DAYS,
        active_threshold=cfg.SIR_ACTIVE_THRESHOLD,
        incidence_source=cfg.SIR_INCIDENCE_SOURCE,
        incidence_blend_weight=cfg.SIR_INCIDENCE_BLEND_WEIGHT,
    )
    train_log = {k: v.copy() for k, v in country_data.items() if k in cfg.TRAIN_COUNTRIES}
    for frame in train_log.values():
        frame["I"] = np.log(frame["I"])
    mean, std = sir_data.compute_norm_stats(train_log)
    val_log = country_data[cfg.VAL_COUNTRY].copy()
    val_log["I"] = np.log(val_log["I"])
    val_norm = (val_log[["S", "I"]] - mean) / (std + 1e-8)
    lookback = sir_data.get_sir_control_lookback(cfg)
    pieces = [frame.assign(country=iso) for iso, frame in country_data.items()]
    return SimpleNamespace(
        raw_sir_df=pd.concat(pieces), norm_mean=mean, norm_std=std,
        signature_sizes=[trajectory.get_signature_size(cfg.STATE_SIZE, cfg.SIGNATURE_DEPTH, augment_time=True, lead_lag=cfg.SIGNATURE_LEAD_LAG) for _ in cfg.SIGNATURE_PATH_LENGTHS],
        macro_size=config_module.get_recon_control_size(cfg),
        n_countries=len(cfg.TRAIN_COUNTRIES), val_country_id=-1, ar_phi=None,
        val_features_df=val_norm.iloc[lookback - 1:],
    )


def bank_spec(country):
    if country in ("FIN", "NOR", "SWE"):
        return NORDIC, 500, 20260718, "fitted-null"
    if country in ("BGR", "HUN", "SVK"):
        return BATCH1, 5000, 20260813, "exploratory-5000-fitted-null"
    if country in ("LTU", "MDA", "SVN"):
        return BATCH2, 5000, 20260814, "fitted-null"
    raise ValueError(country)


def summary(values):
    a = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("Nonfinite beta preactivation")
    flat = a.ravel()
    return {
        "n_substeps": int(flat.size),
        "negative_n": int(np.sum(flat < 0)),
        "exact_zero_n": int(np.sum(flat == 0)),
        "nonpositive_fraction": float(np.mean(flat <= 0)),
        "min_abs_margin": float(np.min(np.abs(flat))),
        "near_1e-6_n": int(np.sum(np.abs(flat) <= 1e-6)),
        "near_1e-4_n": int(np.sum(np.abs(flat) <= 1e-4)),
        "near_1e-3_n": int(np.sum(np.abs(flat) <= 1e-3)),
        "quantiles": dict(zip(
            ("min", "q001", "q01", "q05", "median", "q95", "q99", "q999", "max"),
            np.quantile(flat, (0, .001, .01, .05, .5, .95, .99, .999, 1)).tolist(),
        )),
        "paths_with_nonpositive": int(np.sum(np.any(a.reshape(-1, a.shape[-2] * a.shape[-1]) <= 0, axis=1)))
            if a.ndim >= 2 else None,
    }


def make_probe(model, data, cfg, context):
    dt = float(cfg.SDE_SUBSTEP_DT)
    n_steps = int(context["n_substeps"])
    sqrt_dt = jnp.sqrt(dt)
    mean, std = jnp.asarray(data.norm_mean), jnp.asarray(data.norm_std)
    lower, upper = jnp.array([1e-6, -20.0]), jnp.array([1.0 - 1e-6, 0.0])
    country_idx = jnp.array(-1 if data.val_country_id is None else data.val_country_id, dtype=jnp.int32)
    if model.use_per_country_beta or model.gamma_scale != 0 or model.use_diffusion_net:
        raise ValueError("This audit requires the reported common-beta, frozen-gamma, constant-diffusion fit")

    def prebeta(state, control):
        return model.beta_base + _SIR_BETA_SCALE * jnp.tanh(model._raw_drift_params(state, control)[0])

    def project(state):
        physical = state * std + mean
        clipped = jnp.clip(physical, lower, upper)
        return (clipped - mean) / std, jnp.any(jnp.abs(clipped - physical) > 1e-12)

    def micro_scan(state, control, noises, stochastic):
        def micro(y, noise):
            value = prebeta(y, control)
            drift = model._calculate_drift(y, control, country_idx=country_idx)
            proposed = y + drift * dt
            if stochastic:
                diffusion = model._calculate_diffusion(y, control, country_idx=country_idx)
                proposed = proposed + diffusion * noise * sqrt_dt
            next_y, clipped = project(proposed)
            return next_y, (value, clipped)
        return jax.lax.scan(micro, state, noises)

    def stochastic_path(history, noise, weekday):
        def day(hist, inputs):
            day_noise, offset = inputs
            control = context["control_vector"](hist, (weekday + offset) % 7)
            state, (values, clips) = micro_scan(hist[-1], control, day_noise, True)
            next_hist = jnp.roll(hist, -1, axis=0).at[-1].set(state)
            return next_hist, (state, values, clips)
        return jax.lax.scan(day, history, (noise, jnp.arange(noise.shape[0])))[1]

    def tangent_path(history, future, weekday):
        def day(hist, inputs):
            next_state, offset = inputs
            control = context["control_vector"](hist, (weekday + offset) % 7)
            _, (values, clips) = micro_scan(hist[-1], control, jnp.zeros(n_steps), False)
            next_hist = jnp.roll(hist, -1, axis=0).at[-1].set(next_state)
            return next_hist, (values, clips)
        return jax.lax.scan(day, history, (future, jnp.arange(future.shape[0])))[1]

    return stochastic_path, tangent_path, micro_scan


def audit_country(country, output_dir, observed_only=False):
    bank_root, n_bank, base_seed, purpose = bank_spec(country)
    cfg = native._config(country)
    cache_path = config_module.get_cache_path(cfg)
    data = native.load_cached_data(cfg) if cache_path.exists() else load_audit_data(cfg)
    model = native._load_common_original9_model(cfg, data)
    context = native._make_context(model, data, cfg)
    position = native.bi._start_position(data, cfg)
    horizon = min(cfg.FORECAST_HORIZON, len(data.val_features_df) - 1 - position)
    history = jnp.asarray(data.val_features_df.iloc[position - context["raw_lookback"] + 1:position + 1].values)
    observed = jnp.asarray(data.val_features_df.iloc[position + 1:position + horizon + 1].values)
    weekday = jnp.array(data.val_features_df.index[position].weekday(), dtype=jnp.int32)
    stochastic_path, tangent_path, micro_scan = make_probe(model, data, cfg, context)
    observed_values, observed_clips = eqx.filter_jit(tangent_path)(history, observed, weekday)
    result = {
        "country": country,
        "analysis_role": "post-hoc implementation diagnostic; no new test or model change",
        "beta_base": float(model.beta_base), "gamma_base": float(model.gamma_base),
        "gamma_scale": float(model.gamma_scale), "sigma": float(model.sigma),
        "raw_network_cutoff": float(np.arctanh(-model.beta_base / _SIR_BETA_SCALE)),
        "data_cache_path": str(cache_path),
        "data_source": "existing training cache" if cache_path.exists() else "read-only reconstruction from archived CSV; no training tensors needed",
        "start_date": str(data.val_features_df.index[position].date()),
        "end_date": str(data.val_features_df.index[position + horizon].date()),
        "horizon": horizon, "substeps_per_day": int(context["n_substeps"]),
        "jax_version": jax.__version__, "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "maximum_derivative_at_negative_zero_positive": np.asarray(jax.vmap(jax.grad(lambda x: jnp.maximum(x, 0.)))(jnp.array([-1., 0., 1.]))).tolist(),
        "observed_tangent": summary(observed_values),
        "observed_tangent_projected_substeps": int(np.sum(observed_clips)),
    }
    # A last-layer bound is useful context, but not treated as a tight network bound.
    last_layer = model.drift_net.mlp.layers[-1]
    weight_l1 = float(jnp.sum(jnp.abs(last_layer.weight[0])))
    bias = float(last_layer.bias[0])
    result["loose_global_raw_network_bounds"] = [bias - weight_l1, bias + weight_l1]
    print(country, "observed", result["observed_tangent"], flush=True)
    if not observed_only:
        seed = native._country_seed(base_seed, country, purpose)
        noise = jax.random.normal(jax.random.PRNGKey(seed), shape=(n_bank, horizon, int(context["n_substeps"])))
        sim_batch = eqx.filter_jit(jax.vmap(lambda x: stochastic_path(history, x, weekday)))
        futures, beta_sim, clip_sim = sim_batch(noise)
        beta_sim = np.asarray(beta_sim)
        print(country, "stochastic bank complete", flush=True)
        tangent_batch = eqx.filter_jit(jax.vmap(lambda x: tangent_path(history, x, weekday)))
        beta_tan, clip_tan = tangent_batch(futures)
        beta_tan = np.asarray(beta_tan)
        result.update({"n_bank": n_bank, "country_seed": seed, "reference_bank": str(bank_root / country)})
        for label, subset in (("all", slice(None)), ("pilot", slice(0, None, 2)), ("evaluation", slice(1, None, 2))):
            result[f"reference_stochastic_{label}"] = summary(beta_sim[subset])
            result[f"reference_tangent_{label}"] = summary(beta_tan[subset])
        result["reference_stochastic_projected_substeps"] = int(np.sum(clip_sim))
        result["reference_tangent_projected_substeps"] = int(np.sum(clip_tan))
        print(country, "tangent bank complete; validating original metrics", flush=True)
        # Verify instrumented replay against the native simulator, then all eight
        # archived metrics on every reference path and the observed score.
        sim_native = eqx.filter_jit(jax.vmap(lambda x: context["simulate_one"](history, x, weekday)))
        native_futures, native_clips = sim_native(noise)
        state_error = float(np.max(np.abs(np.asarray(futures) - np.asarray(native_futures))))
        if state_error != 0:
            raise ValueError(f"Instrumented/native state mismatch {state_error}")
        eval_batch = eqx.filter_jit(jax.vmap(lambda x: context["evaluate_future"](history, x, weekday)))
        outputs = eval_batch(futures)
        z, dynkin = np.asarray(outputs[0]), np.asarray(outputs[1])
        covariances = np.asarray(outputs[3])
        plugin_variances = np.asarray(outputs[4])
        full_negative = np.all(beta_tan < 0, axis=-1)
        full_positive = np.all(beta_tan > 0, axis=-1)
        mixed_or_zero = ~(full_negative | full_positive)
        result["reference_covariance_by_beta_branch"] = {}
        for branch, mask in (("all_ten_negative", full_negative), ("all_ten_positive", full_positive), ("mixed_or_zero", mixed_or_zero)):
            selected = covariances[mask]
            item = {"n_one_day_intervals": int(np.sum(mask))}
            if selected.size:
                ratio = selected[:, 1, 1] / plugin_variances[mask][:, 1]
                item.update({
                    "min_tangent_I_variance": float(selected[:, 1, 1].min()),
                    "max_abs_tangent_S_variance": float(np.abs(selected[:, 0, 0]).max()),
                    "max_abs_tangent_SI_covariance": float(np.abs(selected[:, 0, 1]).max()),
                    "tangent_I_to_training_plugin_ratio_min_median_max": np.quantile(ratio, [0, .5, 1]).tolist(),
                })
            result["reference_covariance_by_beta_branch"][branch] = item
        reproduced = pd.DataFrame([{"bootstrap_id": i, "clip_fraction": float(np.mean(np.asarray(clip_sim[i]))), **native.compute_path_metrics(z[i], dynkin[i])} for i in range(n_bank)])
        archived_path = bank_root / country / "null_metric_draws.csv"
        archived = pd.read_csv(archived_path)
        if not np.array_equal(archived["bootstrap_id"].to_numpy(), np.arange(n_bank)):
            raise ValueError("Unexpected bank order or size")
        errors = {}
        for metric in ("clip_fraction", *native.METRIC_SPECS):
            errors[metric] = float(np.max(np.abs(reproduced[metric].to_numpy() - archived[metric].to_numpy())))
            if not np.allclose(reproduced[metric], archived[metric], rtol=2e-6, atol=2e-6):
                raise ValueError(f"Historical reproduction mismatch {country}/{metric}: {errors[metric]}")
        obs_outputs = eqx.filter_jit(context["evaluate_future"])(history, observed, weekday)
        obs_metrics = native.compute_path_metrics(np.asarray(obs_outputs[0]), np.asarray(obs_outputs[1]))
        scores, _ = native.build_split_scores(obs_metrics, reproduced)
        score_path = bank_root / country / "layer_and_global_scores.csv"
        archived_score = pd.read_csv(score_path).iloc[0]
        rank_error = abs(float(scores["global_rank_pvalue"]) - float(archived_score["global_rank_pvalue"]))
        if rank_error > 1e-12:
            raise ValueError(f"Rank mismatch: {rank_error}")
        result["verification"] = {"native_state_max_abs_error": state_error, "archived_metric_max_abs_errors": errors, "archived_metric_rtol": 2e-6, "archived_metric_atol": 2e-6, "note": "Historical float32 arithmetic can differ at last bits on replay; this is numerical agreement, not byte-identical metric reproduction.", "global_rank_pvalue": float(scores["global_rank_pvalue"]), "global_rank_abs_error": rank_error}
        # The published one-day check also uses the Nordic and frozen batch-2
        # banks. Replay every stochastic microstep at all eight recorded anchors.
        if country not in ("BGR", "HUN", "SVK"):
            alignment_values = []
            for anchor_number, offset in enumerate(np.unique(np.linspace(0, horizon - 1, 8, dtype=int))):
                pos = position + int(offset)
                hist = jnp.asarray(data.val_features_df.iloc[pos - context["raw_lookback"] + 1:pos + 1].values)
                ctrl = context["control_vector"](hist, int(data.val_features_df.index[pos].weekday()))
                anchor_seed = native._country_seed(base_seed + anchor_number, country, "alignment")
                anchor_noise = jax.random.normal(jax.random.PRNGKey(anchor_seed), (512, int(context["n_substeps"])))
                anchor_run = eqx.filter_jit(jax.vmap(lambda x: micro_scan(hist[-1], ctrl, x, True)))
                _, (values, _) = anchor_run(anchor_noise)
                alignment_values.append(np.asarray(values))
            result["direct_one_day_stochastic"] = summary(np.stack(alignment_values).reshape(-1, 1, int(context["n_substeps"])))
        result["status"] = "PASS"
    else:
        result["status"] = "OBSERVED_ONLY"
    out = output_dir / f"{country}{'_observed' if observed_only else ''}.json"
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(country, result["status"], "saved", out, flush=True)
    jax.clear_caches()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=["FIN", "NOR", "SWE", "LTU", "MDA", "SVN", "BGR", "HUN", "SVK"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observed-only", action="store_true")
    args = parser.parse_args()
    # Refuse diagnostic output anywhere inside a frozen evidence/model tree.
    # Logged output must be a fresh directory under tmp.
    out = args.output_dir.resolve()
    if not out.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("Diagnostic output must be under repository tmp/")
    out.mkdir(parents=True, exist_ok=False)
    for country in args.countries:
        audit_country(country.upper(), out, args.observed_only)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
