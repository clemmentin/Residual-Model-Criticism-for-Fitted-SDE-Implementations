"""Conditional score-law pilot for the final six-feature SIR CE statistic.

The teacher is exactly in the existing fitted two-dimensional tanh model class.
Training uses the existing solver-integrated objective, with a selectable
plug-in or tangent training variance. The
held-out bank comes from the teacher, and the reference bank from each fit.
This is a controlled synthetic experiment, not a refit of epidemic data.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/mpl_ce_conditional"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import binom, ks_2samp

from experiments import bi_synthetic_solver_calibration as model

METRICS = (
    "martingale_I_mean", "martingale_I_max_abs_cum", "bracket_I_mean_z2",
    "bracket_I_max_abs_cum", "bracket_I_energy_acf1", "bracket_I_energy_acf7",
)
CENTERED = np.array([True, False, True, False, True, True])
MASK = model.width_mask(16)
ZERO_MEAN = jnp.zeros(model.INPUT_DIM)
UNIT_SCALE = jnp.ones(model.INPUT_DIM)
TRUE_SIGMA = 0.42


def teacher_parameters():
    """A fixed eight-active-unit tanh network, represented in the width-16 class.

    State-input hidden units have slope 0.5. The local drift state Jacobian at
    zero is [[-0.22, 0.5], [0.12, -0.30]]. Other active units use the existing
    causal controls. Coefficients are fixed before any diagnostic run.
    """
    w1 = np.zeros((model.INPUT_DIM, model.WIDTH_MAX))
    w2 = np.zeros((model.WIDTH_MAX, model.STATE_DIM))
    w1[0, 0], w1[1, 1] = 0.5, 0.5
    w2[0], w2[1] = [-0.44, 0.24], [1.0, -0.60]
    for unit, input_index, output_weights in (
        (2, 4, (0.18, 0.0)), (3, 5, (0.0, 0.18)),
        (4, 7, (0.0, 0.04)), (5, 8, (0.06, 0.06)),
        (6, 9, (-0.025, 0.0)), (7, 3, (0.0, 0.02)),
    ):
        w1[input_index, unit] = 1.0
        w2[unit] = output_weights
    return dict(w1=jnp.asarray(w1), b1=jnp.zeros(model.WIDTH_MAX),
                w2=jnp.asarray(w2), b2=jnp.zeros(model.STATE_DIM),
                log_sigma=jnp.array(math.log(TRUE_SIGMA - model.SIGMA_FLOOR)))


def teacher_in_fitted_inputs(teacher, mean, scale):
    """Exact affine reparameterization: input standardization retains the truth."""
    return {**teacher, "w1": scale[:, None] * teacher["w1"],
            "b1": teacher["b1"] + mean @ teacher["w1"]}


def ce_features(z):
    """Vectorized copy of the six native features; checked against native code."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] == 0 or not np.isfinite(z).all():
        raise ValueError("Expected finite paths by time residual matrix.")
    energy = z * z - 1
    root_k = math.sqrt(z.shape[1])
    fields = [z.mean(1), np.abs(np.cumsum(z, 1)).max(1) / root_k,
              (z * z).mean(1), np.abs(np.cumsum(energy, 1)).max(1) / root_k]
    for lag in (1, 7):
        if z.shape[1] - lag < 2:
            fields.append(np.zeros(z.shape[0]))
            continue
        left, right = energy[:, :-lag], energy[:, lag:]
        left = left - left.mean(1, keepdims=True)
        right = right - right.mean(1, keepdims=True)
        left_norm, right_norm = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
        regular = ((left_norm / math.sqrt(left.shape[1]) >= 1e-12)
                   & (right_norm / math.sqrt(right.shape[1]) >= 1e-12))
        fields.append(np.divide((left * right).sum(1), left_norm * right_norm,
                                out=np.zeros(z.shape[0]), where=regular))
    features = np.column_stack(fields)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite centring-energy features.")
    return features


def scores(features, centers, scales):
    if (not np.isfinite(features).all() or not np.isfinite(centers).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError("Scores require finite features and positive finite pilot scales.")
    standardized = (features - centers) / scales
    departures = np.where(CENTERED, np.abs(standardized), np.maximum(standardized, 0))
    result = departures.max(1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite centring-energy scores.")
    return result


def compare_score_laws(held, reference, reference_size, total_comparisons):
    """Two-sample CDF gap and a finite-M conditional rejection estimate.

    At held score s, B~Binomial(M, Pr_Q(S>=s)); rejection is B<=k-1.
    This integrates fresh evaluation-bank randomness rather than freezing one
    evaluation threshold. DKW envelopes account for both estimated CDFs.
    They are simultaneous over the planned score-law comparisons (union bound).
    """
    held, reference = np.asarray(held), np.asarray(reference)
    if (held.ndim != 1 or reference.ndim != 1 or held.size == 0 or reference.size == 0
            or not np.isfinite(held).all() or not np.isfinite(reference).all()):
        raise ValueError("Score comparison requires nonempty finite score vectors.")
    n_p, n_q = len(held), len(reference)
    k = math.floor(0.05 * (reference_size + 1))
    f_less = np.searchsorted(np.sort(reference), held, side="left") / n_q
    reject_prob = lambda f: binom.cdf(k - 1, reference_size, 1 - f)
    eps_p = math.sqrt(math.log(4 * total_comparisons / 0.05) / (2 * n_p))
    eps_q = math.sqrt(math.log(4 * total_comparisons / 0.05) / (2 * n_q))
    gap = float(ks_2samp(held, reference, method="asymp").statistic)
    return {
        "cdf_gap": gap,
        "cdf_gap_low": max(0.0, gap - eps_p - eps_q),
        "cdf_gap_high": min(1.0, gap + eps_p + eps_q),
        "conditional_rejection": float(reject_prob(f_less).mean()),
        "conditional_rejection_low": max(0.0, float(reject_prob(np.maximum(0, f_less-eps_q)).mean())-eps_p),
        "conditional_rejection_high": min(1.0, float(reject_prob(np.minimum(1, f_less+eps_q)).mean())+eps_p),
        "nominal_mc_target": k / (reference_size + 1),
        "dkw_cdf_radius_p": eps_p, "dkw_cdf_radius_q": eps_q,
        "held_score_mean": float(held.mean()), "reference_score_mean": float(reference.mean()),
        "reference_unique_fraction": len(np.unique(reference)) / n_q,
    }


def batched_transition_loss(params, states, controls, increments, mean, scale,
                            n_substeps, weight_decay, variance_model):
    """The existing loss, evaluating its known tanh Jacobian in batches.

    Controls are fixed within an observation interval. Computing their affine
    contribution once avoids repeating the large input matrix multiplication
    at every substep. Covariance propagation is the same 2x2 Euler recursion.
    This changes computation only; the loss and gradients are checked against
    model.transition_loss before the paired experiment.
    """
    dt = 1.0/n_substeps
    w_state = params["w1"][:2]/scale[:2, None]
    fixed = ((controls-mean[2:])/scale[2:]) @ params["w1"][2:] + params["b1"]
    fixed = fixed - mean[:2] @ w_state
    noise_variance = jnp.asarray(model.fitted_sigma(params)**2, dtype=states.dtype)*dt
    jac_weights = jnp.stack((w_state[0]*params["w2"][:, 0],
                            w_state[1]*params["w2"][:, 0],
                            w_state[0]*params["w2"][:, 1],
                            w_state[1]*params["w2"][:, 1]), axis=1)

    def body(carry, _):
        state, cov = carry
        raw_hidden = jnp.tanh(state @ w_state + fixed)
        hidden = raw_hidden*MASK
        drift = hidden @ params["w2"] + params["b2"]
        if variance_model == "tangent":
            jac = ((1-raw_hidden**2)*MASK) @ jac_weights
            a, b = 1+dt*jac[:, 0], dt*jac[:, 1]
            c, d = dt*jac[:, 2], 1+dt*jac[:, 3]
            v11, v12, v22 = cov[:, 0], cov[:, 1], cov[:, 2]
            cov = jnp.stack((a*a*v11+2*a*b*v12+b*b*v22,
                             a*c*v11+(a*d+b*c)*v12+b*d*v22,
                             c*c*v11+2*c*d*v12+d*d*v22+noise_variance), axis=1)
        return (state+dt*drift, cov), None

    (end, cov), _ = jax.lax.scan(body, (states, jnp.zeros((len(states), 3), states.dtype)),
                                None, length=n_substeps)
    var = jnp.maximum(cov[:, 2], 1e-10) if variance_model == "tangent" else model.fitted_sigma(params)**2
    residual = increments-(end-states)
    nll = 0.5*residual[:, 1]**2/var+0.5*jnp.log(var)
    l2 = jnp.sum(params["w1"]**2)+jnp.sum(params["w2"]**2)
    return jnp.mean(nll)+model.S_LOSS_SCALE*jnp.mean(residual**2)+weight_decay*l2


def train_batched_candidate(key, states, controls, increments, mean, scale,
                            n_substeps, epochs, variance_model):
    params = model.init_params(key)
    opt_state = model.OPTIMIZER.init(params)
    def body(_, carry):
        prm, opt = carry
        grad = jax.grad(batched_transition_loss)(prm, states, controls, increments,
                                                mean, scale, n_substeps, 1e-4, variance_model)
        updates, opt = model.OPTIMIZER.update(grad, opt, prm)
        updates = jax.tree_util.tree_map(lambda u: -0.003*u, updates)
        return optax.apply_updates(prm, updates), opt
    params, _ = jax.lax.fori_loop(0, epochs, body, (params, opt_state))
    return params


def make_functions(args, teacher):
    n_substeps = args.substeps
    train_variance = getattr(args, "train_variance", "plugin")

    def training_path(key):
        init_key, burn_key, continuation_key = jax.random.split(key, 3)
        x0 = model.INIT_SCALE * jax.random.normal(init_key, (model.STATE_DIM,))
        noise = jax.random.normal(burn_key, (model.LOOKBACK, n_substeps))
        def burn(y, eps):
            next_y = model.advance_observation_step(
                lambda x: model.fitted_drift(teacher, MASK, x, model.ZERO_CONTROLS,
                                             ZERO_MEAN, UNIT_SCALE), y, eps, TRUE_SIGMA)
            return next_y, next_y
        _, states = jax.lax.scan(burn, x0, noise)
        prefix = jnp.concatenate([x0[None], states])
        return model.simulate_reference_paths(
            continuation_key, teacher, MASK, ZERO_MEAN, UNIT_SCALE, prefix,
            1, args.training_steps, n_substeps)[0]

    generate_training = jax.jit(jax.vmap(training_path))
    flatten_training = jax.jit(lambda paths: tuple(
        a.reshape((-1, a.shape[-1])) for a in jax.vmap(model.path_transitions)(paths)))

    @jax.jit
    def fit(key, states, controls, increments):
        mean, scale = model.standardize_inputs(states, controls)
        if getattr(args, "training_implementation", "standard") == "batched":
            params = train_batched_candidate(key, states, controls, increments, mean, scale,
                                             n_substeps, args.epochs, train_variance)
        else:
            params = model.train_candidate(key, MASK, states, controls, increments, mean, scale,
                                           n_substeps, args.epochs, 0.003, 1e-4, train_variance)
        oracle_params = teacher_in_fitted_inputs(teacher, mean, scale)
        fitted_loss = model.transition_loss(params, MASK, states, controls, increments,
                                            mean, scale, n_substeps, 1e-4, train_variance)
        oracle_loss = model.transition_loss(oracle_params, MASK, states, controls, increments,
                                            mean, scale, n_substeps, 1e-4, train_variance)
        true_eff, true_cov = jax.vmap(lambda s, c: model.fitted_tangent_moments(
            teacher, MASK, s, c, ZERO_MEAN, UNIT_SCALE, n_substeps))(states, controls)
        true_r = increments[:, 1] - true_eff[:, 1]
        sigma_plugin_known_drift = jnp.sqrt(jnp.mean(true_r**2))
        sigma_tangent_known_drift = jnp.sqrt(jnp.mean(
            true_r**2 / (true_cov[:, 1, 1] / TRUE_SIGMA**2)))
        grad = jax.grad(model.transition_loss)(
            params, MASK, states, controls, increments, mean, scale, n_substeps, 1e-4, train_variance)
        grad_norm = jnp.sqrt(sum(jnp.sum(v*v) for v in jax.tree_util.tree_leaves(grad)))
        return params, mean, scale, jnp.stack([
            fitted_loss, oracle_loss, sigma_plugin_known_drift,
            sigma_tangent_known_drift, grad_norm])

    @jax.jit
    def simulate_batch(key, params, mean, scale, prefix):
        return model.simulate_reference_paths(key, params, MASK, mean, scale, prefix,
            args.path_batch, model.LOOKBACK + args.horizon, n_substeps)

    @jax.jit
    def residual_batch(params, mean, scale, paths):
        residual, covariance = jax.vmap(lambda path: model.path_residuals(
            params, MASK, mean, scale, path, n_substeps))(paths)
        variance = covariance[..., 1, 1]
        z = residual[..., 1] / jnp.sqrt(jnp.maximum(variance, 1e-12))
        return z, jnp.min(variance)

    return generate_training, flatten_training, fit, simulate_batch, residual_batch


def make_bank(simulate_batch, key, params, mean, scale, prefix, count, batch_size):
    parts = []
    for offset in range(0, count, batch_size):
        paths = simulate_batch(jax.random.fold_in(key, offset), params, mean, scale, prefix)
        parts.append(np.asarray(paths)[:min(batch_size, count-offset)])
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite simulated path; this fit is not discarded.")
    return result


def bank_features(residual_batch, params, mean, scale, paths, batch_size):
    parts, minimum = [], float("inf")
    for offset in range(0, len(paths), batch_size):
        chunk = paths[offset:offset+batch_size]
        n = len(chunk)
        if n < batch_size:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch_size-n, axis=0)])
        z, smallest = residual_batch(params, mean, scale, jnp.asarray(chunk))
        parts.append(ce_features(np.asarray(z)[:n]))
        minimum = min(minimum, float(smallest))
    if minimum <= 1e-12:
        raise ValueError("Variance floor was active; do not silently treat this as the same score.")
    return np.concatenate(parts), minimum


def run_checks(args, teacher):
    from experiments import run_sir_sde_native_country_audit as native
    if tuple(native.PRIMARY_COMPONENTS) != METRICS:
        raise AssertionError("Native primary component set changed.")
    if [native.METRIC_SPECS[x]["tail"] == "centered" for x in METRICS] != CENTERED.tolist():
        raise AssertionError("Native tail conventions changed.")
    rng = np.random.default_rng(2026090901)
    z = rng.normal(size=(32, args.horizon))
    z[0] = 0.6*z[0] + 0.4
    features = ce_features(z)
    original = np.array([[native.compute_path_metrics(row, np.zeros((args.horizon, 1)))[m]
                          for m in METRICS] for row in z])
    feature_error = float(np.max(np.abs(features-original)))
    center, scale = np.median(features[1:], 0), features[1:].std(0, ddof=1)
    dep = np.column_stack([native._departure(features[:,j], center[j], scale[j],
                                           native.METRIC_SPECS[m]["tail"])
                           for j, m in enumerate(METRICS)]).max(1)
    score_error = float(np.max(np.abs(dep-scores(features, center, scale))))
    mean, norm = jnp.asarray(rng.normal(size=model.INPUT_DIM)), jnp.asarray(np.exp(rng.normal(size=model.INPUT_DIM)))
    converted = teacher_in_fitted_inputs(teacher, mean, norm)
    x = jnp.asarray(rng.normal(size=(64, model.INPUT_DIM)))
    drift_0 = jax.vmap(lambda v: model.fitted_drift(teacher, MASK, v[:2], v[2:], ZERO_MEAN, UNIT_SCALE))(x)
    drift_1 = jax.vmap(lambda v: model.fitted_drift(converted, MASK, v[:2], v[2:], mean, norm))(x)
    class_error = float(jnp.max(jnp.abs(drift_0-drift_1)))
    m, k, u = 19, 1, (np.arange(200000)+0.5)/200000
    rank_integral = float(binom.cdf(k-1, m, 1-u).mean())
    if max(feature_error, score_error, class_error) > 1e-10 or abs(rank_integral-k/(m+1)) > 1e-8:
        raise AssertionError("Pilot mathematical/score implementation check failed.")
    return {"native_feature_max_abs_error": feature_error,
            "native_score_max_abs_error": score_error,
            "truth_class_reparameterization_max_abs_error": class_error,
            "uniform_score_rank_integral": rank_integral,
            "uniform_score_rank_target": k/(m+1)}


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/ce_conditional_fit_pilot")
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--train-paths", type=int, nargs="+", default=[48, 192])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--train-variance", choices=("plugin", "tangent"), default="plugin")
    parser.add_argument("--training-implementation", choices=("standard", "batched"), default="standard")
    parser.add_argument("--training-steps", type=int, default=112)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--pilot-paths", type=int, default=2500)
    parser.add_argument("--cdf-paths", type=int, default=8192)
    parser.add_argument("--reference-size", type=int, default=2500)
    parser.add_argument("--path-batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026090907)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Keep complete replicate groups and rerun any interrupted group.")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.train_paths) < 1 or args.horizon < 9 or args.replicates < 1:
        raise ValueError("Invalid pilot settings.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    teacher = teacher_parameters()
    checks = run_checks(args, teacher)
    (args.out_dir / "checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print("Checks:", json.dumps(checks), flush=True)
    if args.check_only:
        return
    if (args.out_dir / "results.csv").exists() and not args.resume:
        raise FileExistsError("Use a new output directory; existing results are preserved.")
    settings = {**vars(args), "out_dir": str(args.out_dir), "truth_sigma": TRUE_SIGMA,
                "teacher": "fixed eight-active-unit tanh drift in width-16 fitted class",
                "fit_width": 16, "learning_rate": 0.003, "weight_decay": 1e-4,
                "train_variance": args.train_variance, "score": "native six-feature S_CE, tangent scalar coordinate 2",
                "paired_design": "nested training paths; common independent prefix and held-out truth bank across training sizes",
                "cdf_error": "95% DKW envelopes simultaneous across all planned fits and oracle comparisons",
                "rejection_estimate": "Binomial integration over a fresh M-path reference bank using empirical conditional score laws",
                "scope": "synthetic pilot; no architecture selection, SIR clipping or upstream reconstruction",
                "jax_version": jax.__version__, "backend": jax.default_backend(), "python": sys.version}
    all_rows, component_rows, completed = [], [], set()
    if args.resume:
        previous = json.loads((args.out_dir / "settings.json").read_text(encoding="utf-8"))
        for key, value in settings.items():
            if key not in ("out_dir", "resume", "check_only") and previous.get(key) != value:
                raise ValueError(f"Cannot resume with changed setting: {key}")
        def read_rows(name):
            path = args.out_dir / name
            if not path.exists():
                return []
            with path.open(encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        saved_rows = read_rows("results.csv")
        saved_components = read_rows("component_results.csv")
        expected = {0, *args.train_paths}
        for replicate in range(args.replicates):
            group = [r for r in saved_rows if int(r["replicate"]) == replicate]
            components = [r for r in saved_components if int(r["replicate"]) == replicate]
            if (len(group) == len(expected)
                    and {int(r["training_paths"]) for r in group} == expected
                    and len(components) == len(expected) * len(METRICS)
                    and {(int(r["training_paths"]), r["metric"]) for r in components}
                    == {(n, metric) for n in expected for metric in METRICS}):
                completed.add(replicate)
                all_rows.extend(group)
                component_rows.extend(components)
        print(f"Resuming: keeping {len(completed)}/{args.replicates} complete groups.", flush=True)
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    np.savez(args.out_dir / "teacher.npz", **{k:np.asarray(v) for k,v in teacher.items()})
    gen_train, flatten, fit, sim, resid = make_functions(args, teacher)
    root_key = jax.random.PRNGKey(args.seed)
    total_comparisons = args.replicates*(1+len(args.train_paths))
    start = time.perf_counter()

    for replicate in range(args.replicates):
        if replicate in completed:
            continue
        rep_key = jax.random.fold_in(root_key, replicate)
        training_keys = jax.random.split(jax.random.fold_in(rep_key, 1), max(args.train_paths)+1)
        training = np.asarray(gen_train(training_keys[:-1]))
        prefix = gen_train(training_keys[-1:])[0, :model.LOOKBACK+1]
        held_paths = make_bank(sim, jax.random.fold_in(rep_key, 2), teacher, ZERO_MEAN, UNIT_SCALE,
                               prefix, args.cdf_paths, args.path_batch)

        for n_train in [0]+args.train_paths:
            stage_start = time.perf_counter()
            if n_train == 0:
                params, mean, scale = teacher, ZERO_MEAN, UNIT_SCALE
                train_info = [float("nan")]*5
            else:
                train = flatten(jnp.asarray(training[:n_train]))
                params, mean, scale, train_info_jax = fit(jax.random.fold_in(rep_key, 3), *train)
                train_info = np.asarray(train_info_jax).tolist()
                print(f"fit {replicate+1}/{args.replicates}, n={n_train}: sigma={float(model.fitted_sigma(params)):.5f}, "
                      f"fit_seconds={time.perf_counter()-stage_start:.1f}", flush=True)
            fitting_seconds = time.perf_counter()-stage_start
            # These streams are independent of the teacher held-out stream.
            q_paths = make_bank(sim, jax.random.fold_in(rep_key, 4), params, mean, scale, prefix,
                               args.cdf_paths, args.path_batch)
            pilot_paths = make_bank(sim, jax.random.fold_in(rep_key, 5), params, mean, scale, prefix,
                                   args.pilot_paths, args.path_batch)
            p_features, p_min = bank_features(resid, params, mean, scale, held_paths, args.path_batch)
            q_features, q_min = bank_features(resid, params, mean, scale, q_paths, args.path_batch)
            pilot_features, pilot_min = bank_features(resid, params, mean, scale, pilot_paths, args.path_batch)
            centers, scales = np.median(pilot_features, axis=0), pilot_features.std(axis=0, ddof=1)
            if not np.isfinite(scales).all() or scales.min() <= 1e-10:
                raise ValueError("Degenerate pilot scales; no fit is discarded.")
            p_scores, q_scores = scores(p_features, centers, scales), scores(q_features, centers, scales)
            row = {"replicate":replicate, "training_paths":n_train,
                   "method":"oracle" if n_train == 0 else f"neural_{args.train_variance}_fit",
                   "fitted_sigma":float(model.fitted_sigma(params)),
                   "fit_loss":train_info[0], "truth_loss_on_training":train_info[1],
                   "known_drift_plugin_sigma":train_info[2], "known_drift_tangent_sigma":train_info[3],
                   "gradient_norm":train_info[4], "fitting_seconds":fitting_seconds,
                   "minimum_tangent_variance":min(p_min, q_min, pilot_min),
                   **compare_score_laws(p_scores, q_scores, args.reference_size, total_comparisons)}
            all_rows.append(row)
            for j, metric in enumerate(METRICS):
                component_rows.append({"replicate":replicate, "training_paths":n_train, "metric":metric,
                    "held_mean":float(p_features[:,j].mean()), "reference_mean":float(q_features[:,j].mean()),
                    "held_median":float(np.median(p_features[:,j])), "reference_median":float(np.median(q_features[:,j])),
                    "pilot_center":float(centers[j]), "pilot_scale":float(scales[j]),
                    "component_cdf_gap":float(ks_2samp(p_features[:,j], q_features[:,j], method="asymp").statistic)})
            stem = f"rep{replicate:02d}_n{n_train}"
            np.savez_compressed(args.out_dir / f"{stem}_scores.npz", held=p_scores, reference=q_scores,
                                held_features=p_features, reference_features=q_features,
                                pilot_features=pilot_features, centers=centers, scales=scales)
            np.savez(args.out_dir / f"{stem}_model.npz", **{k:np.asarray(v) for k,v in params.items()},
                     input_mean=np.asarray(mean), input_scale=np.asarray(scale), prefix=np.asarray(prefix))
            save_rows(args.out_dir / "results.csv", all_rows)
            save_rows(args.out_dir / "component_results.csv", component_rows)
            print(f"done {replicate+1}/{args.replicates}, n={n_train}: gap={row['cdf_gap']:.4f}, "
                  f"conditional_reject={row['conditional_rejection']:.4f}, "
                  f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
    settings["completed_comparisons"] = len(all_rows)
    settings["elapsed_seconds"] = time.perf_counter()-start
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
