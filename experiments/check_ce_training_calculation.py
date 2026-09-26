"""Check the faster calculation against the existing CE pilot training loss."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
from experiments import run_ce_conditional_fit_pilot as pilot
import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/neural_training_comparison")
    folder = parser.parse_args().out_dir
    folder.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / "output/ce_conditional_fit_pilot"
    settings = json.loads((baseline / "settings.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**settings)
    args.epochs, args.train_variance = 100, "tangent"
    teacher = pilot.teacher_parameters()
    gen, flatten, _, _, _ = pilot.make_functions(args, teacher)
    rep_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 0)
    keys = jax.random.split(jax.random.fold_in(rep_key, 1), 193)
    paths = gen(keys[:-1])
    training = flatten(paths)
    mean, scale = pilot.model.standardize_inputs(*training[:2])
    saved = np.load(baseline / "rep00_n192_model.npz")
    np.testing.assert_array_equal(np.asarray(mean), saved["input_mean"])
    np.testing.assert_array_equal(np.asarray(scale), saved["input_scale"])
    prefix = gen(keys[-1:])[0, :pilot.model.LOOKBACK+1]
    np.testing.assert_array_equal(np.asarray(prefix), saved["prefix"])
    initial = pilot.model.init_params(jax.random.fold_in(rep_key, 3))
    parameters = [initial, pilot.teacher_in_fitted_inputs(teacher, mean, scale),
                  {k:jnp.asarray(saved[k]) for k in initial}]
    # Spread the checks across actual state/control pairs from the fixed data.
    small = tuple(a[jnp.arange(0, len(a), 157)] for a in training)
    checks = []
    for variance in ("plugin", "tangent"):
        reference = jax.jit(jax.value_and_grad(lambda p: pilot.model.transition_loss(
            p, pilot.MASK, *small, mean, scale, args.substeps, 1e-4, variance)))
        batched = jax.jit(jax.value_and_grad(lambda p: pilot.batched_transition_loss(
            p, *small, mean, scale, args.substeps, 1e-4, variance)))
        for i, prm in enumerate(parameters):
            val0, grad0 = reference(prm)
            val1, grad1 = batched(prm)
            errors = {k:float(jnp.max(jnp.abs(grad0[k]-grad1[k]))) for k in grad0}
            row = {"variance":variance, "parameters":i, "loss_error":float(abs(val0-val1)),
                   "gradient_max_abs_errors":errors}
            if row["loss_error"] > 1e-10 or max(errors.values()) > 5e-6:
                raise AssertionError(row)
            checks.append(row)
    timings, trained = {}, {}
    for name in ("standard", "batched"):
        args.training_implementation = name
        _, _, fit, _, _ = pilot.make_functions(args, teacher)
        start = time.perf_counter()
        out = fit(jax.random.fold_in(rep_key, 3), *training)
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), out)
        timings[name] = time.perf_counter()-start
        trained[name] = out[0]
        print(f"{name}: {timings[name]:.2f}s", flush=True)
    differences = {k:float(jnp.max(jnp.abs(trained["standard"][k]-trained["batched"][k])))
                   for k in trained["standard"]}
    if max(differences.values()) > 2e-5:
        raise AssertionError(differences)
    result = {"normalization_matches_saved_baseline_exactly":True,
              "prefix_matches_saved_baseline_exactly":True, "loss_gradient_checks":checks,
              "updates":args.epochs, "training_paths":192,
              "parameter_difference_after_updates":differences, "timing_seconds":timings}
    (folder / "training_calculation_checks.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
