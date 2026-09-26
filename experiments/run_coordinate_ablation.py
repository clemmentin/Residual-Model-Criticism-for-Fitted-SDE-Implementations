"""A/B/C coordinate-construction ablation: global vs tangent, matched vs
theoretical calibration.

Three residual-coordinate methods, sharing one fitted model and one reference
bank wherever the design calls for it:

    A  global whitening       (pilot-estimated fixed covariance)  + matched MC
    B  tangent whitening      (per-step propagated covariance)    + theoretical chi-square
    C  tangent whitening      (per-step propagated covariance)    + matched MC   [paper's method]

Every outer replicate: simulate train/selection/pilot/reference paths from the
truth generator; fit one model; freeze it as Q; then, from that *same* frozen
Q, generate three held-out paths -- the matched null, a targeted deviation
(extra noise loaded onto the unforced coordinate only), and a uniform
deviation (the whole existing loading scaled by one constant). A and C rank
each held-out path's energy statistic against the shared reference bank; B
looks up the same tangent-coordinate energy statistic in its working
chi-square reference instead.

``--kappa`` sets the propagation strength (conditions 1/2); ``--nonlinear-kappa``
switches the truth generator's propagating coupling from linear to tanh
(condition 3), so the tangent recursion's linearization is no longer exact.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functools import partial

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import jax

# Float32 matches the retained runs. Enabling x64 also changes JAX random
# draws, so it defines a different realization even with the same seed.
_ENABLE_X64 = False
if _ENABLE_X64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


from experiments import bi_synthetic_solver_calibration as model, coordinate_ablation_model as extra
from experiments.audit_statistics import wilson_interval
_wilson = partial(wilson_interval, clip=False)


DEFAULT_OUT = Path("cache/summaries/coordinate_ablation")
SCENARIOS = ("null", "targeted", "uniform")
METHODS = ("A_global_mc", "B_tangent_theory", "C_tangent_mc")

def _scenario_block(scenario: str) -> tuple[str, ...]:
    # Must mirror the per-scenario append order in `one_trial`:
    # [p_A(3), p_B(3), p_C(3), energy_global(3), energy_tangent(3)].
    p_fields = tuple(
        f"p_{scenario}_{method}_{coord}"
        for method in METHODS
        for coord in ("full", "c1", "c2")
    )
    energy_fields = tuple(
        f"energy_{scenario}_{method}_{coord}"
        for method in ("global", "tangent")
        for coord in ("full", "c1", "c2")
    )
    return p_fields + energy_fields


FIELDS = ("replicate", "valid", "fitted_sigma", "selected_width") + tuple(
    field for scenario in SCENARIOS for field in _scenario_block(scenario)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    # Batch size controls memory use; see REPRODUCE.md for the archived setting.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-paths", type=int, default=48)
    parser.add_argument("--selection-paths", type=int, default=12,
                         help="Unused while --fixed-width is set (no candidate selection); "
                              "kept for metadata parity with run_solver_scale_alignment_calibration.py.")
    # These bank sizes match the retained A/B/C runs.
    parser.add_argument("--pilot-paths", type=int, default=100)
    parser.add_argument("--steps", type=int, default=112)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--nonlinear-kappa", action="store_true")
    parser.add_argument("--epochs", type=int, default=3_000)
    parser.add_argument("--evaluation-paths", type=int, default=200)
    parser.add_argument("--fixed-width", type=int, choices=(8, 16), default=8)
    parser.add_argument("--extra-sigma1", type=float, default=0.12,
                         help="Deviation 1: extra independent noise sd loaded onto coordinate 1.")
    parser.add_argument("--scale-factor", type=float, default=1.35,
                         help="Deviation 2: multiplier on the existing (coordinate-2) loading.")
    parser.add_argument("--condition-tag", type=str, required=True,
                         help="Free-text label for this (kappa, coupling) condition, e.g. 'cond1_weak'.")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "design": "coordinate_ablation_ABC_v1",
        "condition_tag": args.condition_tag,
        "seed": args.seed,
        "replicates": args.replicates,
        "batch_size": args.batch_size,
        "train_paths": args.train_paths,
        "selection_paths": args.selection_paths,
        "pilot_paths": args.pilot_paths,
        "steps": args.steps,
        "substeps": args.substeps,
        "kappa": args.kappa,
        "nonlinear_kappa": args.nonlinear_kappa,
        "epochs": args.epochs,
        "evaluation_paths": args.evaluation_paths,
        "fixed_width": args.fixed_width,
        "extra_sigma1": args.extra_sigma1,
        "scale_factor": args.scale_factor,
        "true_sigma": model.SIGMA_TRUE,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(x) for x in jax.devices()],
        "python": platform.python_version(),
    }


def _validate_resume(path: Path, expected: dict[str, object], overwrite: bool) -> None:
    """Check that an existing run used compatible settings.

    Does not decide the resume point -- see `_csv_row_count`.  The two files
    are written non-atomically (CSV flush, then metadata), so an interrupt in
    between leaves metadata behind the CSV; treating metadata as authoritative
    would silently recompute and duplicate rows.
    """
    if overwrite or not path.exists():
        return
    found = json.loads(path.read_text(encoding="utf-8"))
    ignored = {"replicates", "batch_size", "completed_replicates", "elapsed_seconds",
               "seconds_per_new_replicate"}
    mismatches = {
        key: (found.get(key), value)
        for key, value in expected.items()
        if key not in ignored and found.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Existing run has incompatible metadata: {mismatches}")


def _csv_row_count(csv_path: Path, overwrite: bool) -> int:
    """Number of complete data rows in the trials CSV -- the resume point.

    The CSV is authoritative because it is flushed before metadata is written.
    Validate its exact header before recovering rows: a legacy schema (e.g.
    without ``valid``) must not be mistaken for a file of partial writes.
    A trailing row with the wrong field count (a partial write killed
    mid-line) is discarded so the run resumes from the last intact record.
    """
    if overwrite or not csv_path.exists():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return 0
        if tuple(header) != FIELDS:
            missing = [field for field in FIELDS if field not in header]
            unexpected = [field for field in header if field not in FIELDS]
            raise ValueError(
                f"Existing trials CSV has incompatible schema: {csv_path}. "
                f"Expected {len(FIELDS)} columns with exact names and order, "
                f"found {len(header)} (missing={missing}, unexpected={unexpected}). "
                "Refusing to resume or overwrite existing data. "
                "Use a new --out-dir or --condition-tag for this code version."
            )
        data_rows = list(reader)
    while data_rows and len(data_rows[-1]) != len(FIELDS):
        data_rows.pop()
    return len(data_rows)


def _truncate_csv(csv_path: Path, keep_rows: int) -> None:
    """Rewrite the CSV keeping only `keep_rows` complete data rows."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, data_rows = rows[0], rows[1:]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(data_rows[:keep_rows])


def make_batched_trial(args: argparse.Namespace):
    n_train = args.train_paths
    n_pilot = args.pilot_paths
    n_steps = args.steps
    n_substeps = args.substeps
    kappa = args.kappa
    nonlinear_kappa = args.nonlinear_kappa
    epochs = args.epochs
    n_eval = args.evaluation_paths
    fixed_width = args.fixed_width
    extra_sigma1 = args.extra_sigma1
    scale_factor = args.scale_factor
    horizon = n_steps - model.LOOKBACK

    def simulate_many(key, count):
        keys = jax.random.split(key, count)
        return jax.vmap(
            lambda k: extra.simulate_truth_path_general(
                k, n_steps, kappa, n_substeps, nonlinear_kappa
            )
        )(keys)

    def flatten(paths):
        states, controls, increments = jax.vmap(model.path_transitions)(paths)
        return tuple(x.reshape((-1, x.shape[-1])) for x in (states, controls, increments))

    def one_trial(key):
        # No k_select: width is fixed (--fixed-width), so unlike the base
        # module's own runner this never trains a selection candidate or
        # scores a selection bank -- there is nothing to select between.
        (k_train, k_pilot, k_reference, k_prefix,
         k_null, k_targeted, k_uniform) = jax.random.split(key, 7)

        train = flatten(simulate_many(k_train, n_train))
        input_mean, input_scale = model.standardize_inputs(train[0], train[1])

        mask = model.width_mask(fixed_width)
        # Trains with the unpropagated ("plugin") accumulator, matching the
        # audited SIR training loss (33), which is deliberately distinct from
        # the post-fit propagated tangent covariance (21).  The tangent
        # covariance methods B and C need is computed only at audit time, in
        # `path_residuals` below -- never inside this training loop -- so
        # switching this string does not change what B/C measure, only how
        # cheaply the model is fit.  Training with "tangent" here would nest
        # jax.grad (reverse mode) over jax.jacobian (also reverse mode by
        # default) once per training row per epoch: a reverse-over-reverse
        # composition that is disproportionately expensive to trace, compile,
        # and run, and is not how the audited implementation actually trains.
        params = model.train_candidate(
            k_train, mask, *train, input_mean, input_scale,
            n_substeps, epochs, 3e-3, 1e-4, "plugin",
        )

        prefix_path = extra.simulate_truth_path_general(
            k_prefix, n_steps, kappa, n_substeps, nonlinear_kappa
        )
        prefix = prefix_path[: model.LOOKBACK + 1]

        pilot_paths = model.simulate_reference_paths(
            k_pilot, params, mask, input_mean, input_scale,
            prefix, n_pilot, n_steps, n_substeps,
        )
        global_cov = extra.global_pilot_covariance(
            params, mask, input_mean, input_scale, pilot_paths, n_substeps
        )

        reference_paths = model.simulate_reference_paths(
            k_reference, params, mask, input_mean, input_scale,
            prefix, n_eval, n_steps, n_substeps,
        )

        def tangent_and_global_energy(path):
            resid, cov = model.path_residuals(
                params, mask, input_mean, input_scale, path, n_substeps
            )
            z_tan = extra.cholesky_whiten(resid, cov)
            z_glob = extra.cholesky_whiten(resid, jnp.broadcast_to(global_cov, cov.shape))
            return extra.energy_from_z(z_tan), extra.energy_from_z(z_glob)

        ref_tan_energy, ref_glob_energy = jax.vmap(tangent_and_global_energy)(reference_paths)

        held_out_keys = {"null": k_null, "targeted": k_targeted, "uniform": k_uniform}
        loading_fns = {
            "null": extra.null_loading,
            "targeted": extra.targeted_loading(extra_sigma1),
            "uniform": extra.uniform_scale_loading(scale_factor),
        }

        # Validity must be judged on the INTERMEDIATES, not just the exported
        # vector.  A NaN anywhere in the reference bank makes every
        # `reference >= observed` comparison False, so the exceedance count is
        # zero and the rank p-value comes out finite -- at its floor,
        # 1/(n_eval+1), i.e. maximally "significant".  Checking only the
        # exported p-values therefore cannot detect a poisoned reference bank;
        # this flag can.
        param_finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(params)])
        )
        valid = (
            param_finite
            & jnp.all(jnp.isfinite(global_cov))
            & jnp.all(jnp.isfinite(ref_tan_energy))
            & jnp.all(jnp.isfinite(ref_glob_energy))
            # A degenerate reference bank (zero spread) makes the rank
            # comparison meaningless even when every entry is finite.
            & (jnp.std(ref_tan_energy[:, 0]) > 0.0)
            & (jnp.std(ref_glob_energy[:, 0]) > 0.0)
        )

        out = []
        for scenario in SCENARIOS:
            held = extra.simulate_fitted_path_with_deviation(
                held_out_keys[scenario], params, mask, input_mean, input_scale,
                prefix, n_steps, n_substeps, loading_fns[scenario],
            )
            tan_energy, glob_energy = tangent_and_global_energy(held)
            valid = valid & jnp.all(jnp.isfinite(tan_energy)) & jnp.all(jnp.isfinite(glob_energy))

            p_global = (1.0 + jnp.sum(ref_glob_energy >= glob_energy, axis=0)) / (n_eval + 1.0)
            p_tangent_mc = (1.0 + jnp.sum(ref_tan_energy >= tan_energy, axis=0)) / (n_eval + 1.0)
            p_tangent_theory = extra.theoretical_pvalue(tan_energy, horizon)

            out.extend([p_global, p_tangent_theory, p_tangent_mc])
            out.append(glob_energy)
            out.append(tan_energy)

        flat = jnp.concatenate([jnp.atleast_1d(x) for x in out])
        return jnp.concatenate(
            (jnp.asarray([valid.astype(jnp.float32),
                          model.fitted_sigma(params), float(fixed_width)]), flat)
        )

    return jax.jit(jax.vmap(one_trial))


MIN_HORIZON = 8  # below this a mean-squared energy statistic is too short to rank


def _validate_args(args: argparse.Namespace) -> None:
    """Reject configurations that would silently produce degenerate p-values.

    The failure this guards against is not loud: with ``horizon <= 0`` the
    energy statistics are means over an empty axis, hence NaN.  Every
    ``reference >= observed`` comparison against NaN is False, the exceedance
    count is zero, and the rank p-value collapses to its floor
    ``1 / (n_eval + 1)`` -- i.e. a broken run reports the *most* significant
    value the design can produce.  Fail here instead.
    """
    if args.replicates <= 0 or args.batch_size <= 0:
        raise ValueError("replicates and batch-size must be positive")
    horizon = args.steps - model.LOOKBACK
    if horizon < MIN_HORIZON:
        raise ValueError(
            f"--steps {args.steps} gives horizon {horizon} "
            f"(= steps - LOOKBACK({model.LOOKBACK})); need at least {MIN_HORIZON}. "
            "Shorter horizons make the energy statistics degenerate or NaN, "
            "which silently collapses every rank p-value to its floor."
        )
    if args.substeps < 1:
        raise ValueError("--substeps must be >= 1")
    if args.pilot_paths < 2 or args.evaluation_paths < 2:
        raise ValueError("--pilot-paths and --evaluation-paths must both be >= 2")
    if args.train_paths < 1:
        raise ValueError("--train-paths must be >= 1")
    if args.extra_sigma1 < 0.0:
        raise ValueError("--extra-sigma1 must be non-negative")
    if args.scale_factor <= 0.0:
        raise ValueError("--scale-factor must be positive")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    # Never compute more padded trials than requested: the batched shape is
    # args.batch_size regardless of how many rows are kept, so a small run
    # with a large batch silently pays for the whole batch.
    args.batch_size = min(args.batch_size, args.replicates)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"coord_ablation_{args.condition_tag}"
    csv_path = args.out_dir / f"{stem}_trials.csv"
    metadata_path = args.out_dir / f"{stem}_metadata.json"
    summary_path = args.out_dir / f"{stem}_summary.json"
    metadata = _metadata(args)
    _validate_resume(metadata_path, metadata, args.overwrite)
    completed = _csv_row_count(csv_path, args.overwrite)
    if completed and csv_path.exists():
        # Drop any partial trailing row before appending, so the resumed file
        # cannot contain a half-written record between two complete ones.
        _truncate_csv(csv_path, completed)
    if completed >= args.replicates:
        print(f"{args.condition_tag}: already have {completed}/{args.replicates} replicates; nothing to do.")
        return
    mode = "w" if completed == 0 else "a"
    completed_at_start = completed
    batched_trial = make_batched_trial(args)
    start = time.perf_counter()
    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        while completed < args.replicates:
            count = min(args.batch_size, args.replicates - completed)
            ids = jnp.arange(completed, completed + args.batch_size, dtype=jnp.uint32)
            base_key = jax.random.PRNGKey(args.seed)
            keys = jax.vmap(lambda idx: jax.random.fold_in(base_key, idx))(ids)
            values = np.asarray(batched_trial(keys), dtype=np.float64)[:count]
            # A NaN here (diverged fit, degenerate covariance) would not raise:
            # it propagates into the rank comparisons as False and reports the
            # p-value floor, so a broken batch would look maximally significant.
            if not np.all(np.isfinite(values)):
                bad = int(np.argmax(~np.all(np.isfinite(values), axis=1)))
                raise FloatingPointError(
                    f"Non-finite value in replicate batch starting at {completed} "
                    f"(first offending row offset {bad}). Refusing to write: "
                    "NaN silently collapses rank p-values to their floor."
                )
            # The in-graph validity flag catches what the finiteness check
            # above structurally cannot: a poisoned or degenerate reference
            # bank still yields finite p-values (at the floor).
            valid_col = values[:, FIELDS.index("valid") - 1]
            if not np.all(valid_col == 1.0):
                bad_rows = np.nonzero(valid_col != 1.0)[0]
                raise FloatingPointError(
                    f"{len(bad_rows)} replicate(s) in batch starting at {completed} "
                    f"failed the in-graph validity check (first offset {int(bad_rows[0])}): "
                    "non-finite fit params / pilot covariance / reference or held-out "
                    "energies, or a zero-spread reference bank. Refusing to write."
                )
            for offset, row in enumerate(values):
                record = {"replicate": completed + offset}
                for name, value in zip(FIELDS[1:], row, strict=True):
                    record[name] = float(value)
                writer.writerow(record)
            handle.flush()
            completed += count
            elapsed = time.perf_counter() - start
            checkpoint = dict(metadata)
            checkpoint.update(
                completed_replicates=completed,
                elapsed_seconds=elapsed,
                seconds_per_new_replicate=elapsed / max(1, completed - completed_at_start),
            )
            metadata_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
            print(
                f"{args.condition_tag}: {completed}/{args.replicates} "
                f"({elapsed / max(1, completed - completed_at_start):.3f} s/new trial)",
                flush=True,
            )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary: dict[str, object] = {"condition_tag": args.condition_tag, "n": len(rows)}
    for scenario in SCENARIOS:
        for method in METHODS:
            key = f"p_{scenario}_{method}_full"
            pvalues = np.asarray([float(row[key]) for row in rows])
            entry: dict[str, object] = {"pvalue_mean": float(np.mean(pvalues))}
            for alpha in (0.01, 0.025, 0.05, 0.10):
                rejects = int(np.sum(pvalues <= alpha))
                low, high = _wilson(rejects, len(pvalues))
                entry[f"alpha_{alpha:g}"] = {
                    "rejects": rejects, "rate": rejects / len(pvalues), "wilson95": [low, high],
                }
            summary[key] = entry
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
