"""
Nested full-refit internal-p calibration for the synthetic BI audit.

This runner estimates the size of the deployed fitted-null audit.  Each outer
matched-null replicate independently executes:

1. simulate training, validation-selection, and audit paths;
2. fit candidate neural transition/SDE models;
3. select model width by validation NLL;
4. freeze the selected model;
5. simulate an independent pilot fitted-null bank to standardize S_max;
6. simulate an independent evaluation fitted-null bank to rank the audit path.

The reported p-value for outer replicate r is

    p_r = (1 + #{m: S_ref[r, m] >= S_aud[r]}) / (M + 1),

where S_ref is the evaluation bank standardized using the pilot bank.  The
outer p-values are not ranked against each other.  Rejection rates are computed
directly across independent outer replicates.

Default output location:
  cache/summaries/full_refit_internal_pvalue_calibration_r5000_p500_e999_final/
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


from experiments.audit_statistics import upper_rank
from experiments.bi_synthetic_end_to_end_neural_sde import (
    COMPONENT_METRICS,
    ExperimentConfig,
    METRICS,
    diagnostic_components,
    diagnostics_matrix,
    residuals_for_path,
    select_model,
    simulate_fitted_null_residuals,
    simulate_synthetic_paths,
)

from experiments.full_refit_statistics import COMPONENT_PVALUE_ORDER, DEFAULT_ALPHAS, OUT_DIR, _alpha_table, _component_pvalue_calibration_table, _discrete_uniform_ks_mc_pvalue, _discrete_uniform_ks_stat, _ecdf_table, _summary_table


SEED = 20260710


@dataclass(frozen=True)
class InternalPvalueConfig:
    outer_replicates: int = 5000
    train_paths: int = 32
    selection_paths: int = 8
    steps: int = 96
    pilot_paths: int = 500
    evaluation_paths: int = 999
    epochs: int = 100
    audit_scenario: str = "matched_null"
    ma_rho: float = 0.75
    scale_factor: float = 1.35
    candidate_widths: tuple[int, ...] = (8, 16)


def _inner_config(config: InternalPvalueConfig) -> ExperimentConfig:
    return ExperimentConfig(
        null_trials=config.outer_replicates,
        alt_trials=0,
        audit_paths=1,
        train_paths=config.train_paths,
        selection_paths=config.selection_paths,
        steps=config.steps,
        bootstrap_paths=config.evaluation_paths,
        epochs=config.epochs,
        candidate_widths=config.candidate_widths,
        ma_rho=config.ma_rho,
        scale_factor=config.scale_factor,
    )


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    return str(resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved)


def _write_diagnostic_plot(ecdf: pd.DataFrame, path: Path) -> bool:
    if ecdf.empty:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=160)
    axes[0].step(ecdf["pvalue"], ecdf["ecdf"], where="post", label="empirical ECDF")
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("Internal fitted-null p-value")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].scatter(ecdf["qq_uniform_quantile"], ecdf["pvalue"], s=10, alpha=0.75)
    axes[1].plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Uniform quantile")
    axes[1].set_ylabel("Observed p-value")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _split_bank_audit(
    fit: dict[str, object],
    path: np.ndarray,
    rng: np.random.Generator,
    pilot_paths: int,
    evaluation_paths: int,
) -> dict[str, float]:
    if pilot_paths < 2 or evaluation_paths < 1:
        raise ValueError("Need at least two pilot paths and one evaluation path.")
    obs_residuals = residuals_for_path(fit, path)
    obs_stats = diagnostic_components(obs_residuals)

    pilot_residuals = simulate_fitted_null_residuals(fit, path, pilot_paths, rng)
    eval_residuals = simulate_fitted_null_residuals(fit, path, evaluation_paths, rng)
    pilot_stats = diagnostics_matrix(pilot_residuals)
    eval_stats = diagnostics_matrix(eval_residuals)

    component_cols = list(COMPONENT_METRICS)
    pilot_means = pilot_stats[component_cols].mean(axis=0).to_numpy(dtype=float)
    pilot_stds = pilot_stats[component_cols].std(axis=0, ddof=1).to_numpy(dtype=float)
    pilot_stds = np.where(pilot_stds < 1e-8, 1.0, pilot_stds)

    obs_components = np.asarray([obs_stats[metric] for metric in component_cols], dtype=float)
    eval_components = eval_stats[component_cols].to_numpy(dtype=float)
    pilot_components = pilot_stats[component_cols].to_numpy(dtype=float)
    if not all(np.isfinite(values).all() for values in
               (pilot_components, pilot_means, pilot_stds, obs_components, eval_components)):
        raise FloatingPointError("Non-finite diagnostic statistics; cannot compute calibrated ranks.")
    obs_smax = float(np.max((obs_components - pilot_means) / pilot_stds))
    eval_smax = np.max((eval_components - pilot_means) / pilot_stds, axis=1)
    p_global_smax = upper_rank(obs_smax, eval_smax)

    row: dict[str, float] = {
        "obs_global_smax": obs_smax,
        "p_global_smax_internal": p_global_smax,
        "eval_global_smax_mean": float(np.mean(eval_smax)),
        "eval_global_smax_median": float(np.median(eval_smax)),
        "eval_global_smax_q95": float(np.quantile(eval_smax, 0.95)),
    }
    row.update({f"obs_{key}": float(value) for key, value in obs_stats.items()})
    for metric in component_cols:
        values = eval_stats[metric].to_numpy(dtype=float)
        row[f"p_{metric}_internal"] = upper_rank(obs_stats[metric], values)
        row[f"pilot_{metric}_mean"] = float(pilot_stats[metric].mean())
        row[f"pilot_{metric}_std"] = float(pilot_stats[metric].std(ddof=1))
    return row


def _run_outer_replicate(
    replicate: int,
    seed: int,
    config: InternalPvalueConfig,
    inner: ExperimentConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    train_paths = simulate_synthetic_paths(
        rng,
        config.train_paths,
        config.steps,
        "matched_null",
        config.ma_rho,
        config.scale_factor,
    )
    selection_paths = simulate_synthetic_paths(
        rng,
        config.selection_paths,
        config.steps,
        "matched_null",
        config.ma_rho,
        config.scale_factor,
    )
    audit_paths = simulate_synthetic_paths(
        rng,
        1,
        config.steps,
        config.audit_scenario,
        config.ma_rho,
        config.scale_factor,
    )
    fit, candidates = select_model(train_paths, selection_paths, seed, inner)
    audit = _split_bank_audit(
        fit,
        audit_paths[0],
        rng,
        config.pilot_paths,
        config.evaluation_paths,
    )

    row: dict[str, object] = {
        "scenario": config.audit_scenario,
        "train_scenario": "matched_null",
        "selection_scenario": "matched_null",
        "audit_scenario": config.audit_scenario,
        "replicate": int(replicate),
        "outer_seed": int(seed),
        "selected_width": int(fit["width"]),
        "selected_train_nll": float(fit["train_nll"]),
        "selected_validation_nll": float(fit["validation_nll"]),
        "selected_sigma": float(fit["sigma"]),
        "n_candidate_models": int(len(candidates)),
        "pilot_paths": int(config.pilot_paths),
        "evaluation_paths": int(config.evaluation_paths),
    }
    row.update(audit)

    candidate_rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_rows.append(
            {
                "scenario": config.audit_scenario,
                "train_scenario": "matched_null",
                "selection_scenario": "matched_null",
                "audit_scenario": config.audit_scenario,
                "replicate": int(replicate),
                "outer_seed": int(seed),
                "width": int(candidate["width"]),
                "train_nll": float(candidate["train_nll"]),
                "validation_nll": float(candidate["validation_nll"]),
                "sigma": float(candidate["sigma"]),
                "selected": bool(int(candidate["width"]) == int(fit["width"])),
            }
        )
    return row, candidate_rows


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _output_paths(out_dir: Path, audit_scenario: str) -> dict[str, Path]:
    suffix = "" if audit_scenario == "matched_null" else f"_{_safe_label(audit_scenario)}"
    stem = f"bi_synthetic_full_refit_internal_pvalue{suffix}"
    return {
        "trials": out_dir / f"{stem}_trials.csv",
        "model_selection": out_dir / f"{stem}_model_selection.csv",
        "summary": out_dir / f"{stem}_summary.csv",
        "alpha_table": out_dir / f"{stem}_alpha_table.csv",
        "component_pvalue_calibration": out_dir
        / f"bi_synthetic_full_refit_component_pvalue{suffix}_calibration.csv",
        "ecdf": out_dir / f"{stem}_ecdf.csv",
        "plot": out_dir / f"{stem}_ecdf_qq.png",
        "metadata": out_dir / f"{stem}_metadata.json",
    }


def _append_csv_checkpoint(
    frame: pd.DataFrame,
    path: Path,
    *,
    reset: bool,
) -> None:
    """Append only newly completed rows, or start a fresh checkpoint file."""
    mode = "w" if reset or not path.exists() else "a"
    frame.to_csv(
        path,
        mode=mode,
        header=mode == "w",
        index=False,
        float_format="%.17g",
    )


def _write_checkpoint(
    new_trials: pd.DataFrame,
    new_candidates: pd.DataFrame,
    paths: dict[str, Path],
    args: argparse.Namespace,
    config: InternalPvalueConfig,
    run_metadata: dict[str, object],
    *,
    actual_outer_replicates: int,
    reset: bool,
) -> None:
    """Persist new rows without rebuilding final summaries or rewriting old rows.

    Candidate rows are written first and trial rows last because the trial file
    is the resume completion marker.  If a write is interrupted, rerunning may
    duplicate candidate rows, which ``_load_existing`` removes safely; it will
    not silently skip a completed trial whose candidate rows were never saved.
    """
    if not new_candidates.empty:
        _append_csv_checkpoint(new_candidates, paths["model_selection"], reset=reset)
    if not new_trials.empty:
        _append_csv_checkpoint(new_trials, paths["trials"], reset=reset)

    checkpoint_metadata = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "argv": sys.argv,
        "checkpoint_only": True,
        "derived_outputs_current": False,
        "target_outer_replicates": int(args.outer_replicates),
        "actual_outer_replicates": int(actual_outer_replicates),
        "complete": False,
        "run": run_metadata,
        "config": asdict(config),
    }
    paths["metadata"].write_text(
        json.dumps(checkpoint_metadata, indent=2),
        encoding="utf-8",
    )


def _write_csv_with_fallback(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_csv(path, index=False, float_format="%.17g")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        frame.to_csv(fallback, index=False, float_format="%.17g")
        return fallback


def _write_outputs(
    trials: pd.DataFrame,
    candidates: pd.DataFrame,
    paths: dict[str, Path],
    args: argparse.Namespace,
    config: InternalPvalueConfig,
    run_metadata: dict[str, object],
    write_plot: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the final artifact set and return the two console tables."""
    trials = trials.sort_values("replicate").reset_index(drop=True)
    # Six-decimal rounding moves ranks such as 1/13 off their discrete grid.
    trials.to_csv(paths["trials"], index=False, float_format="%.17g")
    if not candidates.empty:
        candidates = candidates.sort_values(["replicate", "width"]).reset_index(drop=True)

    pvalues = trials["p_global_smax_internal"].to_numpy(dtype=float)
    alphas = tuple(float(alpha) for alpha in args.alphas)
    model_selection_output_path = _write_csv_with_fallback(
        candidates, paths["model_selection"]
    )
    summary = _summary_table(
        pvalues,
        discrete_grid_size=int(config.evaluation_paths) + 1,
        discrete_ks_mc_reps=int(args.ks_mc_reps),
        discrete_ks_mc_seed=int(args.ks_mc_seed),
        scenario=config.audit_scenario,
    )
    alpha_table = _alpha_table(
        pvalues,
        alphas,
        discrete_grid_size=int(config.evaluation_paths) + 1,
    )
    component_pvalue_calibration = _component_pvalue_calibration_table(
        trials,
        grid_size=int(config.evaluation_paths) + 1,
        mc_reps=int(args.ks_mc_reps),
        seed=int(args.ks_mc_seed) + 50_000,
        scenario=config.audit_scenario,
    )
    ecdf = _ecdf_table(pvalues)
    summary_output_path = _write_csv_with_fallback(summary, paths["summary"])
    alpha_output_path = _write_csv_with_fallback(alpha_table, paths["alpha_table"])
    component_output_path = _write_csv_with_fallback(
        component_pvalue_calibration, paths["component_pvalue_calibration"]
    )
    ecdf_output_path = _write_csv_with_fallback(ecdf, paths["ecdf"])
    plot_written = write_plot and _write_diagnostic_plot(ecdf, paths["plot"])

    metadata = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "argv": sys.argv,
        "target_outer_replicates": int(args.outer_replicates),
        "actual_outer_replicates": int(len(trials)),
        "complete": bool(len(trials) >= int(args.outer_replicates)),
        "alphas": alphas,
        "pvalue": "independent internal evaluation-bank upper-rank p-value for pilot-standardized S_max",
        "train_scenario": "matched_null",
        "selection_scenario": "matched_null",
        "audit_scenario": config.audit_scenario,
        "outer_replicates_ranked_against_each_other": False,
        "confidence_interval": "Wilson 95% interval for matched-null rejection rate",
        "ks_calibration": {
            "primary": "discrete uniform rank-grid Monte Carlo",
            "grid_size": int(config.evaluation_paths) + 1,
            "mc_reps": int(args.ks_mc_reps),
            "mc_seed": int(args.ks_mc_seed),
            "continuous_ks_pvalue_retained_as_legacy_diagnostic": True,
        },
        "run": run_metadata,
        "config": asdict(config),
        "inner_experiment_config": asdict(_inner_config(config)),
        "metrics": METRICS,
        "component_metrics_for_global_smax": COMPONENT_METRICS,
        "python": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "outputs": {
            "trials": _display_path(paths["trials"]),
            "model_selection": _display_path(model_selection_output_path),
            "summary": _display_path(summary_output_path),
            "alpha_table": _display_path(alpha_output_path),
            "component_pvalue_calibration": _display_path(
                component_output_path
            ),
            "ecdf": _display_path(ecdf_output_path),
            "ecdf_qq_plot": _display_path(paths["plot"]) if plot_written else None,
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary, alpha_table


def _config_compatibility_fields(config: InternalPvalueConfig) -> dict[str, object]:
    fields = asdict(config)
    fields.pop("outer_replicates", None)
    fields["candidate_widths"] = tuple(fields["candidate_widths"])
    return fields


def _validate_existing_config(
    paths: dict[str, Path],
    trials: pd.DataFrame,
    config: InternalPvalueConfig,
    seed: int,
) -> None:
    if trials.empty:
        return

    if not {"replicate", "outer_seed"}.issubset(trials.columns):
        raise ValueError("Cannot verify resume seed: replicate or outer_seed is missing.")
    base_seeds = (pd.to_numeric(trials["outer_seed"], errors="raise")
                  - pd.to_numeric(trials["replicate"], errors="raise"))
    if not base_seeds.eq(seed).all():
        raise ValueError("Existing trial seeds differ from --seed; use a new --out-dir or --overwrite.")

    expected = _config_compatibility_fields(config)
    if paths["metadata"].exists():
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        saved_seed = metadata.get("run", {}).get("seed")
        if saved_seed is not None and saved_seed != seed:
            raise ValueError("Existing metadata seed differs from --seed; use a new --out-dir or --overwrite.")
        found_config = metadata.get("config", {})
        mismatches = []
        for key, expected_value in expected.items():
            default_value = "matched_null" if key == "audit_scenario" else None
            found_value = found_config.get(key, default_value)
            if key == "candidate_widths" and found_value is not None:
                found_value = tuple(found_value)
            if found_value != expected_value:
                mismatches.append((key, found_value, expected_value))
        if mismatches:
            details = "; ".join(
                f"{key}: existing={found!r}, requested={want!r}"
                for key, found, want in mismatches
            )
            raise ValueError(
                "Existing checkpoint metadata are incompatible with the requested "
                f"configuration in {_display_path(paths['metadata'])}: {details}. "
                "Use a new --out-dir or --overwrite."
            )
        return

    column_checks = {
        "pilot_paths": int(config.pilot_paths),
        "evaluation_paths": int(config.evaluation_paths),
    }
    mismatches = []
    for column, expected_value in column_checks.items():
        if column not in trials.columns:
            mismatches.append((column, "missing", expected_value))
            continue
        found_values = sorted({int(value) for value in trials[column].dropna().astype(int)})
        if found_values != [expected_value]:
            mismatches.append((column, found_values, expected_value))
    if mismatches:
        details = "; ".join(
            f"{key}: existing={found!r}, requested={want!r}"
            for key, found, want in mismatches
        )
        raise ValueError(
            "Existing checkpoint rows are incompatible with the requested "
            f"configuration in {_display_path(paths['trials'])}: {details}. "
            "Use a new --out-dir or --overwrite."
        )
    if "audit_scenario" in trials.columns:
        found_scenarios = sorted(
            {str(value) for value in trials["audit_scenario"].dropna().astype(str)}
        )
        if found_scenarios and found_scenarios != [config.audit_scenario]:
            raise ValueError(
                "Existing checkpoint rows have incompatible audit scenarios in "
                f"{_display_path(paths['trials'])}: existing={found_scenarios!r}, "
                f"requested={config.audit_scenario!r}. Use a new --out-dir or --overwrite."
            )
    elif "scenario" in trials.columns:
        found_scenarios = sorted({str(value) for value in trials["scenario"].dropna().astype(str)})
        if found_scenarios and found_scenarios != [config.audit_scenario]:
            raise ValueError(
                "Existing checkpoint rows have incompatible scenarios in "
                f"{_display_path(paths['trials'])}: existing={found_scenarios!r}, "
                f"requested={config.audit_scenario!r}. Use a new --out-dir or --overwrite."
            )


def _load_existing(
    paths: dict[str, Path],
    overwrite: bool,
    config: InternalPvalueConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if overwrite:
        return pd.DataFrame(), pd.DataFrame()
    if paths["trials"].exists():
        trials = pd.read_csv(paths["trials"])
        if "replicate" in trials.columns:
            trials = trials.sort_values("replicate").drop_duplicates("replicate", keep="last")
    else:
        trials = pd.DataFrame()
    if paths["model_selection"].exists():
        candidates = pd.read_csv(paths["model_selection"])
        if {"replicate", "width"}.issubset(candidates.columns):
            candidates = candidates.sort_values(["replicate", "width"]).drop_duplicates(
                ["replicate", "width"], keep="last"
            )
    else:
        candidates = pd.DataFrame()
    _validate_existing_config(paths, trials, config, seed)
    return trials, candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-replicates", type=int, default=InternalPvalueConfig.outer_replicates)
    parser.add_argument("--train-paths", type=int, default=InternalPvalueConfig.train_paths)
    parser.add_argument("--selection-paths", type=int, default=InternalPvalueConfig.selection_paths)
    parser.add_argument("--steps", type=int, default=InternalPvalueConfig.steps)
    parser.add_argument("--pilot-paths", type=int, default=InternalPvalueConfig.pilot_paths)
    parser.add_argument("--evaluation-paths", type=int, default=InternalPvalueConfig.evaluation_paths)
    parser.add_argument("--epochs", type=int, default=InternalPvalueConfig.epochs)
    parser.add_argument(
        "--audit-scenario",
        choices=("matched_null", "ma_innovation", "scale_innovation"),
        default=InternalPvalueConfig.audit_scenario,
        help=(
            "Scenario for the single held-out audit path. Training and validation-selection "
            "paths remain matched_null."
        ),
    )
    parser.add_argument("--ma-rho", type=float, default=InternalPvalueConfig.ma_rho)
    parser.add_argument("--scale-factor", type=float, default=InternalPvalueConfig.scale_factor)
    parser.add_argument(
        "--candidate-widths",
        nargs="+",
        type=int,
        default=list(InternalPvalueConfig.candidate_widths),
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-new-replicates",
        type=int,
        default=None,
        help="When resuming, run at most this many new outer replicates.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Append newly completed rows every N replicates without recomputing final "
            "summaries (default: 1)."
        ),
    )
    parser.add_argument(
        "--ks-mc-reps",
        type=int,
        default=100_000,
        help="Monte Carlo repetitions for the discrete rank-grid KS calibration.",
    )
    parser.add_argument("--ks-mc-seed", type=int, default=20260711)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = InternalPvalueConfig(
        outer_replicates=args.outer_replicates,
        train_paths=args.train_paths,
        selection_paths=args.selection_paths,
        steps=args.steps,
        pilot_paths=args.pilot_paths,
        evaluation_paths=args.evaluation_paths,
        epochs=args.epochs,
        audit_scenario=args.audit_scenario,
        ma_rho=args.ma_rho,
        scale_factor=args.scale_factor,
        candidate_widths=tuple(args.candidate_widths),
    )
    paths = _output_paths(args.out_dir, config.audit_scenario)
    inner = _inner_config(config)

    trials, candidates = _load_existing(paths, args.overwrite, config, args.seed)
    completed = set()
    if not trials.empty and "replicate" in trials.columns:
        completed = {int(value) for value in trials["replicate"].astype(int).tolist()}
    missing = [idx for idx in range(int(args.outer_replicates)) if idx not in completed]
    if args.max_new_replicates is not None:
        missing = missing[: int(args.max_new_replicates)]

    run_metadata = {
        "source": "fresh_or_resumed_run",
        "seed": int(args.seed),
        "existing_replicates_before_run": int(len(completed)),
        "new_replicates_requested_this_run": int(len(missing)),
    }

    new_rows: list[dict[str, object]] = []
    new_candidates: list[dict[str, object]] = []
    reset_checkpoint_files = bool(args.overwrite)
    for offset, replicate in enumerate(missing, start=1):
        outer_seed = int(args.seed + replicate)
        print(
            f"Nested full-refit internal-p {config.audit_scenario} audit replicate "
            f"{offset}/{len(missing)} (replicate={replicate}, seed={outer_seed})",
            flush=True,
        )
        row, candidate_rows = _run_outer_replicate(replicate, outer_seed, config, inner)
        new_rows.append(row)
        new_candidates.extend(candidate_rows)

        if args.checkpoint_every > 0 and offset % int(args.checkpoint_every) == 0:
            checkpoint_trials = pd.DataFrame(new_rows)
            checkpoint_candidates = pd.DataFrame(new_candidates)
            run_metadata["new_replicates_completed_this_run"] = int(offset)
            _write_checkpoint(
                checkpoint_trials,
                checkpoint_candidates,
                paths,
                args,
                config,
                run_metadata,
                actual_outer_replicates=int(len(trials) + len(checkpoint_trials)),
                reset=reset_checkpoint_files,
            )
            reset_checkpoint_files = False
            trials = pd.concat([trials, checkpoint_trials], ignore_index=True)
            candidates = pd.concat(
                [candidates, checkpoint_candidates], ignore_index=True
            )
            new_rows.clear()
            new_candidates.clear()

    if new_rows:
        trials = pd.concat([trials, pd.DataFrame(new_rows)], ignore_index=True)
    if new_candidates:
        candidates = pd.concat([candidates, pd.DataFrame(new_candidates)], ignore_index=True)
    run_metadata["new_replicates_completed_this_run"] = int(len(missing))
    summary, alpha_table = _write_outputs(
        trials,
        candidates,
        paths,
        args,
        config,
        run_metadata,
        write_plot=not args.no_plot,
    )

    print("\nNested full-refit internal-p calibration summary")
    print(
        summary.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nRejection-rate table")
    print(
        alpha_table.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    for key in (
        "trials",
        "model_selection",
        "summary",
        "alpha_table",
        "component_pvalue_calibration",
        "ecdf",
        "plot",
        "metadata",
    ):
        if key == "plot" and args.no_plot:
            continue
        print(f"Saved {_display_path(paths[key])}")


if __name__ == "__main__":
    main()
