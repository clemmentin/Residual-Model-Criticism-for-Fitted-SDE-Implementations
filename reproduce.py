"""Reproduce the current manuscript tables, figures and SIR fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def paper_sir_config(out_dir: Path):
    """Reuse the paper runner's settings; write new fits outside the archive."""
    from experiments.run_sir_sde_native_country_audit import _config

    config = _config("GBR")
    # GBR is in the development cohort. This diagnostic validation pass does
    # not select a checkpoint; all 60 epochs are used.
    config.MODEL_SAVE_PATH = out_dir / "sir_fit" / "model.eqx"
    config.SNAPSHOT_DIR = out_dir / "sir_fit" / "snapshots"
    return config


def paper_tables(task: str, out_dir: Path) -> None:
    """Aggregate retained per-fit estimates using the manuscript definitions."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    if task in ("table1", "tables"):
        source = ROOT / "experiments/nonlinear_weight_selection/coupling/results.csv"
        trials = pd.read_csv(source)
        groups = trials.loc[trials.kappa.eq(0.1)].groupby(["training_paths", "method"], sort=False)
        table = groups.agg(fits=("fit", "size"), largest_gap=("gap", "max"),
                           rejection_min=("null_rejection_cv", "min"),
                           rejection_max=("null_rejection_cv", "max"),
                           power_y=("weak_matched", "mean"))
        if not table.fits.eq(64).all() or len(table) != 12:
            raise ValueError("Expected four weighting methods for each of three sizes, 64 fits each.")
        table.to_csv(out_dir / "table1_nonlinear_weights.csv")
    if task in ("table2", "tables"):
        source = ROOT / "output/neural_training_comparison/combined_results.csv"
        trials = pd.read_csv(source)
        table = trials.groupby(["training_paths", "train_variance"], sort=False).agg(
            analyses=("replicate", "size"), gap_median=("cdf_gap", "median"),
            gap_min=("cdf_gap", "min"), gap_max=("cdf_gap", "max"),
            mean_conditional_rejection=("conditional_rejection", "mean"))
        if not table.analyses.eq(12).all() or len(table) != 5:
            raise ValueError("Expected one oracle and four fitted conditions, 12 analyses each.")
        table.to_csv(out_dir / "table2_neural_matching.csv")
    print(f"Tables written to {out_dir}", flush=True)


def paper_figures(out_dir: Path) -> None:
    """Build the manuscript figures and retained historical comparisons."""
    from experiments import make_paper1_core_figures as core
    from experiments import make_manuscript_figures as styled

    residuals = ROOT / "output/sir_figure_data/residuals"
    if any(not (residuals / f"{country}_residual_paths.npz").is_file()
           for country in ("CZE", "GRC")):
        from experiments.replay_fixed_calendar_residual_paths import main as replay
        replay()
    out_dir.mkdir(parents=True, exist_ok=True)
    core.FIGURE_DIR = out_dir
    core.configure_style()
    core.make_reference_law_figure()
    core.make_nordic_audit_figure()
    core.make_random_country_replication_figure()
    core.make_nordic_window_sensitivity_figure()
    styled.main(["--out-dir", str(out_dir)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", nargs="?", default="table1",
        choices=("table1", "table2", "tables", "full-refit", "figures", "sir-settings", "sir-fit"),
        help="Default: rebuild the supplementary nonlinear weight comparison from retained results.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    started = time.perf_counter()

    if args.task in ("table1", "table2", "tables"):
        paper_tables(args.task, out_dir / "tables")
    elif args.task == "full-refit":
        from experiments.full_refit_statistics import OUT_DIR
        from experiments.summarize_full_refit_calibration import summarize

        for name in (
            "bi_synthetic_full_refit_internal_pvalue_ecdf.csv",
            "bi_synthetic_full_refit_component_pvalue_calibration.csv",
        ):
            if not (OUT_DIR / name).is_file():
                parser.error(
                    f"Missing {OUT_DIR / name}. Unpack the matching artifact bundle's "
                    "cache/ and models/ folders beside reproduce.py; see README.md."
                )
        summarize(OUT_DIR, out_dir / "tables", mc_reps=100_000, seed=20260711)
    elif args.task == "figures":
        paper_figures(out_dir / "figures")
    else:
        config = paper_sir_config(out_dir)
        print(json.dumps(vars(config), indent=2, default=str), flush=True)
        if args.task == "sir-fit":
            import logging
            import jax
            from config import get_model_save_path
            from sir_data import load_or_create_training_data
            from sir_training import train_ensemble, validate_training_config

            if not config.OWID_CACHE_PATH.is_file():
                parser.error("Missing cache/owid_covid_data.csv; unpack the matching bundle first.")
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
            validate_training_config(config)
            data = load_or_create_training_data(config)
            validate_training_config(config, n_train=data.train_set.ys.shape[0])
            train_ensemble(config, data, jax.random.PRNGKey(config.SEED))
            print(f"Fitted model: {get_model_save_path(config)}", flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
