"""Build descriptive six-country residual and cumulative-energy figures.

The countries are the two randomized new-country batches already reported in
the paper.  Reference residual paths are regenerated with the exact recorded
5,000-path seeds; it does not define a new test or new evidence.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


from experiments import run_sir_sde_native_country_audit as native


COUNTRIES = ("BGR", "HUN", "SVK", "LTU", "MDA", "SVN")
BATCH1 = set(COUNTRIES[:3])
N_BOOTSTRAP = 5000
CACHE_DIR = ROOT / "cache" / "summaries" / "six_country_path_visualization"
FIGURE_DIR = ROOT / "output" / "figures"
BATCH1_DIR = ROOT / "cache" / "summaries" / "random_country_5000_reference_draws"
BATCH2_DIR = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_random_country_replication"
    / "random_country_batch2_frozen"
    / "evaluation"
)

BLUE = "#31688e"
RED = "#bd2732"
GREY = "#555555"
LIGHT_GREY = "#d7d7d7"
REFERENCE_BLUE = "#4c78a8"


def _root_for(country: str) -> Path:
    return BATCH1_DIR if country in BATCH1 else BATCH2_DIR


def _seed_for(country: str) -> int:
    if country in BATCH1:
        return native._country_seed(
            20260813, country, "exploratory-5000-fitted-null"
        )
    return native._country_seed(20260814, country, "fitted-null")


def _build_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=False)
    for country in COUNTRIES:
        print(f"=== path visualization reproduction: {country} ===", flush=True)
        cfg = native._config(country)
        native.bi._validate_supported_config(cfg)
        data = native.load_cached_data(cfg)
        model = native._load_common_original9_model(cfg, data)
        seed = _seed_for(country)
        result = native._evaluate_country_paths(
            model,
            data,
            cfg,
            n_bootstrap=N_BOOTSTRAP,
            seed=seed,
            retain_null_z=True,
        )
        null_z = np.asarray(result["raw"]["null_z_I"], dtype=np.float32)
        observed_z = np.asarray(result["raw"]["observed_z_I"], dtype=np.float64)
        np.savez_compressed(
            CACHE_DIR / f"{country}_residual_paths.npz",
            observed_z_I=observed_z,
            evaluation_z_I=null_z[1::2],
        )
        print(f"saved {country} residual paths", flush=True)


def _load_paths(country: str) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE_DIR / f"{country}_residual_paths.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing residual-path cache: {path}")
    with np.load(path) as payload:
        return (
            np.asarray(payload["observed_z_I"], dtype=float),
            np.asarray(payload["evaluation_z_I"], dtype=float),
        )


def _global_p(country: str) -> float:
    row = pd.read_csv(_root_for(country) / country / "layer_and_global_scores.csv").iloc[0]
    return float(row["global_rank_pvalue"])


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220)
    plt.close(fig)


def _panel_title(country: str) -> str:
    p = _global_p(country)
    decision = "nonrejection" if p > 0.05 else "rejection"
    return f"{country}: {decision} (global $p={p:.3g}$)"


def make_residual_figure() -> None:
    observed = {country: _load_paths(country)[0] for country in COUNTRIES}
    limit = max(float(np.max(np.abs(values))) for values in observed.values()) * 1.08
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.88, hspace=0.34, wspace=0.18)
    for ax, country in zip(axes.flat, COUNTRIES):
        values = observed[country]
        days = np.arange(1, len(values) + 1)
        color = BLUE if country == "BGR" else RED
        ax.axhline(0.0, color=LIGHT_GREY, linewidth=0.9)
        ax.plot(days, values, color=color, linewidth=1.15)
        ax.scatter(days, values, color=color, s=8, alpha=0.70, linewidths=0)
        ax.set_xlim(1, len(values))
        ax.set_ylim(-limit, limit)
        ax.set_title(_panel_title(country), loc="left")
        ax.grid(axis="y", color="#ececec", linewidth=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"standardized residual $z_k$")
    for ax in axes[-1, :]:
        ax.set_xlabel("held-out day")
    fig.suptitle(
        "Observed observation-step standardized residuals in the two randomized country batches",
        fontsize=13.0,
        color=GREY,
        y=0.965,
    )
    _save(fig, "paper1_six_country_residual_paths")


def make_cumulative_energy_figure() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.0), sharex=True, sharey=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.88, hspace=0.34, wspace=0.18)
    all_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for country in COUNTRIES:
        observed_z, reference_z = _load_paths(country)
        root_n = np.sqrt(len(observed_z))
        observed_curve = np.cumsum(observed_z**2 - 1.0) / root_n
        reference_curves = np.cumsum(reference_z**2 - 1.0, axis=1) / root_n
        local_bound = 0.0
        q05, q95 = np.quantile(reference_curves, [0.05, 0.95], axis=0)
        local_bound = max(
            float(np.max(np.abs(observed_curve))),
            float(np.max(np.abs(q05))),
            float(np.max(np.abs(q95))),
        ) * 1.08
        all_curves[country] = (observed_curve, reference_curves, local_bound)

    for ax, country in zip(axes.flat, COUNTRIES):
        observed_curve, reference_curves, local_bound = all_curves[country]
        days = np.arange(1, len(observed_curve) + 1)
        q05, median, q95 = np.quantile(reference_curves, [0.05, 0.50, 0.95], axis=0)
        color = BLUE if country == "BGR" else RED
        ax.fill_between(days, q05, q95, color=REFERENCE_BLUE, alpha=0.17, linewidth=0)
        ax.plot(days, median, color=REFERENCE_BLUE, linewidth=1.0, linestyle="--")
        ax.plot(days, observed_curve, color=color, linewidth=1.55)
        ax.axhline(0.0, color=LIGHT_GREY, linewidth=0.8)
        ax.set_xlim(1, len(observed_curve))
        ax.set_ylim(-local_bound, local_bound)
        ax.set_title(_panel_title(country), loc="left")
        ax.grid(axis="y", color="#ececec", linewidth=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\sum_{j\leq k}(z_j^2-1)/\sqrt{60}$")
    for ax in axes[-1, :]:
        ax.set_xlabel("held-out day $k$")
    axes[0, 0].plot([], [], color=RED, linewidth=1.55, label="observed")
    axes[0, 0].plot([], [], color=REFERENCE_BLUE, linestyle="--", label="reference median")
    axes[0, 0].fill_between([], [], [], color=REFERENCE_BLUE, alpha=0.17, label="90% pointwise envelope")
    axes[0, 0].legend(frameon=False, fontsize=8.1, loc="upper left")
    fig.suptitle(
        "Cumulative residual energy against matched finite-grid references",
        fontsize=13.0,
        color=GREY,
        y=0.965,
    )
    _save(fig, "paper1_six_country_cumulative_energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Verified residual-path cache directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory for the two generated PDF and PNG figures.",
    )
    return parser.parse_args()


def main(
    cache_dir: Path | None = None,
    figure_dir: Path | None = None,
) -> None:
    global CACHE_DIR, FIGURE_DIR
    if cache_dir is not None:
        CACHE_DIR = cache_dir.resolve()
    if figure_dir is not None:
        FIGURE_DIR = figure_dir.resolve()
    if not all(
        (CACHE_DIR / f"{country}_residual_paths.npz").is_file()
        for country in COUNTRIES
    ):
        _build_cache()
    _configure_style()
    make_residual_figure()
    make_cumulative_energy_figure()
    print("Wrote six-country residual and cumulative-energy figures.", flush=True)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    args = parse_args()
    main(cache_dir=args.cache_dir, figure_dir=args.out_dir)
