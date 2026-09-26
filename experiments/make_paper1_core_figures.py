"""Rebuild the core figures for the paper.

The script uses only deterministic synthetic constructions and frozen CSV
artifacts already cited by the paper.  Run it from any working directory with

    python experiments/make_paper1_core_figures.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "output" / "figures"
GEOMETRY_DIR = ROOT / "cache" / "summaries" / "finite_step_geometry_synthetic"
TEMPORAL_NORDIC_DIR = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_temporal_audit"
    / "nordic_four_block_B5000_frozen"
    / "temporal_windows"
)
SHORT_NORDIC_DIR = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_short_window_audit"
    / "nordic_split30_B5000_frozen"
    / "short_windows"
)
EXPLORATORY_RANDOM_COUNTRY_DIR = (
    ROOT
    / "cache"
    / "summaries"
    / "random_country_5000_reference_draws"
)
FIXED_CALENDAR_HOLDOUT_DIRS = {
    "CZE": (
        ROOT
        / "cache"
        / "summaries"
        / "fixed_calendar_holdout"
        / "causal_window",
        "holdout",
    ),
    "GRC": (
        ROOT
        / "cache"
        / "summaries"
        / "fixed_calendar_holdout"
        / "second_holdout",
        "second_holdout",
    ),
}
FROZEN_RANDOM_COUNTRY_DIR = (
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
MID_GREY = "#8a8a8a"
LIGHT_GREY = "#d7d7d7"
COUNTRY_COLORS = ["#3b6fb6", "#7b5ba7", "#c4423f"]
COUNTRY_MARKERS = ["o", "s", "^"]
REFERENCE_BLUE = BLUE


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220)
    plt.close(fig)


def add_rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edgecolor: str,
    facecolor: str,
    textcolor: str = GREY,
    fontsize: float = 10.2,
    fontweight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        fontweight=fontweight,
        linespacing=1.15,
    )


def make_audit_workflow_figure() -> None:
    """Show the fixed implementation and the two-stage rank calculation."""
    fig, ax = plt.subplots(figsize=(10.8, 5.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = MID_GREY,
        linewidth: float = 1.35,
    ) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": linewidth},
        )

    ax.text(
        0.5,
        0.965,
        "Pilot builds the ruler; evaluation ranks the held-out path once",
        ha="center",
        va="center",
        fontsize=13.6,
        color=GREY,
    )

    add_rounded_box(
        ax,
        0.055,
        0.795,
        0.89,
        0.105,
        "FIX ONE IMPLEMENTATION\n"
        "fitted coefficients + reconstruction + controls + initialization + solver",
        edgecolor=BLUE,
        facecolor="#eaf2f6",
        textcolor=BLUE,
        fontsize=10.0,
        fontweight="bold",
    )

    ax.text(
        0.06,
        0.712,
        "1   PILOT: BUILD AND FREEZE THE RULER",
        ha="left",
        va="center",
        fontsize=10.2,
        color=BLUE,
        fontweight="bold",
    )
    add_rounded_box(
        ax,
        0.075,
        0.565,
        0.235,
        0.105,
        "Pilot paths\nfrom the reference law",
        edgecolor=BLUE,
        facecolor="white",
        textcolor=GREY,
        fontsize=9.5,
    )
    add_rounded_box(
        ax,
        0.385,
        0.545,
        0.42,
        0.145,
        r"Apply $B_c$ and $T$; freeze centres," "\n"
        r"scales, and aggregate score $s$",
        edgecolor=BLUE,
        facecolor="#f5f8fa",
        textcolor=GREY,
        fontsize=9.7,
        fontweight="bold",
    )
    arrow((0.31, 0.617), (0.385, 0.617), BLUE)
    arrow((0.88, 0.795), (0.78, 0.69), BLUE)

    ax.text(
        0.06,
        0.465,
        "2   EVALUATION: RANK ONE HELD-OUT PATH",
        ha="left",
        va="center",
        fontsize=10.2,
        color=RED,
        fontweight="bold",
    )
    add_rounded_box(
        ax,
        0.055,
        0.270,
        0.20,
        0.105,
        "Held-out path",
        edgecolor=RED,
        facecolor="#fff8f8",
        textcolor=GREY,
        fontsize=9.2,
    )
    add_rounded_box(
        ax,
        0.055,
        0.095,
        0.20,
        0.105,
        "Independent evaluation paths\nfrom the reference law",
        edgecolor=BLUE,
        facecolor="white",
        textcolor=GREY,
        fontsize=9.2,
    )
    add_rounded_box(
        ax,
        0.335,
        0.155,
        0.245,
        0.165,
        "Apply the same frozen\n"
        "$B_c$, $T$, and $s$\n"
        "to every path",
        edgecolor=GREY,
        facecolor="#f6f6f6",
        textcolor=GREY,
        fontsize=9.7,
        fontweight="bold",
    )
    add_rounded_box(
        ax,
        0.645,
        0.270,
        0.13,
        0.105,
        "One held-out\nscore",
        edgecolor=RED,
        facecolor="#fff8f8",
        textcolor=GREY,
        fontsize=9.2,
    )
    add_rounded_box(
        ax,
        0.645,
        0.095,
        0.13,
        0.105,
        "Reference\nscores",
        edgecolor=BLUE,
        facecolor="white",
        textcolor=GREY,
        fontsize=9.2,
    )
    add_rounded_box(
        ax,
        0.835,
        0.155,
        0.115,
        0.165,
        "Monte Carlo\np-value",
        edgecolor=RED,
        facecolor="#fff8f8",
        textcolor=GREY,
        fontsize=9.3,
        fontweight="bold",
    )

    arrow((0.255, 0.323), (0.335, 0.270), RED)
    arrow((0.255, 0.148), (0.335, 0.205), BLUE)
    arrow((0.58, 0.270), (0.645, 0.323), RED)
    arrow((0.58, 0.205), (0.645, 0.148), BLUE)
    arrow((0.775, 0.323), (0.835, 0.270), RED)
    arrow((0.775, 0.148), (0.835, 0.205), BLUE)
    arrow((0.595, 0.545), (0.47, 0.32), BLUE)

    ax.text(
        0.5,
        0.025,
        "The pilot bank never enters the Monte Carlo reference; only independent evaluation paths do.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=RED,
        fontweight="bold",
    )
    save_figure(fig, "paper1_audit_workflow")


def lag_correlation(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def audit_metrics(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            abs(np.std(values, ddof=1) - 1.0),
            abs(lag_correlation(values)),
            abs(lag_correlation(values**2)),
        ]
    )


def make_motivation_figure() -> None:
    # A paired Gaussian construction gives standard-normal marginals and
    # near-zero signed lag correlation while retaining moderate dependence in
    # adjacent squared coordinates.  The fixed seed gives a non-boundary
    # illustrative rejection (p=0.033 with 999 evaluation paths).
    rng = np.random.default_rng(1001)
    shocks = rng.normal(size=30)
    independent = rng.normal(size=30)
    signs = rng.choice([-1.0, 1.0], size=30)
    rho = 0.5
    observed = np.empty(60)
    observed[0::2] = shocks
    observed[1::2] = signs * (
        rho * shocks + np.sqrt(1.0 - rho**2) * independent
    )

    pilot_rng = np.random.default_rng(20260717)
    pilot = np.array([audit_metrics(pilot_rng.normal(size=60)) for _ in range(500)])
    centres = np.median(pilot, axis=0)
    scales = np.std(pilot, axis=0, ddof=1)

    evaluation_rng = np.random.default_rng(1)
    evaluation = np.array(
        [audit_metrics(evaluation_rng.normal(size=60)) for _ in range(999)]
    )
    evaluation_scores = np.max((evaluation - centres) / scales, axis=1)
    observed_score = float(np.max((audit_metrics(observed) - centres) / scales))
    rank_p = (1 + np.sum(evaluation_scores >= observed_score)) / 1000

    ordered = np.sort(observed)
    probabilities = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
    normal = NormalDist()
    gaussian_quantiles = np.array([normal.inv_cdf(float(p)) for p in probabilities])
    acf = lag_correlation(observed)
    squared_acf = lag_correlation(observed**2)
    scale = np.std(observed, ddof=1)

    # Integrating the same paired-sign coordinates gives a path-level view.
    # Choose its scale so that the descriptive path R-squared rounds to 0.94.
    time = np.arange(61)
    mean_path = (
        0.95 * np.sin(time / 4.7)
        + 0.45 * np.sin(time / 9.2)
        + 0.018 * time
    )
    cumulative_coordinates = np.concatenate([[0.0], np.cumsum(observed)])

    def path_r_squared(noise_scale: float) -> float:
        path = mean_path + noise_scale * cumulative_coordinates
        return 1.0 - np.sum((path - mean_path) ** 2) / np.sum(
            (path - np.mean(path)) ** 2
        )

    lower, upper = 0.0, 1.0
    while path_r_squared(upper) > 0.94:
        upper *= 2.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if path_r_squared(midpoint) > 0.94:
            lower = midpoint
        else:
            upper = midpoint
    path = mean_path + ((lower + upper) / 2.0) * cumulative_coordinates
    r_squared = path_r_squared((lower + upper) / 2.0)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(time, path, color=RED, linewidth=2.0, label="observed path")
    ax.plot(time, mean_path, color=BLUE, linestyle="--", linewidth=1.7, label="reference mean path")
    ax.set(xlabel="observation index", ylabel="state")
    ax.set_title("(a) Mean evolution\nfit looks acceptable", loc="left")
    ax.text(
        0.04,
        0.90,
        rf"path fit $R^2={r_squared:.2f}$",
        transform=ax.transAxes,
        color=GREY,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": MID_GREY},
    )
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    ax.scatter(gaussian_quantiles, ordered, s=23, color=MID_GREY, alpha=0.85)
    limits = (-2.55, 2.55)
    ax.plot(limits, limits, linestyle="--", color=GREY, linewidth=1.2)
    ax.set(xlim=limits, ylim=limits, xlabel="reference Gaussian quantiles", ylabel="standardized-residual quantiles")
    ax.set_title("(b) Marginal variation\none-dimensional fit looks acceptable", loc="left")
    ax.text(0.04, 0.93, f"sample scale = {scale:.2f}", transform=ax.transAxes, va="top", color=GREY)

    ax = axes[1, 0]
    previous_energy = observed[:-1] ** 2
    current_energy = observed[1:] ** 2
    within_pair = np.arange(len(previous_energy)) % 2 == 0
    ax.scatter(
        previous_energy[~within_pair],
        current_energy[~within_pair],
        s=25,
        facecolors="none",
        edgecolors=MID_GREY,
        label="between pairs",
    )
    ax.scatter(
        previous_energy[within_pair],
        current_energy[within_pair],
        s=25,
        color=RED,
        alpha=0.7,
        label="within pair",
    )
    energy_limit = max(float(np.max(previous_energy)), float(np.max(current_energy))) * 1.05
    ax.plot([0, energy_limit], [0, energy_limit], linestyle="--", color=MID_GREY, linewidth=1.0)
    ax.set(xlim=(0, energy_limit), ylim=(0, energy_limit), xlabel=r"previous squared residual $Z_{k-1}^2$", ylabel=r"current squared residual $Z_k^2$")
    ax.set_title("(c) Joint path law\nresidual energy is serially dependent", loc="left")
    ax.text(
        0.04,
        0.91,
        rf"$\mathrm{{ACF}}(Z,1)={acf:+.2f}$" + "\n" + rf"$\mathrm{{ACF}}(Z^2,1)={squared_acf:+.2f}$",
        transform=ax.transAxes,
        va="top",
        color=RED,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": LIGHT_GREY},
    )
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    ax.hist(evaluation_scores, bins=22, color=LIGHT_GREY, edgecolor="white")
    ax.axvline(observed_score, color=RED, linewidth=2.0)
    ax.set_xlabel(r"global score $S_{\max}$")
    ax.set_ylabel("reference paths")
    ax.set_title("(d) Matched simulation\nthe selected feature is atypical", loc="left")
    ax.text(
        0.95,
        0.82,
        f"rank $p={rank_p:.3f}$",
        transform=ax.transAxes,
        ha="right",
        color=RED,
        fontsize=12,
    )
    ax.text(
        0.95,
        0.70,
        "one score map\nfor held-out and reference paths",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GREY,
    )

    fig.suptitle(
        "A fitted SDE specifies a joint path law, not only a mean path",
        fontsize=14,
        color=GREY,
    )
    save_figure(fig, "paper1_intro_audit_motivation")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_synthetic_restriction_map() -> None:
    """Show what each diagnostic detects without implying causal identification."""
    rows = read_csv(ROOT / "cache" / "summaries" / "bi_synthetic_calibration.csv")
    by_scenario = {row["scenario"]: row for row in rows}

    scenarios = [
        ("Short smoothing", "C_ma3_smooth"),
        ("Long smoothing", "C_ma14_smooth"),
        ("Diffusion too small", "D_sigma_x0.5"),
        ("Diffusion too large", "D_sigma_x2.0"),
        ("Jumps", "E_jump_shock"),
    ]
    diagnostics = [
        ("Scale", "bi_z_std"),
        ("Excess\nkurtosis", "bi_z_kurt"),
        ("ACF(1)", "bi_acf1"),
        ("ACF(7)", "bi_acf7"),
        ("Cumulative\nmaximum", "bi_max_w_norm"),
    ]
    matched = by_scenario["A_true_bm"]
    departures = np.array(
        [
            [
                float(by_scenario[scenario][field]) - float(matched[field])
                for _, field in diagnostics
            ]
            for _, scenario in scenarios
        ]
    )
    scaled = departures / np.max(np.abs(departures), axis=0)

    cmap = LinearSegmentedColormap.from_list(
        "restriction_departure", [BLUE, "#f3f3f3", "#df7900"]
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.25))
    image = ax.imshow(
        scaled,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(diagnostics)), [label for label, _ in diagnostics])
    ax.set_yticks(np.arange(len(scenarios)), [label for label, _ in scenarios])
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(diagnostics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scenarios), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(scaled.shape[0]):
        for column_index in range(scaled.shape[1]):
            value = scaled[row_index, column_index]
            displayed_value = 0.0 if abs(value) < 0.005 else value
            ax.text(
                column_index,
                row_index,
                f"{displayed_value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.64 else GREY,
                fontsize=10.5,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.055)
    colorbar.set_ticks([-1, 0, 1], labels=["below", "matched", "above"])
    colorbar.ax.tick_params(length=0, pad=6)
    colorbar.set_label(
        "Departure from matched mean\n(scaled within each column)", labelpad=10
    )

    fig.suptitle(
        "Different departures can affect the same statistics",
        fontsize=14,
        color=GREY,
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Compare response patterns, not magnitudes between columns. "
        "Values are descriptive, not p-values.",
        ha="center",
        color=GREY,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.23, right=0.88, top=0.73, bottom=0.13)
    save_figure(fig, "paper1_synthetic_restriction_map")


def _select_rows(rows: list[dict], **field_values: float) -> list[dict]:
    """Rows where every named column, cast to float, matches the given value."""
    return [
        row
        for row in rows
        if all(float(row[field]) == value for field, value in field_values.items())
    ]


def make_reference_law_figure() -> None:
    covariance = read_csv(GEOMETRY_DIR / "finite_step_covariance_grid.csv")
    size_grid = read_csv(GEOMETRY_DIR / "analytic_size_grid.csv")

    selected_covariance = _select_rows(covariance, coupling=1.0, h=1.0)
    exact_row = next(row for row in selected_covariance if row["method"] == "exact")
    euler_covariance = sorted(
        (row for row in selected_covariance if row["method"] == "euler"),
        key=lambda row: int(row["substeps"]),
    )

    selected_size = sorted(
        (
            row
            for row in _select_rows(size_grid, coupling=1.0, h=1.0)
            if row["method"] == "euler"
        ),
        key=lambda row: int(row["substeps"]),
    )

    substeps = np.array([int(row["substeps"]) for row in euler_covariance])
    second_eigenvalues = np.array([float(row["eigenvalue_2"]) for row in euler_covariance])
    exact_second = float(exact_row["eigenvalue_2"])
    # Case 3: correct finite-step geometry applied to the wrong target law.
    target_mismatch_size = np.array(
        [float(row["combined_size"]) for row in selected_size]
    )
    # Case 1: Gaussian draws audited with their matching Euler covariance.
    covariance_matched_size = np.array(
        [float(row["matched_solver_size"]) for row in selected_size]
    )
    # Case 2: draws from the implemented law, audited by the instantaneous
    # zero-support rule.  This is the arm the abstract's claim refers to.
    instantaneous_rule_size = np.array(
        [float(row["instantaneous_rule_size"]) for row in selected_size]
    )

    # Sized for \textwidth in the paper: a wider canvas would be scaled down far
    # enough to make the legend and annotations unreadable in print.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(substeps, second_eigenvalues, marker="o", color=BLUE, label="Euler covariance")
    ax.axhline(exact_second, color=GREY, linestyle="--", label="exact transition")
    ax.scatter([1], [0], color=RED, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(substeps, [str(value) for value in substeps])
    ax.set_ylim(-0.008, max(second_eigenvalues) * 1.17)
    ax.set_xlabel(r"Euler substeps $m$")
    ax.set_ylabel("second covariance eigenvalue")
    ax.set_title("(a) Observation-step support\nchanges at the first drift update", loc="left")
    ax.text(1.18, 0.006, "rank 1", color=RED)
    ax.text(2.6, second_eigenvalues[1] - 0.002, r"rank 2 for $m\geq2$", color=BLUE)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.24, 0.02))

    ax = axes[1]
    # Curves are directly labelled rather than keyed to a legend: at \textwidth
    # a three-entry legend crowds the curves it describes.
    ax.plot(substeps, instantaneous_rule_size, marker="o", color=RED)
    ax.plot(
        substeps, target_mismatch_size, marker="^", color=MID_GREY, linestyle=":"
    )
    ax.plot(substeps, covariance_matched_size, marker="s", color=BLUE)
    ax.axhline(0.05, color=GREY, linestyle="--", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xticks(substeps, [str(value) for value in substeps])
    ax.set_ylim(-0.04, 1.34)
    ax.set_xlabel(r"Euler substeps $m$")
    ax.set_ylabel("rejection rate")
    ax.set_title(
        "(b) Only the instantaneous rule fails\nfor every multi-substep implementation",
        loc="left",
    )
    ax.text(
        6.0,
        1.10,
        "(2) instantaneous zero-support rule\n1.00 for every $m\\geq2$",
        color=RED,
        fontsize=8.0,
        ha="center",
        va="center",
        linespacing=1.35,
    )
    ax.annotate(
        "correct at $m=1$, where the\nrule matches the implementation",
        xy=(1, 0.05),
        xytext=(2.6, 0.40),
        color=RED,
        fontsize=7.6,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.8),
    )
    ax.annotate(
        "(3) exact-SDE draws, Euler reference",
        xy=(12, 0.043),
        xytext=(4.2, 0.30),
        color=MID_GREY,
        fontsize=7.6,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=MID_GREY, linewidth=0.8),
    )
    ax.text(
        4.2, 0.155, "(1) covariance-matched audit:  nominal 0.05", color=BLUE, fontsize=8.0,
        ha="left",
    )

    fig.suptitle(
        "The reference law determines which restrictions are valid",
        fontsize=13,
        color=GREY,
    )
    save_figure(fig, "paper1_geometry_protocol_comparison")


def make_nordic_audit_figure() -> None:
    """Localize the three anchor-window reference-tail positions."""
    countries = ["FIN", "NOR", "SWE"]
    component_rows = [
        row
        for row in read_csv(TEMPORAL_NORDIC_DIR / "component_summaries.csv")
        if row["window_label"] == "anchor"
        and row["metric"] == "bracket_I_max_abs_cum"
    ]
    score_rows = [
        row
        for row in read_csv(TEMPORAL_NORDIC_DIR / "window_scores.csv")
        if row["window_label"] == "anchor"
    ]
    components = {row["country"]: row for row in component_rows}
    scores = {row["country"]: row for row in score_rows}
    observed = np.array(
        [float(components[country]["observed"]) for country in countries]
    )
    reference_draws: dict[str, np.ndarray] = {}
    tail_counts: dict[str, int] = {}
    for country, held_out in zip(countries, observed):
        evaluation_rows = [
            row
            for row in read_csv(
                TEMPORAL_NORDIC_DIR
                / country
                / "anchor"
                / "null_metric_draws.csv"
            )
            if int(row["bootstrap_id"]) % 2 == 1
        ]
        reference = np.array(
            [float(row["bracket_I_max_abs_cum"]) for row in evaluation_rows]
        )
        expected_n = int(scores[country]["evaluation_n"])
        if len(reference) != expected_n:
            raise ValueError(
                f"Expected {expected_n} evaluation paths for {country}, "
                f"found {len(reference)}."
            )
        reference_draws[country] = reference
        tail_counts[country] = int(np.sum(reference >= held_out))

    counts = [tail_counts[country] for country in countries]
    if counts != [0, 3, 1]:
        raise ValueError(f"Unexpected anchor tail counts: {counts}")

    layer_labels = [
        "Global\nmax",
        "Squared\nresiduals",
        "Signed\nresiduals",
        "Generator\n(separate)",
    ]
    fields = [
        "global_rank_pvalue",
        "bracket_rank_pvalue",
        "martingale_rank_pvalue",
        "generator_rank_pvalue",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    ax = axes[1]
    row_positions = np.arange(len(countries), dtype=float)[::-1]
    jitter_rng = np.random.default_rng(20260722)
    for row_position, country in zip(row_positions, countries):
        reference = reference_draws[country]
        jitter = jitter_rng.uniform(-0.17, 0.17, size=len(reference))
        ax.scatter(
            reference,
            row_position + jitter,
            s=7,
            color=REFERENCE_BLUE,
            alpha=0.14,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    ax.scatter(
        observed,
        row_positions,
        color=RED,
        marker="D",
        s=52,
        zorder=4,
        label="held-out score",
    )
    for held_out, row_position in zip(observed, row_positions):
        ax.text(
            held_out + 0.12,
            row_position,
            f"{held_out:.2f}",
            va="center",
            color=RED,
        )
    ax.scatter(
        [],
        [],
        s=18,
        color=REFERENCE_BLUE,
        alpha=0.40,
        label="2,500 evaluation references",
    )
    ax.set_yticks(row_positions, countries)
    ax.set_xlim(0, max(observed) * 1.13)
    ax.set_ylim(-0.55, 2.85)
    ax.set_xlabel(
        r"dominant statistic $T_E^{\mathrm{cum}}="
        r"K^{-1/2}\max_j\left|\sum_{k\leq j}(z_{I,k}^2-1)\right|$"
    )
    ax.set_title(
        f"(b) Only {counts[0]}, {counts[1]}, and {counts[2]} of 2,500 references reach\n"
        "the FIN, NOR, and SWE held-out values",
        loc="left",
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=2,
    )

    ax = axes[0]
    lower_limit = 3.0e-4
    ax.axhspan(lower_limit, 0.05, color="#fbefef", zorder=0)
    offsets = [-0.12, 0.0, 0.12]
    for country, color, marker, offset in zip(
        countries, COUNTRY_COLORS, COUNTRY_MARKERS, offsets
    ):
        values = [float(scores[country][field]) for field in fields]
        ax.scatter(
            np.arange(len(fields)) + offset,
            values,
            marker=marker,
            s=46,
            color=color,
            label=country,
            zorder=3,
        )
    ax.axhline(0.05, color=GREY, linestyle="--", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_ylim(lower_limit, 0.45)
    ax.set_yticks(
        [4e-4, 1e-3, 1e-2, 5e-2, 1e-1],
        ["0.0004", "0.001", "0.01", "0.05", "0.1"],
    )
    ax.set_xticks(np.arange(len(fields)), layer_labels, rotation=0)
    ax.set_ylabel("Monte Carlo reference rank")
    ax.set_title(
        "(a) Squared-residual statistics\ndrive the extreme global rank",
        loc="left",
    )
    ax.text(3.08, 0.039, "0.05 marker", color=RED, ha="right", fontsize=8.8)
    ax.annotate(
        "",
        xy=(0.88, 0.006),
        xytext=(0.12, 0.006),
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.35},
    )
    ax.text(0.50, 0.0073, "localizes", color=RED, ha="center", fontsize=8.6)
    ax.legend(frameon=False, ncol=3, loc="upper center")

    fig.suptitle(
        r"Extreme global rank $\longrightarrow$ grouped squared-residual rank "
        r"$\longrightarrow$ dominant cumulative statistic in the extreme reference tail",
        fontsize=13.5,
        color=GREY,
    )
    save_figure(fig, "paper1_nordic_sde_native_audit")


def _country_global_reference_scores(
    root: Path, country: str
) -> tuple[np.ndarray, float, float]:
    """Reconstruct odd-bank global scores from the recorded pilot ruler."""
    country_dir = root / country
    draws = read_csv(country_dir / "null_metric_draws.csv")
    component_rows = read_csv(country_dir / "component_summary.csv")
    score_rows = read_csv(country_dir / "layer_and_global_scores.csv")
    if len(score_rows) != 1:
        raise ValueError(f"Expected one score row for {country}, found {len(score_rows)}.")
    score = score_rows[0]
    components = {row["metric"]: row for row in component_rows}
    primary_metrics = score["primary_component_metrics"].split(",")
    evaluation_rows = [row for row in draws if int(row["bootstrap_id"]) % 2 == 1]
    expected_n = int(score["evaluation_n"])
    if len(evaluation_rows) != expected_n:
        raise ValueError(
            f"Expected {expected_n} evaluation paths for {country}, "
            f"found {len(evaluation_rows)}."
        )

    departures = []
    for metric in primary_metrics:
        component = components[metric]
        values = np.array([float(row[metric]) for row in evaluation_rows])
        center = float(component["pilot_median"])
        scale = float(component["pilot_std"])
        standardized = (values - center) / scale
        if component["tail"] == "centered":
            departure = np.abs(standardized)
        elif component["tail"] == "upper":
            departure = np.maximum(standardized, 0.0)
        else:
            raise ValueError(
                f"Unknown registered tail {component['tail']!r} for {country}/{metric}."
            )
        departures.append(departure)

    global_reference = np.max(np.column_stack(departures), axis=1)
    observed = float(score["observed_global_score"])
    recorded_p = float(score["global_rank_pvalue"])
    reconstructed_p = float(
        (1 + np.sum(global_reference >= observed)) / (len(global_reference) + 1)
    )
    if not np.isclose(reconstructed_p, recorded_p, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Global-rank reconstruction failed for {country}: "
            f"{reconstructed_p} != {recorded_p}."
        )
    return global_reference, observed, recorded_p


def make_registered_country_replication_figure() -> None:
    """Show the descriptive LTU/MDA/SVN reference comparisons."""
    countries = ["LTU", "MDA", "SVN"]
    score_rows = {
        row["country"]: row
        for row in read_csv(FROZEN_RANDOM_COUNTRY_DIR / "country_scores.csv")
    }
    energy_rows = {
        row["country"]: row
        for row in read_csv(FROZEN_RANDOM_COUNTRY_DIR / "component_summaries.csv")
        if row["metric"] == "bracket_I_mean_z2"
    }
    missing = [
        country
        for country in countries
        if country not in score_rows or country not in energy_rows
    ]
    if missing:
        raise ValueError(f"Missing frozen replication rows for {missing}.")

    global_p = np.array(
        [float(score_rows[country]["global_rank_pvalue"]) for country in countries]
    )
    observed = np.array(
        [float(energy_rows[country]["observed"]) for country in countries]
    )
    q025 = np.array(
        [float(energy_rows[country]["evaluation_q025"]) for country in countries]
    )
    q500 = np.array(
        [float(energy_rows[country]["evaluation_q500"]) for country in countries]
    )
    q975 = np.array(
        [float(energy_rows[country]["evaluation_q975"]) for country in countries]
    )

    expected_p = np.array(
        [0.004398240703718513, 0.00039984006397441024, 0.001999200319872051]
    )
    expected_observed = np.array(
        [0.35766957039541103, 3.9348936218717814, 0.29671254534516933]
    )
    if not np.allclose(global_p, expected_p, rtol=0.0, atol=1e-12):
        raise ValueError("Unexpected global ranks in the frozen replication artifact.")
    if not np.allclose(observed, expected_observed, rtol=0.0, atol=1e-12):
        raise ValueError("Unexpected energy summaries in the frozen replication artifact.")

    x = np.arange(len(countries), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.78, wspace=0.28)

    ax = axes[0]
    lower_limit = 3.0e-4
    ax.axhline(0.05, color=GREY, linestyle="--", linewidth=1.1)
    ax.scatter(
        x,
        global_p,
        marker="D",
        s=66,
        facecolor=RED,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    for xi, value in zip(x, global_p):
        ax.annotate(
            f"{value:.4f}",
            (xi, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=RED,
            fontsize=9,
        )
    ax.text(2.38, 0.040, "0.05 marker", color=RED, ha="right", fontsize=8.8)
    ax.set_yscale("log")
    ax.set_ylim(lower_limit, 0.11)
    ax.set_yticks(
        [4e-4, 1e-3, 1e-2, 5e-2, 1e-1],
        ["0.0004", "0.001", "0.01", "0.05", "0.1"],
    )
    ax.set_xlim(-0.45, 2.45)
    ax.set_xticks(x, countries)
    ax.set_ylabel(r"global $S_{\mathrm{CE}}$ reference rank")
    ax.set_title("(a) Descriptive Monte Carlo reference ranks", loc="left")

    ax = axes[1]
    ax.errorbar(
        x,
        q500,
        yerr=np.vstack((q500 - q025, q975 - q500)),
        fmt="o",
        markersize=6,
        color=REFERENCE_BLUE,
        ecolor=REFERENCE_BLUE,
        elinewidth=5.5,
        alpha=0.62,
        capsize=0,
        label="evaluation median and 95% interval",
        zorder=2,
    )
    ax.scatter(
        x,
        observed,
        marker="D",
        s=66,
        facecolor=RED,
        edgecolor="white",
        linewidth=0.7,
        label="held-out value",
        zorder=3,
    )
    directions = ["deficit", "excess", "deficit"]
    for xi, value, direction in zip(x, observed, directions):
        ax.annotate(
            direction,
            (xi, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=RED,
            fontsize=8.8,
        )
    ax.axhline(1.0, color=LIGHT_GREY, linewidth=0.9, zorder=0)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0.0, 4.45)
    ax.set_xticks(x, countries)
    ax.set_ylabel(r"mean squared standardized residual")
    ax.set_title("(b) Transition-energy direction", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Randomized draw from an outcome-filtered eligible pool",
        fontsize=13.5,
        color=GREY,
        y=0.96,
    )
    save_figure(fig, "paper1_registered_country_replication")


def make_sir_country_diagnostic_figure() -> None:
    """Show the fixed-calendar holdouts and the descriptive batch on one energy scale."""
    holdouts = ["CZE", "GRC"]
    expected = {
        "CZE": (2.77891820987, 0.0611755297881, 1.41235995224, 2.89740056487,
                0.638462420818),
        "GRC": (5.92255647501, 0.000399840063974, 1.38872348538, 2.92648363871,
                0.371718154851),
    }

    score, rank, ref_q500, ref_q950 = [], [], [], []
    obs, q025, q500, q975 = [], [], [], []
    for country in holdouts:
        directory, prefix = FIXED_CALENDAR_HOLDOUT_DIRS[country]
        rows = read_csv(directory / (prefix + "_results.csv"))
        if len(rows) != 1 or rows[0]["country"] != country:
            raise ValueError(f"Unexpected holdout result rows for {country}.")
        row = rows[0]
        values = (
            float(row["observed_global_score"]),
            float(row["global_rank_pvalue"]),
            float(row["global_null_q500"]),
            float(row["global_null_q950"]),
            float(row["observed_bracket_I_mean_z2"]),
        )
        if not np.allclose(values, expected[country], rtol=0.0, atol=1e-11):
            raise ValueError(f"Unexpected frozen holdout summary for {country}.")
        score.append(values[0])
        rank.append(values[1])
        ref_q500.append(values[2])
        ref_q950.append(values[3])
        obs.append(values[4])

        draws = read_csv(directory / (prefix + "_reference_draws.csv"))
        # The freeze record splits the bank as: even bootstrap_id pilot,
        # odd bootstrap_id reference.  Only the reference half is plotted.
        bank = np.array(
            [
                float(d["bracket_I_mean_z2"])
                for d in draws
                if int(d["bootstrap_id"]) % 2 == 1
            ]
        )
        if bank.size != int(row["reference_n"]):
            raise ValueError(f"Unexpected reference-bank size for {country}.")
        q025.append(float(np.quantile(bank, 0.025)))
        q500.append(float(np.quantile(bank, 0.500)))
        q975.append(float(np.quantile(bank, 0.975)))

    descriptive = ["LTU", "MDA", "SVN"]
    energy_rows = {
        row["country"]: row
        for row in read_csv(FROZEN_RANDOM_COUNTRY_DIR / "component_summaries.csv")
        if row["metric"] == "bracket_I_mean_z2"
    }
    expected_observed = {
        "LTU": 0.35766957039541103,
        "MDA": 3.9348936218717814,
        "SVN": 0.29671254534516933,
    }
    for country in descriptive:
        if country not in energy_rows:
            raise ValueError(f"Missing frozen replication row for {country}.")
        row = energy_rows[country]
        value = float(row["observed"])
        if not np.isclose(value, expected_observed[country], rtol=0.0, atol=1e-12):
            raise ValueError(f"Unexpected frozen energy summary for {country}.")
        obs.append(value)
        q025.append(float(row["evaluation_q025"]))
        q500.append(float(row["evaluation_q500"]))
        q975.append(float(row["evaluation_q975"]))

    countries = holdouts + descriptive
    score = np.array(score)
    ref_q500 = np.array(ref_q500)
    ref_q950 = np.array(ref_q950)
    obs = np.array(obs)
    q025 = np.array(q025)
    q500 = np.array(q500)
    q975 = np.array(q975)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35),
                             gridspec_kw={"width_ratios": [1.0, 1.55]})
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.78, wspace=0.26)

    ax = axes[0]
    xh = np.arange(len(holdouts), dtype=float)
    ax.vlines(
        xh, ref_q500, ref_q950, color=BLUE, linewidth=7.5, alpha=0.55, zorder=1,
        label="reference median to 95th percentile",
    )
    ax.scatter(
        xh, score, marker="D", s=66, facecolor=RED, edgecolor="white",
        linewidth=0.7, zorder=3, label="held-out score",
    )
    for xi, value, rank_value in zip(xh, score, rank):
        ax.annotate(
            "rank %.4f" % rank_value,
            (xi, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=RED,
            fontsize=8.8,
        )
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 7.2)
    ax.set_xticks(xh, holdouts)
    ax.set_ylabel("global centring-energy score")
    ax.set_title("(a) Fixed-calendar holdouts: score vs reference bank",
                 loc="left", fontsize=10.0)
    ax.legend(frameon=False, loc="lower right", fontsize=8.2)

    ax = axes[1]
    x = np.arange(len(countries), dtype=float)
    ax.axvspan(-0.55, 1.5, color="#eaf1f6", zorder=-2, linewidth=0)
    ax.axvspan(1.5, 4.55, color="#f4f4f4", zorder=-2, linewidth=0)
    ax.vlines(
        x, q025, q975, color=BLUE, linewidth=7.5, alpha=0.55, zorder=1,
        label="reference 95% interval",
    )
    ax.scatter(
        x, q500, marker="_", s=170, color=GREY, linewidth=1.4, zorder=2,
        label="reference median",
    )
    ax.scatter(
        x, obs, marker="D", s=62, facecolor=RED, edgecolor="white",
        linewidth=0.7, zorder=3, label="held-out value",
    )
    for xi, value in zip(x, obs):
        ax.annotate(
            "deficit" if value < 1.0 else "excess",
            (xi, value),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            color=RED,
            fontsize=8.5,
        )
    ax.axhline(1.0, color=LIGHT_GREY, linewidth=0.9, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(0.18, 6.4)
    ax.set_xticks(x, countries)
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0], ["0.25", "0.5", "1", "2", "4"])
    ax.set_ylabel("mean squared standardized residual")
    ax.set_title("(b) Transition-energy direction, all five windows",
                 loc="left", fontsize=10.5)
    ax.text(0.5, 5.6, "rules recorded before evaluation", ha="center",
            va="top", fontsize=8.2, color=GREY)
    ax.text(3.0, 5.6, "descriptive: outcome-filtered pool", ha="center",
            va="top", fontsize=8.2, color=GREY)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, fontsize=8.2, handletextpad=0.5, columnspacing=1.8)

    fig.suptitle(
        "Fixed-calendar holdouts and the descriptive batch on one energy scale",
        fontsize=13.0,
        color=GREY,
        y=0.955,
    )
    save_figure(fig, "paper1_sir_country_diagnostics")


def make_random_country_replication_figure() -> None:
    """Compare two descriptive randomized new-country batches."""
    batches = [
        (
            "Exploratory precision rerun",
            EXPLORATORY_RANDOM_COUNTRY_DIR,
            ["BGR", "HUN", "SVK"],
        ),
        (
            "Score/draw-fixed batch",
            FROZEN_RANDOM_COUNTRY_DIR,
            ["LTU", "MDA", "SVN"],
        ),
    ]
    countries = [country for _label, _root, group in batches for country in group]
    roots = {
        country: root
        for _label, root, group in batches
        for country in group
    }
    reference: dict[str, np.ndarray] = {}
    observed: dict[str, float] = {}
    global_p: dict[str, float] = {}
    for country in countries:
        reference[country], observed[country], global_p[country] = (
            _country_global_reference_scores(roots[country], country)
        )

    expected_p = {
        "BGR": 0.31827269092363053,
        "HUN": 0.006397441023590564,
        "SVK": 0.00039984006397441024,
        "LTU": 0.004398240703718513,
        "MDA": 0.00039984006397441024,
        "SVN": 0.001999200319872051,
    }
    for country, expected in expected_p.items():
        if not np.isclose(global_p[country], expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Unexpected global rank for {country}: {global_p[country]} != {expected}."
            )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.05),
        gridspec_kw={"width_ratios": [0.84, 1.25]},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.185, top=0.79, wspace=0.16)

    # Panel (a): the decision pattern, with the two evidence statuses separated.
    ax = axes[0]
    x_positions = np.array([0.0, 1.0, 2.0, 3.75, 4.75, 5.75])
    lower_limit = 3.0e-4
    ax.axhspan(lower_limit, 0.05, color="#fbefef", zorder=0)
    ax.axvspan(-0.48, 2.48, color="#edf4f8", alpha=0.38, linewidth=0, zorder=-2)
    ax.axvspan(3.27, 6.23, color="#f3f3f3", alpha=0.55, linewidth=0, zorder=-2)
    ax.axvline(3.125, color=LIGHT_GREY, linewidth=0.9, zorder=0)
    for x, country in zip(x_positions, countries):
        reject = global_p[country] <= 0.05
        ax.scatter(
            x,
            global_p[country],
            marker="D",
            s=58 if reject else 68,
            facecolor=RED if reject else "white",
            edgecolor=RED,
            linewidth=1.5 if not reject else 0.7,
            zorder=4,
        )
        p_label = (
            f"{global_p[country]:.4f}"
            if global_p[country] < 0.01
            else f"{global_p[country]:.3f}"
        )
        ax.annotate(
            p_label,
            (x, global_p[country]),
            xytext=(0, 8 if country != "BGR" else -15),
            textcoords="offset points",
            ha="center",
            va="bottom" if country != "BGR" else "top",
            color=RED,
            fontsize=8.4,
        )
    ax.axhline(0.05, color=GREY, linestyle="--", linewidth=1.1)
    ax.text(6.05, 0.039, "0.05 marker", color=RED, ha="right", fontsize=8.8)
    ax.annotate(
        "inside reference cloud",
        xy=(x_positions[0], global_p["BGR"]),
        xytext=(0.55, 0.22),
        textcoords="data",
        color=GREY,
        fontsize=8.8,
        arrowprops={"arrowstyle": "-", "color": MID_GREY, "lw": 0.9},
    )
    ax.text(
        1.0,
        -0.12,
        "exploratory precision rerun",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        color=BLUE,
        fontsize=8.7,
    )
    ax.text(
        4.75,
        -0.12,
        "score/draw-fixed batch",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        color=GREY,
        fontsize=8.7,
    )
    ax.set_yscale("log")
    ax.set_ylim(lower_limit, 0.62)
    ax.set_yticks(
        [4e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        ["0.0004", "0.001", "0.01", "0.05", "0.1", "0.5"],
    )
    ax.set_xlim(-0.52, 6.27)
    ax.set_xticks(x_positions, countries)
    ax.set_ylabel("global Monte Carlo reference rank")
    ax.set_title(
        "(a) Five ranks are below 0.05; BGR lies\ninside its reference distribution",
        loc="left",
    )

    # Panel (b): all six held-out global scores against their own evaluation banks.
    ax = axes[1]
    row_positions = np.array([6.2, 5.2, 4.2, 2.6, 1.6, 0.6])
    jitter_rng = np.random.default_rng(20260813)
    max_raw_score = max(
        max(float(np.max(reference[country])), observed[country])
        for country in countries
    )
    for y, country in zip(row_positions, countries):
        transformed_reference = np.log1p(reference[country])
        jitter = jitter_rng.uniform(-0.17, 0.17, size=len(transformed_reference))
        ax.scatter(
            transformed_reference,
            y + jitter,
            s=7,
            color=REFERENCE_BLUE,
            alpha=0.13,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        reject = global_p[country] <= 0.05
        x_held = float(np.log1p(observed[country]))
        ax.scatter(
            [x_held],
            [y],
            marker="D",
            s=54 if reject else 66,
            facecolor=RED if reject else "white",
            edgecolor=RED,
            linewidth=1.55 if not reject else 0.7,
            zorder=4,
        )
        p_text = (
            f"rank={global_p[country]:.4f}"
            if global_p[country] < 0.01
            else f"rank={global_p[country]:.3f}"
        )
        label_left = country == "MDA"
        ax.annotate(
            p_text,
            (x_held, y),
            xytext=(-7 if label_left else 7, 0),
            textcoords="offset points",
            ha="right" if label_left else "left",
            va="center",
            color=RED,
            fontsize=8.2,
        )
    ax.axhline(3.4, color=LIGHT_GREY, linewidth=0.9)
    ax.text(
        0.985,
        0.965,
        "exploratory precision rerun",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=BLUE,
        fontsize=8.6,
    )
    ax.text(
        0.985,
        0.445,
        "score/draw-fixed batch",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GREY,
        fontsize=8.6,
    )
    ax.scatter([], [], marker="D", s=52, color=RED, label="held-out global score")
    ax.scatter(
        [],
        [],
        s=18,
        color=REFERENCE_BLUE,
        alpha=0.40,
        label="2,500 evaluation references",
    )
    raw_ticks = np.array([0, 1, 2, 4, 8, 16, 32], dtype=float)
    raw_ticks = raw_ticks[raw_ticks <= max(32.0, max_raw_score * 1.05)]
    ax.set_xticks(np.log1p(raw_ticks), [str(int(value)) for value in raw_ticks])
    ax.set_xlim(-0.03, np.log1p(max_raw_score * 1.18))
    ax.set_ylim(0.05, 6.75)
    ax.set_yticks(row_positions, countries)
    ax.set_xlabel(r"pilot-standardized global score $S_{\mathrm{CE}}$ (compressed axis)")
    ax.set_title(
        "(b) BGR stays inside its reference cloud;\nthe other held-out scores lie in the tail",
        loc="left",
    )
    ax.legend(
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.49, 0.49),
        ncol=2,
        fontsize=8.5,
    )

    fig.suptitle(
        r"Two outcome-filtered country batches $\longrightarrow$ descriptive reference ranks",
        fontsize=13.5,
        color=GREY,
        y=0.965,
    )
    save_figure(fig, "paper1_random_country_replication")


def make_nordic_window_sensitivity_figure() -> None:
    """Summarize the frozen post-hoc temporal and horizon extensions."""
    countries = ["FIN", "NOR", "SWE"]
    window_order = [
        "m2_a",
        "m2_b",
        "m1_a",
        "m1_b",
        "anchor_a",
        "anchor_b",
        "p1_a",
        "p1_b",
    ]
    short_rows = read_csv(SHORT_NORDIC_DIR / "window_scores.csv")
    temporal_rows = read_csv(TEMPORAL_NORDIC_DIR / "window_scores.csv")
    short_components = read_csv(SHORT_NORDIC_DIR / "component_summaries.csv")
    temporal_components = read_csv(
        TEMPORAL_NORDIC_DIR / "component_summaries.csv"
    )
    short_by_key = {
        (row["country"], row["window_label"]): row for row in short_rows
    }
    temporal_by_key = {
        (row["country"], row["window_label"]): row for row in temporal_rows
    }
    short_component_by_key = {
        (row["country"], row["window_label"], row["metric"]): row
        for row in short_components
    }
    temporal_component_by_key = {
        (row["country"], row["window_label"], row["metric"]): row
        for row in temporal_components
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.15),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.34, 1.0]},
    )

    ax = axes[0]
    positions = np.arange(len(window_order), dtype=float)
    lower_limit = 3.0e-4
    ax.axhspan(lower_limit, 0.05, color="#fbefef", zorder=0)
    for country, color, marker in zip(countries, COUNTRY_COLORS, COUNTRY_MARKERS):
        values = np.array(
            [
                float(short_by_key[(country, label)]["global_rank_pvalue"])
                for label in window_order
            ]
        )
        ax.plot(
            positions,
            values,
            color=color,
            linewidth=1.0,
            alpha=0.48,
            zorder=1,
        )
        ax.scatter(
            positions,
            values,
            color=color,
            marker=marker,
            s=44,
            label=f"{country}: 8 windows",
            zorder=3,
        )

    representative = float(
        short_by_key[("NOR", "m2_a")]["global_rank_pvalue"]
    )
    ax.scatter(
        [0],
        [representative],
        facecolors="white",
        edgecolors=RED,
        marker="D",
        linewidths=1.8,
        s=105,
        zorder=5,
    )
    ax.annotate(
        r"NOR m2/a: rank $=0.444$",
        xy=(0, representative),
        xytext=(0.30, 0.27),
        textcoords="data",
        color=RED,
        fontsize=9.2,
        arrowprops={"arrowstyle": "-", "color": MID_GREY, "lw": 0.9},
    )
    ax.axhline(0.05, color=GREY, linestyle="--", linewidth=1.15)
    ax.text(
        7.05,
        0.038,
        "0.05 marker",
        color=RED,
        ha="right",
        va="top",
        fontsize=8.8,
    )
    ax.set_yscale("log")
    ax.set_ylim(lower_limit, 0.8)
    ax.set_yticks(
        [4e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        ["0.0004", "0.001", "0.01", "0.05", "0.1", "0.5"],
    )
    ax.set_xlim(-0.42, 7.42)
    ax.set_xticks(positions, ["a", "b", "a", "b", "a", "b", "a", "b"])
    for center, label in zip([0.5, 2.5, 4.5, 6.5], ["m2", "m1", "anchor", "p1"]):
        ax.text(
            center,
            -0.16,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.5,
            color=GREY,
        )
    ax.set_ylabel("global Monte Carlo reference rank")
    ax.set_title(
        "(a) Reference ranks across 24\nnon-overlapping 30-increment windows",
        loc="left",
        x=0.02,
    )
    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.96),
        fontsize=8.5,
    )

    comparison = [
        (
            "Parent: 60 inc.",
            temporal_component_by_key[("NOR", "m2", "bracket_I_mean_z2")],
            temporal_by_key[("NOR", "m2")],
            False,
            TEMPORAL_NORDIC_DIR / "NOR" / "m2" / "null_metric_draws.csv",
        ),
        (
            "First half: 30 inc.",
            short_component_by_key[("NOR", "m2_a", "bracket_I_mean_z2")],
            short_by_key[("NOR", "m2_a")],
            True,
            SHORT_NORDIC_DIR / "NOR" / "m2_a" / "null_metric_draws.csv",
        ),
        (
            "Second half: 30 inc.",
            short_component_by_key[("NOR", "m2_b", "bracket_I_mean_z2")],
            short_by_key[("NOR", "m2_b")],
            False,
            SHORT_NORDIC_DIR / "NOR" / "m2_b" / "null_metric_draws.csv",
        ),
    ]

    ax = axes[1]
    row_positions = np.array([2.0, 1.0, 0.0])
    ax.axvline(1.0, color=MID_GREY, linestyle="--", linewidth=1.05)
    for y, (label, component, score, compatible, null_draw_path) in zip(
        row_positions, comparison
    ):
        observed = float(component["observed"])
        global_p = float(score["global_rank_pvalue"])
        evaluation_rows = [
            row
            for row in read_csv(null_draw_path)
            if int(row["bootstrap_id"]) % 2 == 1
        ]
        bootstrap_ids = np.array(
            [int(row["bootstrap_id"]) for row in evaluation_rows], dtype=float
        )
        null_energy = np.array(
            [float(row["bracket_I_mean_z2"]) for row in evaluation_rows]
        )
        jitter = (
            np.mod(bootstrap_ids * 0.6180339887498949, 1.0) - 0.5
        ) * 0.30
        ax.scatter(
            null_energy,
            y + jitter,
            s=7,
            color=REFERENCE_BLUE,
            alpha=0.14,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        ax.scatter(
            [observed],
            [y],
            marker="D",
            s=58,
            color="white" if compatible else RED,
            edgecolor=RED,
            linewidth=1.7,
            zorder=3,
        )
        ax.text(
            observed + 0.045,
            y + 0.13,
            f"{observed:.3f}",
            color=RED,
            fontsize=8.8,
        )
        ax.text(
            2.24,
            y,
            f"{global_p:.3f}",
            ha="right",
            va="center",
            color=GREY,
            fontsize=9.0,
        )

    ax.set_yticks(row_positions, [item[0] for item in comparison])
    ax.set_xlim(0.15, 2.28)
    ax.set_ylim(-0.45, 2.72)
    ax.set_xticks([0.25, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel(r"mean squared standardized residual $\mathbb{E}[z^2]$")
    ax.set_title(
        "(b) NOR m2: reference values\nfor the parent and its halves",
        loc="left",
    )
    ax.scatter([], [], marker="D", s=50, color=RED, label="held-out value")
    ax.scatter(
        [],
        [],
        s=18,
        color=REFERENCE_BLUE,
        alpha=0.40,
        label="2,500 evaluation references",
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=2,
        fontsize=8.0,
    )
    ax.text(2.24, 2.34, "global rank", color=GREY, ha="right", fontsize=8.8)
    ax.text(
        1.03,
        2.34,
        r"reference value $=1$",
        color=GREY,
        ha="left",
        fontsize=8.4,
    )
    fig.suptitle(
        r"Split the 12 parent windows $\longrightarrow$ 23 of 24 ranks remain below 0.05",
        fontsize=13.5,
        color=GREY,
    )
    save_figure(fig, "paper1_nordic_window_sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory for generated PDF and PNG files.",
    )
    return parser.parse_args()


def main(out_dir: Path | None = None) -> None:
    global FIGURE_DIR
    if out_dir is not None:
        FIGURE_DIR = out_dir.resolve()
    configure_style()
    make_motivation_figure()
    make_audit_workflow_figure()
    make_synthetic_restriction_map()
    make_reference_law_figure()
    make_nordic_audit_figure()
    make_registered_country_replication_figure()
    make_sir_country_diagnostic_figure()
    make_random_country_replication_figure()
    make_nordic_window_sensitivity_figure()


if __name__ == "__main__":
    main(parse_args().out_dir)
