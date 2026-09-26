"""Draw manuscript figures from the existing builders and saved results.

Reuses the existing figure builders. A local save callback changes appearance
and redirects outputs. Application details read the existing result tables and
residual paths. Numeric artists are compared before and after style adjustment.
Use --additions-only for the neural, MDA and two nonlinear weight comparisons.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/figures"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection, QuadMesh
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd

from experiments import plot_linear_geometry_mechanism as linear
from experiments import plot_sir_diagnostics as candidates

spec = importlib.util.spec_from_file_location(
    "paper_core_figures", ROOT / "experiments/make_paper1_core_figures.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

INK = "#22211F"
MUTED = "#6A6963"
REF = "#8F8E86"
GRID = "#DEDDD6"
PALE = "#D8D7D1"
ACCENT = "#D65E3F"
PAPER = "#F3F2EE"
WIDTH = 16.1 / 2.54
HEAT = LinearSegmentedColormap.from_list(
    "restrained_difference", ["#6F8F9F", "#F7F6F2", "#CC7855"])

NAMES = {
    "paper1_intro_audit_motivation": "01_main_motivation",
    "paper1_finite_grid_geometry": "02_main_geometry",
    "paper1_linear_coordinate_sensitivity": "03_main_sensitivity",
    "paper1_sir_country_diagnostics": "04_main_sir_summary",
    "01_fixed_calendar": "05_fixed_calendar",
    "02_country_energy_patterns": "06_country_energy",
    "03_window_energy_directions": "07_window_energy",
    "paper1_nordic_window_sensitivity": "paper1_nordic_window_sensitivity",
}


def configure() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9,
        "axes.titlesize": 9.5, "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.labelcolor": INK,
        "text.color": INK, "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.fontsize": 8.2, "axes.linewidth": .55,
        "axes.edgecolor": "#B0AFA9", "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "figure.facecolor": "white",
        "savefig.facecolor": "white", "mathtext.fontset": "dejavusans",
    })


def numeric_artists(fig: Figure) -> list[tuple[str, np.ndarray]]:
    """Only values/geometries, excluding colours, fonts and marker sizes."""
    values = []
    for a, ax in enumerate(fig.axes):
        values.append((f"{a}/limits", np.array([ax.get_xlim(), ax.get_ylim()])))
        for i, line in enumerate(ax.lines):
            values.append((f"{a}/line/{i}", np.array(line.get_xydata(), copy=True)))
        for i, collection in enumerate(ax.collections):
            if isinstance(collection, PathCollection):
                values.append((f"{a}/points/{i}", np.array(collection.get_offsets(), copy=True)))
            elif isinstance(collection, LineCollection):
                for j, segment in enumerate(collection.get_segments()):
                    values.append((f"{a}/segment/{i}/{j}", np.array(segment, copy=True)))
            else:
                array = collection.get_array()
                if array is not None:
                    values.append((f"{a}/array/{i}", np.array(array, copy=True)))
        for i, image in enumerate(ax.images):
            values.append((f"{a}/image/{i}", np.array(image.get_array(), copy=True)))
            values.append((f"{a}/image_norm/{i}", np.array([image.norm.vmin, image.norm.vmax])))
        for i, shape in enumerate(ax.patches):
            if isinstance(shape, Ellipse):
                values.append((f"{a}/ellipse/{i}", np.array([
                    *shape.center, shape.width, shape.height, shape.angle])))
            elif isinstance(shape, Rectangle):
                values.append((f"{a}/rectangle/{i}", np.array([
                    shape.get_x(), shape.get_y(), shape.get_width(), shape.get_height()])))
    return values


def common_style(fig: Figure) -> None:
    for text in fig.findobj(Text):
        text.set_fontfamily("Arial")
        size = text.get_fontsize()
        text.set_fontsize(max(8.2, min(size, 9.5)))
        text.set_fontweight("normal")
        box = text.get_bbox_patch()
        if box is not None:
            box.set_facecolor("white")
            box.set_edgecolor("none")
    for ax in fig.axes:
        ax.title.set_fontsize(9.5)
        ax.title.set_fontweight("medium")
        ax._left_title.set_fontsize(9.5)
        ax._left_title.set_fontweight("medium")
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(axis="both", which="major", labelsize=8.3,
                       width=.55, length=3, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color("#B0AFA9")
            spine.set_linewidth(.55)
        for grid in ax.get_xgridlines()+ax.get_ygridlines():
            grid.set_linewidth(.4)
        for line in ax.lines:
            line.set_linewidth(min(line.get_linewidth(), 1.3))
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_sizes(np.minimum(collection.get_sizes(), 45))
        legend = ax.get_legend()
        if legend:
            legend.set_frame_on(False)
            for label in legend.get_texts():
                label.set_fontsize(8.2)
                label.set_color(MUTED)
    if fig._suptitle is not None:
        fig._suptitle.set_fontfamily("Arial")
        fig._suptitle.set_fontsize(11.5)
        fig._suptitle.set_fontweight("medium")
        fig._suptitle.set_color(INK)


def motivation_style(fig: Figure) -> None:
    fig.set_size_inches(WIDTH, 4.8)
    fig._suptitle.set_visible(False)
    a, b, c, d = fig.axes
    a.lines[0].set_color(INK)
    a.lines[1].set_color(REF)
    a.texts[0].set_visible(False)
    handles, labels = a.get_legend_handles_labels()
    a.legend(handles, ["Observed", "Reference mean"], frameon=False, ncol=2,
             loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0,
             fontsize=8.2, handlelength=1.5, columnspacing=1)
    for ax, title in zip(fig.axes, ["(a) Mean evolution", "(b) Marginal variation",
                                  "(c) Energy dependence", "(d) Reference comparison"]):
        ax.set_title(title, loc="left", fontsize=9.5)
    a.set_title(r"(a) Mean evolution ($R^2=0.94$)", loc="left", fontsize=9.5, pad=21)
    b.set_title("(b) Marginal variation", loc="left", fontsize=9.5, pad=21)
    b.set_xlabel("Gaussian quantile")
    b.set_ylabel("Residual quantile")
    c.set_xlabel(r"Previous energy $Z_{k-1}^2$")
    c.set_ylabel(r"Current energy $Z_k^2$")
    b.collections[0].set_facecolor(REF)
    b.collections[0].set_edgecolor(REF)
    b.collections[0].set_sizes([15])
    b.lines[0].set_color(REF)
    c.collections[0].set_edgecolor(REF)
    c.collections[0].set_sizes([17])
    c.collections[1].set_facecolor(ACCENT)
    c.collections[1].set_edgecolor(ACCENT)
    c.collections[1].set_sizes([18])
    c.collections[1].set_alpha(.85)
    c.lines[0].set_color(REF)
    for label in c.texts:
        label.set_color(INK)
    c.legend(frameon=False, loc="lower right", fontsize=8.2)
    for rectangle in d.patches:
        rectangle.set_facecolor(PALE)
        rectangle.set_edgecolor("white")
        rectangle.set_linewidth(.45)
    d.lines[0].set_color(ACCENT)
    d.lines[0].set_linewidth(1.3)
    for label in d.texts:
        label.set_color(INK if "rank" in label.get_text() else MUTED)
        label.set_fontsize(9 if "rank" in label.get_text() else 8.3)
        if label.get_text().startswith("rank "):
            label.set_text(label.get_text().removeprefix("rank "))
        if "one score map" in label.get_text():
            label.set_visible(False)
    fig.set_layout_engine("constrained", h_pad=.055, w_pad=.055,
                          hspace=.055, wspace=.055)


def geometry_style(fig: Figure) -> None:
    fig.set_size_inches(WIDTH, 2.3)
    for ax in fig.axes[:2]:
        for line in ax.lines:
            if line.get_label() == "Instantaneous / L=1":
                line.set_color(INK)
                line.set_linewidth(1.8)
                line.set_zorder(4)
            else:
                line.set_color(GRID)
                line.set_linewidth(.5)
        for shape in ax.patches:
            if isinstance(shape, Ellipse):
                shape.set_edgecolor(REF if "Instantaneous" in shape.get_label() else ACCENT)
                shape.set_linewidth(1.15)
    handles, _ = fig.axes[0].get_legend_handles_labels()
    fig.axes[0].legend(handles, [r"$L=1$", r"$L=10$"], loc="lower left",
                       fontsize=8.2, frameon=False, handlelength=1.7)
    for ax, title in zip(fig.axes, ["(a) Support", "(b) Covariance", "(c) Propagation strength"]):
        ax.set_title(title, loc="left", fontsize=9.2)
    ax = fig.axes[2]
    for line, colour, dash in zip(ax.lines, [REF, INK, ACCENT], [":", "--", "-"]):
        line.set_color(colour)
        line.set_linestyle(dash)
        line.set_linewidth(1.15)
    ax.legend(frameon=False, fontsize=8.2, loc="center right", bbox_to_anchor=(1, .48))
    for grid in ax.get_ygridlines():
        grid.set_color(GRID)
        grid.set_linewidth(.45)
    fig.set_layout_engine("constrained", w_pad=.035, h_pad=.04, wspace=.025)


def sensitivity_style(fig: Figure) -> None:
    fig.set_size_inches(WIDTH, 4.8)
    for ax in fig.axes[:6]:
        ax.images[0].set_cmap(HEAT)
        for label in ax.texts:
            label.set_color(INK)
            label.set_fontsize(8.4)
            label.set_linespacing(1.15)
        ax.tick_params(which="both", length=0)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8.5)
        ax.set_ylabel(ax.get_ylabel(), fontsize=8.5)
    for collection in fig.axes[-1].collections:
        if isinstance(collection, QuadMesh):
            collection.set_cmap(HEAT)
    fig.axes[-1].set_xlabel("Rejection-rate difference (percentage points)", fontsize=8.5)
    fig.axes[-1].tick_params(labelsize=8.2)
    fig.set_layout_engine("constrained", h_pad=.055, w_pad=.065,
                          hspace=.09, wspace=.045)


def sir_style(fig: Figure) -> None:
    fig.set_size_inches(WIDTH, 2.75)
    fig._suptitle.set_visible(False)
    for ax in fig.axes:
        for collection in ax.collections:
            if isinstance(collection, LineCollection):
                collection.set_color("#B0AFA9")
                collection.set_alpha(1)
                collection.set_linewidth(4.2)
            elif isinstance(collection, PathCollection):
                if "held-out" in collection.get_label():
                    count = len(collection.get_offsets())
                    colours = [INK]*count
                    if count >= 2:
                        colours[1] = ACCENT
                    collection.set_facecolor(colours)
                    collection.set_edgecolor("white")
                    collection.set_sizes([38])
                else:
                    collection.set_color(MUTED)
        for label in ax.texts:
            label.set_color(MUTED)
            label.set_fontsize(8.3)
            if label.get_text() in ("deficit", "excess"):
                label.set_visible(False)
        for label in ax.get_xticklabels():
            label.set_color(INK)
    for i, shape in enumerate(fig.axes[1].patches):
        shape.set_facecolor(PAPER if i == 0 else "#FAFAF8")
    # The caption records the different selection rules and reference ranges.
    for label in fig.axes[1].texts:
        text = label.get_text()
        if text == "rules recorded before evaluation":
            label.set_text("Fixed calendar")
        elif text == "descriptive: outcome-filtered pool":
            label.set_text("Descriptive batch")
        if label.get_text() in ("Fixed calendar", "Descriptive batch"):
            label.set_transform(fig.axes[1].get_xaxis_transform())
            label.set_position((label.get_position()[0], 1.02))
            label.set_va("bottom")
            label.set_fontsize(8.2)
    fig.axes[0].set_title("(a) Global scores", loc="left", fontsize=9.5, pad=23)
    fig.axes[1].set_title("(b) Energy across five windows", loc="left", fontsize=9.5, pad=23)
    fig.axes[0].set_ylabel(r"Global score $S_{\mathrm{CE}}$")
    fig.axes[1].set_ylabel(r"Mean energy $K^{-1}\sum z_k^2$")
    for ax in fig.axes:
        ax.get_legend().remove()
    fig.legend(handles=[
        Line2D([], [], marker="D", ls="", ms=5, color=INK, label="Observed"),
        Line2D([], [], color="#B0AFA9", lw=4, label="Reference range"),
        Line2D([], [], marker="_", ls="", ms=9, color=MUTED, label="Median in (b)"),
    ], loc="lower center", bbox_to_anchor=(.53, .005), ncol=3,
        frameon=False, fontsize=8.2, handlelength=1.5, columnspacing=1.5)
    fig.subplots_adjust(left=.085, right=.98, bottom=.225, top=.77, wspace=.43)


def holdout_style(fig: Figure) -> None:
    """Align the six panels and keep annotations close to their features."""
    fig.set_size_inches(WIDTH, 5.5)
    positions = [
        (.10, .73, .385, .18), (.585, .73, .385, .18),
        (.10, .42, .385, .18), (.585, .42, .385, .18),
        (.10, .11, .385, .18), (.585, .11, .385, .18),
    ]
    titles = [
        "(a) CZE: state path", "(b) GRC: state path",
        "(c) CZE: cumulative energy", "(d) GRC: cumulative energy",
        "(e) CZE: cumulative residuals", "(f) GRC: cumulative residuals",
    ]
    for ax, position, title in zip(fig.axes, positions, titles):
        ax.set_position(position)
        ax.set_title(title, loc="left", fontsize=9, pad=9)
        ax.tick_params(labelsize=8, length=2.5)
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(True, color="#F0EFEC", linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
    fig.axes[1].set_ylabel("")
    fig.axes[3].set_ylabel("")
    fig.axes[2].set_ylabel(r"$C_E(n)$")
    fig.axes[4].set_ylabel(r"$C_z(n)$")
    fig.axes[5].set_ylabel("")
    fig.axes[2].set_xlabel("")
    fig.axes[3].set_xlabel("")
    energy = fig.axes[3]
    energy.texts[0].set_text(energy.texts[0].get_text().replace("Mean energy:", "Mean:"))
    energy.texts[0].set_fontsize(8)
    energy.texts[0].set_color(MUTED)
    minimum = energy.texts[1]
    minimum.set_anncoords("axes fraction")
    minimum.set_position((.74, .065))
    minimum.set_ha("right")
    minimum.set_va("bottom")
    minimum.set_fontsize(8)
    minimum.arrow_patch.set_visible(False)
    for ax in fig.axes[4:6]:
        ax.texts[0].set_fontsize(8)
        ax.texts[0].set_color(MUTED)
    for text in fig.texts:
        if "Global" in text.get_text():
            text.set_position((.535, .02))
            text.set_fontsize(8.5)


def country_style(fig: Figure) -> None:
    """Use country columns and feature rows on the existing common scales."""
    fig.set_size_inches(WIDTH, 4.45)
    dates = [text.get_text() for text in fig.texts[3::2]]
    for text in fig.texts:
        text.set_visible(False)
    lefts = (.105, .415, .725)
    countries = ("LTU", "MDA", "SVN")
    for i, (left, country, date) in enumerate(zip(lefts, countries, dates)):
        daily, energy = fig.axes[2*i:2*i+2]
        daily.set_position([left, .545, .245, .245])
        energy.set_position([left, .135, .245, .285])
        colour = ACCENT if country == "MDA" else INK
        fig.text(left, .875, f"({chr(97+i)}) {country}", color=colour,
                 fontsize=10, weight="medium")
        fig.text(left, .832, date, color=MUTED, fontsize=8)
        for ax in (daily, energy):
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_xticks([0, 30, 60])
            ax.tick_params(labelsize=8, length=2.5, labelbottom=ax is energy)
            ax.spines["left"].set_visible(False)
            ax.yaxis.grid(True, color="#F0EFEC", linewidth=.45, zorder=0)
            ax.set_axisbelow(True)
            if i:
                ax.tick_params(labelleft=False)
            for label in ax.texts:
                label.set_visible(False)
        if country == "MDA":
            peak = daily.texts[0]
            peak.set_visible(True)
            peak.set_anncoords("offset points")
            peak.set_position((-3, 7))
            peak.set_fontsize(8)
            peak.arrow_patch.set_visible(False)
    fig.axes[0].set_ylabel(r"Daily energy $z_k^2$", fontsize=9, labelpad=7)
    fig.axes[1].set_ylabel(r"Cumulative energy $C_E$", fontsize=9, labelpad=7)
    fig.text(.535, .032, "Window day", ha="center", color=MUTED, fontsize=8.5)


def nordic_window_style(fig: Figure) -> None:
    """Keep the saved window comparisons in the manuscript's figure style."""
    fig.set_size_inches(WIDTH, 3.3)
    fig.set_layout_engine(None)
    fig._suptitle.set_visible(False)
    left, right = fig.axes
    left.set_position([.11, .22, .36, .65])
    right.set_position([.66, .22, .32, .65])
    left.set_title("(a) Half-window ranks", loc="left", pad=10)
    right.set_title("(b) NOR: parent and halves", loc="left", pad=10)
    left.set_ylabel("Global reference rank")
    left.texts[0].set_text("NOR: 0.444")
    left.texts[0].set_position((.35, .27))
    left.texts[0].set_fontsize(8)
    left.texts[1].set_text("0.05")
    left.texts[1].set_fontsize(8)
    for label, text in zip(left.texts[2:], ("-120 d", "-60 d", "Original", "+60 d")):
        label.set_text(text)
        label.set_fontsize(8)
    for line, points, color in zip(left.lines[:3], left.collections[:3],
                                   (INK, REF, ACCENT)):
        line.set_color(color)
        points.set_color(color)
        points.set_sizes([17])
    for patch in left.patches:
        patch.set_facecolor(PAPER)
    handles, _ = left.get_legend_handles_labels()
    left.legend(handles, ["FIN", "NOR", "SWE"], loc="upper right",
                frameon=False, fontsize=7.8, handletextpad=.4)
    right.set_yticklabels(["Parent (60)", "First (30)", "Second (30)"])
    right.tick_params(axis="y", labelsize=8)
    right.set_xlabel("Mean squared residual")
    right.set_xticks([.5, 1, 1.5, 2], ["0.5", "1", "1.5", "2"])
    right.get_legend().set_visible(False)
    for label in right.texts:
        label.set_fontsize(7.8)
    for label in right.texts[1:6:2]:
        label.set_bbox(dict(facecolor="white", edgecolor="none", pad=.5))
    right.texts[-2].set_text("Global rank")
    right.texts[-1].set_visible(False)


STYLES = {
    "paper1_intro_audit_motivation": motivation_style,
    "paper1_finite_grid_geometry": geometry_style,
    "paper1_linear_coordinate_sensitivity": sensitivity_style,
    "paper1_sir_country_diagnostics": sir_style,
    "01_fixed_calendar": holdout_style,
    "02_country_energy_patterns": country_style,
    "paper1_nordic_window_sensitivity": nordic_window_style,
}


def neural_matching_figure() -> None:
    """Plot saved paired score gaps and rejection probabilities, without refitting."""
    results = pd.read_csv(
        ROOT / "output/neural_training_comparison/combined_results.csv")
    fitted = results.loc[results.training_paths.gt(0)]
    target = float(results.nominal_mc_target.iloc[0])
    metrics = [
        ("cdf_gap", "Score gap"),
        ("conditional_rejection", "Rejection probability"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.1), sharey="col")
    for i, n in enumerate((48, 192)):
        groups = {
            name: fitted.loc[fitted.training_paths.eq(n) & fitted.train_variance.eq(name)]
                        .set_index("replicate").sort_index()
            for name in ("plugin", "tangent")
        }
        assert list(groups["plugin"].index) == list(groups["tangent"].index) == list(range(12))
        offsets = np.linspace(-.10, .10, 12)
        for j, (metric, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            left, right = groups["plugin"][metric], groups["tangent"][metric]
            for offset, y0, y1 in zip(offsets, left, right):
                ax.plot([offset, 1 + offset], [y0, y1], color=PALE,
                        lw=.75, zorder=1)
            for x, name, color in ((0, "plugin", INK), (1, "tangent", ACCENT)):
                group = groups[name]
                values = group[metric].to_numpy()
                ax.errorbar(x + offsets, values,
                            yerr=np.vstack([values - group[metric + "_low"],
                                            group[metric + "_high"] - values]),
                            fmt="none", ecolor=color, alpha=.35, lw=.7, zorder=2)
                ax.scatter(x + offsets, values, s=15, color=color, zorder=3)
                ax.plot([x - .17, x + .17], [np.median(values)] * 2,
                        color=color, lw=2.2, zorder=4)
            if metric == "conditional_rejection":
                ax.axhline(target, color=REF, ls="--", lw=.8, zorder=0)
            upper = np.ceil(fitted[metric + "_high"].max() / .1) * .1
            ax.set(xlim=(-.3, 1.3), ylim=(0, upper), ylabel=ylabel,
                   xticks=[0, 1], xticklabels=["Plug-in", "Tangent"])
            ax.set_title(f"({chr(97 + 2*i + j)}) {n} training paths", loc="left")
            ax.set_xlabel("Training variance")
            ax.grid(axis="y", color=GRID, lw=.4, zorder=0)
    fig.subplots_adjust(left=.10, right=.99, bottom=.11, top=.94,
                        wspace=.33, hspace=.57)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"08_main_neural_matching.{suffix}", dpi=220)
    plt.close(fig)
    print("Drew neural score matching from 24 saved training-data pairs", flush=True)


def mda_dependence_figure() -> None:
    """Show two recorded components of the descriptive MDA window."""
    components = pd.read_csv(
        candidates.data_source.WINDOWS / "MDA/midpoint_p60/component_summary.csv"
    ).set_index("metric")
    metrics = [
        ("bracket_I_mean_z2", "(a) Mean energy", (.5, 1.5), [.5, 1., 1.5]),
        ("bracket_I_energy_acf1", "(b) Lag-one energy correlation",
         (-.3, .55), [-.2, 0, .2, .4]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.85))
    for ax, (metric, title, limits, ticks) in zip(axes, metrics):
        row = components.loc[metric]
        assert int(row.evaluation_n) == 2500
        ax.plot([row.evaluation_q025, row.evaluation_q975], [0, 0],
                color=PALE, lw=5, solid_capstyle="butt")
        ax.plot(row.evaluation_q500, 0, marker="|", ms=12, mew=1, color=INK)
        ax.scatter(row.observed, 0, s=31, color=ACCENT, edgecolor="white", lw=.6, zorder=3)
        ax.annotate(f"{row.observed:.3f}", (row.observed, 0), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=ACCENT)
        ax.set(xlim=limits, ylim=(-.55, 1.1), yticks=[], xticks=ticks)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", pad=7)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=4.5, color=ACCENT, label="Observed"),
        Line2D([], [], marker="|", ls="", ms=9, color=INK, label="Reference median"),
        Line2D([], [], color=PALE, lw=4, label="95% reference interval"),
    ], loc="upper center", bbox_to_anchor=(.5, 1.), ncol=3, frameon=False,
        handlelength=1.4, columnspacing=1.2)
    fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.63, wspace=.43)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"09_main_mda_dependence.{suffix}", dpi=220)
    plt.close(fig)
    print("Drew MDA components from their saved reference quantiles", flush=True)


def nonlinear_weight_figure() -> None:
    """Show saved interval bounds and detection rates from the paired fits."""
    source = ROOT / "experiments/nonlinear_weight_selection"
    settings = json.loads((source / "separation/settings.json").read_text())
    bounds = pd.read_csv(source / "separation/bound_refinement.csv")
    points = pd.read_csv(source / "separation/point_bounds.csv")
    paths = max(settings["samples"])
    interval = bounds.loc[bounds.paths.eq(paths) & bounds.ridge.eq(0)
                          & bounds.partition.eq("split_32")]
    assert len(interval) == 1
    interval = interval.iloc[0]
    lower = points.loc[points.paths.eq(paths) & points.ridge.eq(0), "gap_lower"].max()
    upper = interval.cdf_upper
    rejection_upper = interval.rejection_deviation_upper

    results = pd.read_csv(source / "coupling/results.csv")
    fitted = results.loc[results.kappa.eq(.1)
                         & results.training_paths.eq(settings["training_paths"])]
    groups = {method: fitted.loc[fitted.method.eq(method)].set_index("fit").sort_index()
              for method in ("selected", "level")}
    assert list(groups["selected"].index) == list(groups["level"].index) == list(range(64))
    assert groups["selected"].ridge_over_fitted_kappa_squared.eq(.3).all()
    assert groups["level"].ridge_over_fitted_kappa_squared.eq(0).all()

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    ax = axes[0]
    ax.plot([lower, upper], [1, 1], color=INK, lw=2, marker="|", ms=9, mew=1)
    ax.plot([0, rejection_upper], [0, 0], color=INK, lw=2, marker="|", ms=9, mew=1)
    ax.annotate(f"[{np.floor(lower * 1e5) / 1e5:.5f}, "
                f"{np.ceil(upper * 1e5) / 1e5:.5f}]",
                ((lower + upper) / 2, 1), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=.5))
    ax.annotate(r"$\leq$ " + f"{np.ceil(rejection_upper * 1e5) / 1e5:.5f}",
                (rejection_upper / 2, 0), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(settings["tolerance"], color=REF, ls="--", lw=.8, zorder=0)
    ax.text(settings["tolerance"], -.32, "0.02 limit", ha="center", fontsize=8, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=.5))
    ax.set(xlim=(-.0006, .032), ylim=(-.5, 1.55), xticks=[0, .01, .02, .03],
           yticks=[0, 1], yticklabels=["Rejection error\n$R_0(I)$", "Score gap\n$D_0(I)$"],
           xlabel="Absolute probability difference")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("(a) Error bounds", loc="left", pad=11)
    ax.text(.5, 1.02, r"Full whitening ($\ell=0$)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, color=MUTED)

    ax = axes[1]
    offsets = np.linspace(-.065, .065, 64)
    left = 100 * groups["selected"].weak_matched.to_numpy()
    right = 100 * groups["level"].weak_matched.to_numpy()
    for offset, y0, y1 in zip(offsets, left, right):
        ax.plot([offset, 1 + offset], [y0, y1], color=PALE, lw=.55, alpha=.55, zorder=1)
    for x, values, color in ((0, left, INK), (1, right, ACCENT)):
        ax.scatter(x + offsets, values, s=9, color=color, alpha=.55, zorder=2)
        ax.plot([x - .17, x + .17], [values.mean()] * 2, color=color, lw=2.2, zorder=3)
        ax.text(x, values.max() + 2.0, f"{values.mean():.2f}%", ha="center", fontsize=8.5,
                color=color)
    ax.axhline(100 * settings["alpha"], color=REF, ls="--", lw=.8, zorder=0)
    ax.text(1.2, 5, "5%", ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set(xlim=(-.25, 1.25), ylim=(0, 36), yticks=[0, 10, 20, 30],
           ylabel="Power (%)", xticks=[0, 1],
           xticklabels=["CDF rule\n" + r"$\ell=0.3$", "Level rule\n" + r"$\ell=0$"])
    ax.set_title(r"(b) Power against added $Y$-noise", loc="left", pad=11)
    ax.grid(axis="y", color=GRID, lw=.4, zorder=0)
    fig.subplots_adjust(left=.15, right=.985, bottom=.24, top=.82, wspace=.72)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"10_main_weight_selection.{suffix}", dpi=220,
                    bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"D_0 bounds: [{lower:.8f}, {upper:.8f}]; R_0 upper: {rejection_upper:.8f}; "
          f"mean detection rates: {left.mean():.4f}%, {right.mean():.4f}%", flush=True)


def feedback_weight_figure() -> None:
    """Show the saved selection bounds, not empirical final-test frequencies."""
    source = ROOT / "experiments/feedback_weight_selection/results"
    settings = json.loads((source / "fit00/settings.json").read_text())
    methods = settings["candidate_order"]
    tables = [pd.read_csv(source / f"fit{i:02d}/bounds.csv").set_index("method")
              .loc[methods] for i in range(16)]
    errors = np.stack([table.calibration_bound.to_numpy() for table in tables])
    powers = np.stack([table.power_lower.to_numpy() for table in tables])
    assert np.isfinite(errors).all() and np.isfinite(powers).all()
    passes = (errors <= settings["epsilon"]) & (powers >= settings["required_power"])
    for table, eligible in zip(tables, passes):
        np.testing.assert_array_equal(table.passes.to_numpy(), eligible)
    assert passes[:, 1].all() and not passes[:, 0].any()
    offsets = np.linspace(-.16, .16, len(tables))
    labels = ["Raw", "Average covariance", r"Ridge $\ell=0.01$",
              r"Ridge $\ell=0.03$", r"Ridge $\ell=0.1$"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharey=True)
    for ax, values, cutoff, limits, ticks, title in zip(
        axes, (errors, powers),
        (settings["epsilon"], settings["required_power"]),
        ((0, .024), (0, .21)), ([0, .01, .02], [0, .1, .2]),
        ("(a) Error upper bound", "(b) Power lower bound"),
    ):
        for j in range(len(methods)):
            ax.scatter(values[:, j], j + offsets, s=13,
                       color=ACCENT if j == 1 else INK, alpha=.75, zorder=3)
        ax.axvline(cutoff, color=REF, ls="--", lw=.9, zorder=1)
        ax.set(xlim=limits, ylim=(4.5, -.5), xticks=ticks, yticks=range(5),
               xlabel="Probability")
        ax.set_title(title, loc="left", pad=10)
        ax.grid(axis="x", color=GRID, lw=.4, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels(labels)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(left=.26, right=.98, bottom=.20, top=.85, wspace=.30)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"11_supp_feedback_selection.{suffix}", dpi=220)
    plt.close(fig)
    print("Feedback candidates meeting both bounds:",
          dict(zip(methods, passes.sum(axis=0).tolist())), flush=True)


def main(argv=None) -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additions-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    neural_matching_figure()
    mda_dependence_figure()
    nonlinear_weight_figure()
    feedback_weight_figure()
    if args.additions_only:
        return
    core.BLUE, core.REFERENCE_BLUE = REF, REF
    core.RED, core.GREY, core.MID_GREY, core.LIGHT_GREY = ACCENT, MUTED, REF, PALE
    candidates.ACCENT = ACCENT
    original_save = Figure.savefig
    processed = set()
    checks = []

    def save_manuscript_figure(fig, filename, *args, **kwargs):
        if isinstance(filename, PdfPages):
            return original_save(fig, filename, *args, **kwargs)
        stem = Path(filename).stem
        name = NAMES[stem]
        if name not in processed:
            before = numeric_artists(fig)
            common_style(fig)
            if stem in STYLES:
                STYLES[stem](fig)
            after = numeric_artists(fig)
            assert [k for k, _ in before] == [k for k, _ in after]
            for (key, a), (_, b) in zip(before, after):
                np.testing.assert_array_equal(a, b, err_msg=f"Numeric change: {name}/{key}")
            checks.append({"figure": name, "numeric_arrays_checked": len(before),
                           "numeric_data_unchanged": True})
            processed.add(name)
            print(f"Styled {name}; checked {len(before)} numeric arrays", flush=True)
            booklet.savefig(fig)
        kwargs.update(dpi=220, bbox_inches=None, pad_inches=0)
        return original_save(fig, OUT / f"{name}{Path(filename).suffix}", *args, **kwargs)

    with PdfPages(OUT / "manuscript_figures.pdf", metadata={
        "Title": "Current manuscript figures",
        "Subject": "Motivation, geometry, sensitivity, and application figures",
    }) as booklet, patch.object(Figure, "savefig", save_manuscript_figure):
        core.make_motivation_figure()
        core.make_nordic_window_sensitivity_figure()
        linear.geometry_figure()
        linear.sensitivity_figure()
        records = {c: candidates.data_source.load_paths(c)
                   for c in ("CZE", "GRC", "LTU", "MDA", "SVN")}
        candidates.plot_holdouts(records)
        candidates.plot_country_patterns(records)
        candidates.data_source.OUT.mkdir(parents=True, exist_ok=True)
        windows = candidates.data_source.window_energy_table()
        candidates.plot_windows(windows)
    assert len(processed) == 7
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
