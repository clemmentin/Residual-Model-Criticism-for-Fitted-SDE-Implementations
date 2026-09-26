"""Draw the three SIR application figures at the manuscript's text width.

Read the existing residual paths and component summaries; no model is refitted
and no reference bank is regenerated.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from experiments import sir_figure_data as data_source

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

OUT = ROOT / "output/figures"
INK = "#22211F"
MUTED = "#6A6963"
REF = "#8F8E86"
GRID = "#DEDDD6"
BAND = "#ECEBE7"
PAPER = "#F0EFEB"
ACCENT = "#D65E3F"
WIDTH = 16.1 / 2.54


def configure() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B0AFA9", "axes.linewidth": .55,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.width": .55, "ytick.major.width": .55,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def path_legend(fig, y=.99, *, sample_paths=False) -> None:
    handles = [
        Line2D([], [], color=INK, lw=1.4, label="Observed"),
        Line2D([], [], color=REF, lw=.85, ls="--", label="Reference median"),
        Patch(facecolor=BAND, edgecolor="none", label="90% pointwise range"),
    ]
    if sample_paths:
        handles.insert(1, Line2D([], [], color="#A7A69F", lw=.7,
                                label="6 reference paths"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, y),
        ncol=len(handles),
        frameon=False, fontsize=8.2, handlelength=1.6, columnspacing=1.1)


def save(fig, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=220)
    plt.close(fig)
    print(f"Drew {stem}", flush=True)


def line_panel(ax, observed, reference, *, cumulative=True, color=INK, width=1.3) -> None:
    q = np.quantile(reference, [.05, .5, .95], axis=0)
    x = np.arange(1, 61)
    if cumulative:
        x = np.r_[0, x]
        observed = np.r_[0., observed]
        q = np.column_stack([np.zeros(3), q])
    ax.fill_between(x, q[0], q[2], facecolor=BAND, linewidth=0, zorder=1)
    ax.axhline(0, color=GRID, lw=.6, zorder=2)
    ax.plot(x, q[1], color=REF, lw=.85, ls=(0, (3, 3)), zorder=3)
    ax.plot(x, observed, color=color, lw=width, zorder=4)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_axisbelow(True)


def daily_stems(ax, z, reference, *, square=False, color=INK) -> None:
    values = z*z if square else z
    refs = reference*reference if square else reference
    q = np.quantile(refs, [.05, .5, .95], axis=0)
    days = np.arange(1, 61)
    ax.fill_between(days, q[0], q[2], color=BAND, linewidth=0, zorder=1)
    ax.plot(days, q[1], color=REF, lw=.7, ls=(0, (3, 3)), zorder=2)
    ax.axhline(0, color=GRID, lw=.65, zorder=2)
    ax.vlines(days, 0, values, color=color, linewidth=.65, zorder=3)
    ax.scatter(days, values, color=color, s=5, linewidths=0, zorder=4)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_axisbelow(True)


def plot_holdouts(records: dict) -> None:
    grc, cze = records["GRC"], records["CZE"]
    g, rg = grc["observed"], grc["reference"]
    c, rc = cze["observed"], cze["reference"]
    cg = np.cumsum(g*g-1)/np.sqrt(60)
    fig = plt.figure(figsize=(WIDTH, 5.95))
    path_legend(fig, sample_paths=True)
    columns = [.105, .595]
    for letter, country, left in zip("ab", ("CZE", "GRC"), columns):
        item = records[country]
        ax = fig.add_axes([left, .745, .38, .15])
        dates = item["state_dates"]
        q = np.quantile(item["reference_log_I"], [.05, .5, .95], axis=0)
        ax.fill_between(dates, q[0], q[2], color=BAND, alpha=.55, linewidth=0, zorder=1)
        for path in item["reference_log_I"][:6]:
            ax.plot(dates, path, color="#A7A69F", lw=.7, alpha=.8, zorder=2)
        ax.plot(dates, q[1], color=REF, lw=.85, ls=(0, (3, 3)), zorder=3)
        ax.plot(dates, item["observed_log_I"], color=ACCENT if country == "GRC" else INK,
                lw=1.4, zorder=4)
        ax.set_xlim(dates[0], dates[-1])
        ax.set_xticks(pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-02"]),
                      ["1 Jan", "1 Feb", "2 Mar"])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_ylabel(r"$\log I_t$")
        ax.set_title(f"({letter}) {country}: reconstructed " + r"$\log I_t$",
                     loc="left", fontsize=9.5, pad=8)

    state_limits = [ax.get_ylim() for ax in fig.axes]
    for ax in fig.axes:
        ax.set_ylim(min(lo for lo, _ in state_limits),
                    max(hi for _, hi in state_limits))

    comparison_ax = fig.add_axes([columns[0], .42, .38, .235])
    energy_ax = fig.add_axes([columns[1], .42, .38, .235])
    cze_centring_ax = fig.add_axes([columns[0], .13, .38, .16])
    centring_ax = fig.add_axes([columns[1], .13, .38, .16])

    line_panel(energy_ax, cg, np.cumsum(rg*rg-1, axis=1)/np.sqrt(60),
               color=ACCENT, width=1.75)
    energy_ax.set_ylim(-7.05, 3.1)
    energy_ax.set_yticks([-6, -4, -2, 0, 2])
    energy_ax.set_title("(d) GRC: energy deficit", loc="left", fontsize=9.5, pad=8)
    energy_ax.set_ylabel("Cumulative energy")
    energy_ax.set_xlabel("Holdout day")
    ratio = grc["metrics"][data_source.METRIC] / grc["reference_energy"]
    energy_ax.text(.035, .94, f"Mean energy: {ratio:.0%} of ref. median",
                   transform=energy_ax.transAxes, va="top", fontsize=8.4)
    minimum_day = int(np.argmin(cg))+1
    energy_ax.scatter(minimum_day, cg[minimum_day-1], s=24, color=ACCENT, zorder=5)
    energy_ax.annotate(
        f"Day {minimum_day}: {cg[minimum_day-1]:.2f}",
        xy=(minimum_day, cg[minimum_day-1]), xytext=(7, -6.3),
        fontsize=8.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=.65),
    )
    line_panel(comparison_ax, np.cumsum(c*c-1)/np.sqrt(60),
               np.cumsum(rc*rc-1, axis=1)/np.sqrt(60))
    comparison_ax.set_ylim(-7.05, 3.1)
    comparison_ax.set_yticks([-6, -4, -2, 0, 2])
    comparison_ax.set_title("(c) CZE: energy", loc="left", fontsize=9.5, pad=8)
    comparison_ax.set_ylabel("Cumulative energy")
    comparison_ax.set_xlabel("Holdout day")

    for ax, letter, country in [(cze_centring_ax, "e", "CZE"),
                                 (centring_ax, "f", "GRC")]:
        item = records[country]
        z, ref = item["observed"], item["reference"]
        line_panel(ax, np.cumsum(z)/np.sqrt(60),
                   np.cumsum(ref, axis=1)/np.sqrt(60),
                   color=ACCENT if country == "GRC" else INK)
        ax.set_ylim(-2.1, 2.1)
        ax.set_yticks([-2, 0, 2])
        ax.set_title(f"({letter}) {country}: cumulative residuals",
                     loc="left", fontsize=9.5, pad=8)
        rank = item["score"]["martingale_rank_pvalue"]
        ax.text(.04, .94, f"Centring rank: {rank:.3f}", transform=ax.transAxes,
                va="top", fontsize=8.3)
        ax.set_ylabel(r"$C_z(n)$")
        ax.set_xlabel("Holdout day")

    fig.text(.54, .025, r"Global $S_{\mathrm{CE}}$: CZE $p=0.0612$; GRC $p=1/2501$",
             fontsize=9, va="center", ha="center")
    save(fig, "01_fixed_calendar")


def plot_country_patterns(records: dict) -> None:
    fig = plt.figure(figsize=(WIDTH, 4.35))
    path_legend(fig)
    fig.text(.245, .875, r"Daily squared residual $z_k^2$", fontsize=9.5)
    fig.text(.655, .875, "Cumulative energy", fontsize=9.5)
    for i, (country, bottom) in enumerate(zip(("LTU", "MDA", "SVN"), (.655, .385, .115))):
        item = records[country]
        z, ref = item["observed"], item["reference"]
        colour = ACCENT if country == "MDA" else INK
        d_ax = fig.add_axes([.245, bottom, .285, .175])
        c_ax = fig.add_axes([.655, bottom, .32, .175])
        daily_stems(d_ax, z, ref, square=True, color=INK)
        d_ax.set_ylim(-1.5, 48)
        d_ax.set_yticks([0, 20, 40])
        cum = np.cumsum(z*z-1)/np.sqrt(60)
        line_panel(c_ax, cum, np.cumsum(ref*ref-1, axis=1)/np.sqrt(60),
                   color=colour, width=1.5 if country == "MDA" else 1.1)
        c_ax.set_ylim(-7.2, 28)
        c_ax.set_yticks([-5, 0, 10, 20])
        fig.text(.018, bottom+.105, f"({chr(97+i)}) {country}", fontsize=9.5,
                 fontweight="medium", color=colour)
        start = pd.Timestamp(item["start"])
        fig.text(.018, bottom+.048, f"{start:%d %b %Y}", fontsize=8.3, color=MUTED)
        if i < 2:
            d_ax.tick_params(labelbottom=False)
            c_ax.tick_params(labelbottom=False)
        else:
            d_ax.set_xlabel("Window day")
            c_ax.set_xlabel("Window day")
        if country == "MDA":
            peak = int(np.argmax(z*z))+1
            d_ax.annotate(f"{z[peak-1]**2:.1f}", (peak, z[peak-1]**2),
                         xytext=(-12, 8), textcoords="offset points", ha="center", fontsize=8.5,
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=.55))
            c_ax.annotate("Builds after day 30", xy=(40, cum[39]), xytext=(2, 24),
                          fontsize=8.4, color=INK,
                          arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6))
    save(fig, "02_country_energy_patterns")


def plot_windows(table: pd.DataFrame) -> None:
    labels = [
        ("calendar_2021_06_01", "1 Jun 2021"),
        ("midpoint_m120", "Original - 120 days"),
        ("midpoint_m60", "Original - 60 days"),
        ("calendar_2022_06_01", "1 Jun 2022"),
        ("published_midpoint", "Original window"),
        ("midpoint_p60", "Original + 60 days"),
        ("calendar_2023_06_01", "1 Jun 2023"),
    ]
    fig = plt.figure(figsize=(WIDTH, 5.05))
    bottom, height = .415, .47
    row_y = lambda pos: bottom+height*(1-(pos+.5)/7)
    for i, (_, label) in enumerate(labels):
        fig.text(.015, row_y(i), label, fontsize=8.5,
                 fontweight="medium" if i == 4 else "normal", va="center")
    for i, (x, country) in enumerate(zip((.31, .545, .78), ("LTU", "MDA", "SVN"))):
        ax = fig.add_axes([x, bottom, .195, height])
        ax.set_xscale("log")
        ax.set_xlim(.025, 7)
        ax.set_ylim(6.5, -.5)
        ax.axvline(1, color=REF, lw=.8, ls=(0, (3, 3)), zorder=1)
        ax.axhspan(3.5, 4.5, color=PAPER, linewidth=0, zorder=0)
        rows = table.loc[table.country.eq(country)].set_index("window")
        colour = ACCENT if country == "MDA" else INK
        for pos, (name, _) in enumerate(labels):
            row = rows.loc[name]
            ax.plot([row.reference_low, row.reference_high], [pos, pos],
                    color="#B0AFA9", lw=2.6, solid_capstyle="butt", zorder=2)
            original = name == "published_midpoint"
            ax.scatter(row.energy_ratio, pos, marker="D" if original else "o",
                       s=29 if original else 19, facecolor=colour if original else "white",
                       edgecolor=colour, linewidth=.9, zorder=4)
            if country == "MDA" and name in ("published_midpoint", "midpoint_p60"):
                ax.annotate(f"{row.energy_ratio:.2f}", (row.energy_ratio, pos), xytext=(4, 5),
                            textcoords="offset points", va="bottom", fontsize=8.5, color=INK, zorder=5)
        ax.set_title(f"({chr(97+i)}) {country}", loc="left", fontsize=9.5, pad=8)
        ax.set_yticks([])
        ax.set_xticks([.05, .25, 1, 4], ["0.05", "0.25", "1", "4"])
        ax.minorticks_off()
        ax.spines["left"].set_visible(False)
    fig.text(.64, .338, "Mean energy / reference median (log scale)",
             ha="center", fontsize=9)
    fig.legend(handles=[
        Line2D([], [], marker="D", ls="", ms=5, color=INK, label="Original"),
        Line2D([], [], marker="o", ls="", ms=4, mfc="white", color=INK, label="Additional"),
        Line2D([], [], color="#B0AFA9", lw=3, label="95% reference interval"),
    ], loc="upper center", bbox_to_anchor=(.51, 1), ncol=3,
        frameon=False, fontsize=8.2, handlelength=1.6, columnspacing=1.2)

    # Show two saved components from the same window in their original units.
    components = pd.read_csv(data_source.WINDOWS / "MDA/midpoint_p60/component_summary.csv").set_index("metric")
    example = table.loc[table.country.eq("MDA") & table.window.eq("midpoint_p60")].iloc[0]
    energy = components.loc[data_source.METRIC]
    np.testing.assert_allclose(energy.observed / energy.evaluation_q500, example.energy_ratio)
    dependence = components.loc["bracket_I_energy_acf1"]
    assert energy.evaluation_q025 < energy.observed < energy.evaluation_q975
    assert dependence.observed > dependence.evaluation_q975
    assert example.dominant_metric == "bracket_I_energy_acf1"
    fig.text(.015, .266, "MDA, original + 60 days (8 Aug 2022)", fontsize=9.3, fontweight="medium")
    fig.text(.975, .266, f"Global reference rank = {example.global_reference_rank:.4f}",
             ha="right", fontsize=8.5)
    for x, metric, title, limits, ticks in [
        (.105, data_source.METRIC, "(d) Mean energy", (.5, 1.5), [.5, 1, 1.5]),
        (.64, "bracket_I_energy_acf1", "(e) Lag-one energy correlation", (-.3, .55), [-.2, 0, .2, .4]),
    ]:
        ax = fig.add_axes([x, .09, .335, .095])
        row = components.loc[metric]
        ax.plot([row.evaluation_q025, row.evaluation_q975], [0, 0],
                color="#B0AFA9", lw=5, solid_capstyle="butt")
        ax.plot(row.evaluation_q500, 0, marker="|", ms=12, mew=1, color=INK)
        ax.scatter(row.observed, 0, s=31, color=ACCENT, edgecolor="white", lw=.6, zorder=3)
        ax.annotate(f"{row.observed:.3f}", (row.observed, 0), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=ACCENT)
        ax.set(xlim=limits, ylim=(-.5, 1.15), yticks=[], xticks=ticks)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=9, pad=7)
    save(fig, "03_window_energy_directions")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    records = {c: data_source.load_paths(c) for c in ("CZE", "GRC", "LTU", "MDA", "SVN")}
    # The existing loader checks the displayed transformations against saved metrics.
    data_source.OUT.mkdir(parents=True, exist_ok=True)
    windows = data_source.window_energy_table()
    plot_holdouts(records)
    plot_country_patterns(records)
    plot_windows(windows)


if __name__ == "__main__":
    main()
