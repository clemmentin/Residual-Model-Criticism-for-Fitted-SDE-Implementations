"""
Discrepancy diagnostics for SIR Brownian-inversion residuals.

Brownian inversion residuals z_t are mapped to probability coordinates
u_t = Phi(z_t).  Under a valid Brownian innovation model, the marginal sequence
should occupy [0, 1] uniformly, and lag pairs (u_t, u_{t+lag}) should occupy
[0, 1]^2 uniformly.  This script measures those occupation errors and calibrates
them with the fitted-null bootstrap.

The 1D "interval_discrepancy" is the Kuiper-style full-interval discrepancy:
    sup over intervals I | empirical_mass(I) - length(I) |.

The 2D "rect_discrepancy" is a grid approximation to:
    sup over rectangles R | empirical_mass(R) - area(R) |.

Outputs:
    cache/summaries/sir_discrepancy_summary.csv
    cache/summaries/sir_discrepancy_draws.csv
    cache/summaries/sir_discrepancy.png
    cache/summaries/sir_discrepancy_<case>.png
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import logging
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

from brownian_inversion import _evaluate_bi_residuals  # noqa: E402
import config as config_module
from experiments.audit_statistics import upper_rank
from experiments.run_sir_bi_bootstrap import (
    CASES,
    _apply_updates,
    _load_cached_data,
    _load_model,
    _simulate_null_residuals,
    _start_position,
)


SUMMARY_DIR = ROOT / "cache" / "summaries"
SUMMARY_OUT = SUMMARY_DIR / "sir_discrepancy_summary.csv"
DRAWS_OUT = SUMMARY_DIR / "sir_discrepancy_draws.csv"
PLOT_OUT = SUMMARY_DIR / "sir_discrepancy.png"

DEFAULT_CASES = [
    "baseline_frzgamma",
    "recov28",
    "recov28_frzgamma",
    "baseline_zwd1_frzgamma",
]

CONDITIONS = ["raw_phi", "center_scale_phi", "shuffle_phi"]
CONDITION_LABELS = {
    "raw_phi": "raw",
    "center_scale_phi": "center + scale",
    "shuffle_phi": "shuffle order",
}
PLOT_METRICS = [
    ("interval_discrepancy_1d", "1D marginal discrepancy", "single residuals"),
    ("lag1_rect_discrepancy_2d", "Lag-1 rectangle discrepancy", r"$(u_t,u_{t-1})$"),
    ("lag7_rect_discrepancy_2d", "Lag-7 rectangle discrepancy", r"$(u_t,u_{t-7})$"),
]
FOCUS_CASE = "recov28_frzgamma"
PLOT_CASES = [
    ("baseline_frzgamma", "baseline"),
    ("recov28_frzgamma", "repaired"),
    ("baseline_zwd1_frzgamma", "marginal regularized"),
]
CASE_NOTES = {
    "baseline_frzgamma": [
        ("center + scale mostly fixes the one-dimensional diagnostic", "#245c2f"),
        ("lag-1 geometry remains above the fitted-null band", "#a32020"),
        ("lag-7 geometry remains above the fitted-null band", "#a32020"),
    ],
    "recov28_frzgamma": [
        ("center + scale fixes the one-dimensional marginal diagnostic", "#245c2f"),
        ("lag-1 geometry remains high until temporal order is shuffled", "#a32020"),
        ("lag-7 geometry shows the same temporal-order failure", "#a32020"),
    ],
    "baseline_zwd1_frzgamma": [
        ("marginal shape is partly repaired", "#245c2f"),
        ("lag-1 geometry is worse, not better", "#a32020"),
        ("weekly-order geometry remains far outside the null band", "#a32020"),
    ],
}


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        raise ValueError("Need at least 10 finite residuals.")
    return arr


def _center_scale(values: np.ndarray) -> np.ndarray:
    arr = _finite(values)
    return (arr - float(np.mean(arr))) / max(float(np.std(arr, ddof=1)), 1e-12)


def _conditioned_residuals(values: np.ndarray, condition: str, seed: int = 0) -> np.ndarray:
    arr = _finite(values)
    if condition == "raw_phi":
        return arr
    if condition == "center_scale_phi":
        return _center_scale(arr)
    if condition == "shuffle_phi":
        rng = np.random.default_rng(seed)
        return _center_scale(rng.permutation(arr))
    raise ValueError(f"Unknown condition: {condition}")


def _uniform_coordinate(values: np.ndarray) -> np.ndarray:
    return np.clip(norm.cdf(_finite(values)), 0.0, 1.0)


def _lag_corr(values: np.ndarray, lag: int) -> float:
    arr = _finite(values)
    if arr.size <= lag:
        return float("nan")
    lhs = arr[:-lag]
    rhs = arr[lag:]
    if np.std(lhs) < 1e-12 or np.std(rhs) < 1e-12:
        return float("nan")
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _one_dim_discrepancy(u_values: np.ndarray) -> dict:
    u = np.sort(_uniform_coordinate(u_values))
    n = int(u.size)
    ranks = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(ranks / n - u))
    d_minus = float(np.max(u - (ranks - 1.0) / n))
    return {
        "star_discrepancy_1d": max(d_plus, d_minus),
        "interval_discrepancy_1d": d_plus + d_minus,
        "d_plus_1d": d_plus,
        "d_minus_1d": d_minus,
        "u_min": float(u[0]),
        "u_max": float(u[-1]),
    }


@lru_cache(maxsize=None)
def _rectangle_specs(grid_size: int) -> tuple[np.ndarray, ...]:
    m = int(grid_size)
    coords = []
    areas = []
    for x0 in range(m):
        for x1 in range(x0 + 1, m + 1):
            width = (x1 - x0) / m
            for y0 in range(m):
                for y1 in range(y0 + 1, m + 1):
                    coords.append((x0, x1, y0, y1))
                    areas.append(width * (y1 - y0) / m)
    coord_arr = np.asarray(coords, dtype=np.int16)
    area_arr = np.asarray(areas, dtype=float)

    anchored = []
    anchored_areas = []
    for x1 in range(1, m + 1):
        for y1 in range(1, m + 1):
            anchored.append((0, x1, 0, y1))
            anchored_areas.append((x1 / m) * (y1 / m))
    anchored_arr = np.asarray(anchored, dtype=np.int16)
    anchored_area_arr = np.asarray(anchored_areas, dtype=float)
    return coord_arr, area_arr, anchored_arr, anchored_area_arr


def _grid_counts(points: np.ndarray, grid_size: int) -> np.ndarray:
    m = int(grid_size)
    bins = np.floor(np.clip(points, 0.0, 1.0 - 1e-15) * m).astype(int)
    counts = np.zeros((m, m), dtype=np.int32)
    np.add.at(counts, (bins[:, 0], bins[:, 1]), 1)
    return counts


def _grid_rect_discrepancy(points: np.ndarray, grid_size: int) -> dict:
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 10:
        raise ValueError("Need at least 10 finite lag pairs.")

    counts = _grid_counts(pts, grid_size)
    prefix = np.pad(counts, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    rects, areas, anchored, anchored_areas = _rectangle_specs(grid_size)

    x0, x1, y0, y1 = rects.T
    rect_counts = prefix[x1, y1] - prefix[x0, y1] - prefix[x1, y0] + prefix[x0, y0]
    rect_diff = rect_counts / pts.shape[0] - areas
    rect_idx = int(np.argmax(np.abs(rect_diff)))

    ax0, ax1, ay0, ay1 = anchored.T
    anchored_counts = prefix[ax1, ay1] - prefix[ax0, ay1] - prefix[ax1, ay0] + prefix[ax0, ay0]
    anchored_diff = anchored_counts / pts.shape[0] - anchored_areas
    anchored_idx = int(np.argmax(np.abs(anchored_diff)))

    worst = rects[rect_idx]
    worst_anchor = anchored[anchored_idx]
    m = float(grid_size)
    return {
        "rect_discrepancy_2d": float(abs(rect_diff[rect_idx])),
        "rect_signed_diff_2d": float(rect_diff[rect_idx]),
        "rect_x0": float(worst[0] / m),
        "rect_x1": float(worst[1] / m),
        "rect_y0": float(worst[2] / m),
        "rect_y1": float(worst[3] / m),
        "star_discrepancy_2d_grid": float(abs(anchored_diff[anchored_idx])),
        "star_signed_diff_2d_grid": float(anchored_diff[anchored_idx]),
        "star_rect_x1": float(worst_anchor[1] / m),
        "star_rect_y1": float(worst_anchor[3] / m),
    }


def _metrics(
    residuals: np.ndarray,
    condition: str,
    lags: list[int],
    grid_size: int,
    seed: int = 0,
) -> dict:
    adjusted = _conditioned_residuals(residuals, condition, seed=seed)
    u = _uniform_coordinate(adjusted)
    row = {
        "condition": condition,
        "n_residuals": int(adjusted.size),
        "z_mean": float(np.mean(adjusted)),
        "z_std": float(np.std(adjusted, ddof=1)),
        **_one_dim_discrepancy(adjusted),
    }
    for lag in lags:
        if u.size <= lag + 2:
            continue
        points = np.column_stack([u[:-lag], u[lag:]])
        lag_metrics = _grid_rect_discrepancy(points, grid_size=grid_size)
        row[f"acf{lag}"] = _lag_corr(adjusted, lag)
        row[f"n_pairs_lag{lag}"] = int(points.shape[0])
        for key, value in lag_metrics.items():
            row[f"lag{lag}_{key}"] = value
    return row


def _upper_pvalue(observed: float, null_values: np.ndarray) -> float:
    vals = np.asarray(null_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not np.isfinite(observed) or vals.size == 0:
        return float("nan")
    return upper_rank(observed, vals)


def _summary_rows_for_metric(
    case: str,
    experiment: str,
    start_date: str,
    n_bootstrap: int,
    observed: dict,
    null_df: pd.DataFrame,
    metric: str,
    lag: int | None,
) -> dict:
    null_values = null_df[metric].to_numpy(dtype=float)
    observed_value = float(observed[metric])
    return {
        "case": case,
        "experiment": experiment,
        "start_date": start_date,
        "condition": observed["condition"],
        "metric": metric,
        "lag": lag if lag is not None else "",
        "n_residuals": int(observed["n_residuals"]),
        "n_bootstrap": int(n_bootstrap),
        "observed": observed_value,
        "null_mean": float(np.mean(null_values)),
        "null_std": float(np.std(null_values, ddof=1)),
        "null_q025": float(np.quantile(null_values, 0.025)),
        "null_q500": float(np.quantile(null_values, 0.500)),
        "null_q975": float(np.quantile(null_values, 0.975)),
        "empirical_percentile": float(np.mean(null_values <= observed_value)),
        "empirical_pvalue_upper": _upper_pvalue(observed_value, null_values),
        "observed_acf_lag": (
            float(observed.get(f"acf{lag}", np.nan)) if lag is not None else float("nan")
        ),
    }


def _run_case(
    case: str,
    lags: list[int],
    n_bootstrap: int,
    seed: int,
    grid_size: int,
) -> tuple[list[dict], list[dict]]:
    cfg = _apply_updates(CASES[case])
    data = _load_cached_data(cfg)
    model, model_path = _load_model(cfg, data)
    start_pos = _start_position(data, cfg)
    start_date = str(data.val_features_df.index[start_pos].date())
    experiment = config_module.get_experiment_tag(cfg)

    observed_residuals = _evaluate_bi_residuals(model, start_date, data, cfg)
    if observed_residuals is None:
        raise RuntimeError(f"No observed residuals for {case}.")

    null_residuals, clip_fraction = _simulate_null_residuals(
        model,
        data,
        cfg,
        start_pos=start_pos,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    summary_rows: list[dict] = []
    draw_rows: list[dict] = []
    for condition in CONDITIONS:
        observed = _metrics(
            observed_residuals,
            condition=condition,
            lags=lags,
            grid_size=grid_size,
            seed=seed,
        )

        condition_draw_rows = []
        for bootstrap_id, residuals in enumerate(null_residuals):
            metrics = _metrics(
                residuals,
                condition=condition,
                lags=lags,
                grid_size=grid_size,
                seed=seed + bootstrap_id + 10_000,
            )
            row = {
                "case": case,
                "experiment": experiment,
                "start_date": start_date,
                "model_path": str(model_path.resolve().relative_to(ROOT)),
                "bootstrap_id": int(bootstrap_id),
                "clip_fraction": float(clip_fraction[bootstrap_id]),
                **metrics,
            }
            draw_rows.append(row)
            condition_draw_rows.append(row)

        null_df = pd.DataFrame(condition_draw_rows)
        for metric in ["star_discrepancy_1d", "interval_discrepancy_1d"]:
            summary_rows.append(
                _summary_rows_for_metric(
                    case,
                    experiment,
                    start_date,
                    n_bootstrap,
                    observed,
                    null_df,
                    metric,
                    lag=None,
                )
            )
        for lag in lags:
            for metric in [
                f"lag{lag}_star_discrepancy_2d_grid",
                f"lag{lag}_rect_discrepancy_2d",
            ]:
                summary_rows.append(
                    _summary_rows_for_metric(
                        case,
                        experiment,
                        start_date,
                        n_bootstrap,
                        observed,
                        null_df,
                        metric,
                        lag=lag,
                    )
                )

    return summary_rows, draw_rows


def _format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "p=NA"
    if value < 0.01:
        return "p<.01"
    return f"p={value:.2f}"


def _case_plot_path(case: str) -> Path:
    return SUMMARY_DIR / f"sir_discrepancy_{case}.png"


def _plot_single_case(summary: pd.DataFrame, case: str, case_label: str, out_path: Path) -> None:
    available = [item for item in PLOT_METRICS if item[0] in set(summary["metric"])]
    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(5.8 * len(available), 4.6),
        squeeze=False,
    )
    fig.suptitle(
        f"Probability-coordinate discrepancy audit: {case} ({case_label})",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    for col_idx, (metric, title, subtitle) in enumerate(available):
        ax = axes[0][col_idx]
        subset = summary[(summary["case"] == case) & (summary["metric"] == metric)].copy()
        subset["condition"] = pd.Categorical(subset["condition"], CONDITIONS, ordered=True)
        subset = subset.sort_values("condition")
        x = np.arange(len(subset))

        q025 = subset["null_q025"].to_numpy(dtype=float)
        q500 = subset["null_q500"].to_numpy(dtype=float)
        q975 = subset["null_q975"].to_numpy(dtype=float)
        observed = subset["observed"].to_numpy(dtype=float)
        pvalues = subset["empirical_pvalue_upper"].to_numpy(dtype=float)

        ax.fill_between(x, q025, q975, color="#c7ccd8", alpha=0.55, label="fitted-null 95% band")
        ax.plot(x, q500, color="#e68632", marker="x", lw=1.8, ls="--", label="fitted-null median")
        ax.plot(x, observed, color="#1f77b4", marker="o", lw=2.4, label="observed")

        for xi, yi, pv, upper in zip(x, observed, pvalues, q975):
            color = "#a32020" if yi > upper else "#245c2f"
            ax.annotate(
                _format_pvalue(pv),
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color=color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS.get(str(value), str(value)) for value in subset["condition"]],
            rotation=18,
            ha="right",
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
        ax.set_ylabel("discrepancy" if col_idx == 0 else "")
        ymax = max(float(np.nanmax(q975)), float(np.nanmax(observed))) * 1.25
        ax.set_ylim(0.0, ymax)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        note, color = CASE_NOTES.get(case, CASE_NOTES[FOCUS_CASE])[col_idx]
        ax.text(0.02, -0.33, note, transform=ax.transAxes, fontsize=9, color=color)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    present_cases = set(summary["case"])
    cases = [(case, label) for case, label in PLOT_CASES if case in present_cases]
    if not cases:
        fallback = FOCUS_CASE if FOCUS_CASE in present_cases else str(summary["case"].iloc[0])
        cases = [(fallback, fallback)]
    for case, label in cases:
        _plot_single_case(summary, case, label, _case_plot_path(case))
    available = [item for item in PLOT_METRICS if item[0] in set(summary["metric"])]
    fig, axes = plt.subplots(
        len(cases),
        len(available),
        figsize=(5.5 * len(available), 3.25 * len(cases)),
        squeeze=False,
        sharex=False,
    )
    fig.suptitle(
        "Probability-coordinate discrepancy audit",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    for row_idx, (case, case_label) in enumerate(cases):
        for col_idx, (metric, title, subtitle) in enumerate(available):
            ax = axes[row_idx][col_idx]
            subset = summary[(summary["case"] == case) & (summary["metric"] == metric)].copy()
            subset["condition"] = pd.Categorical(subset["condition"], CONDITIONS, ordered=True)
            subset = subset.sort_values("condition")
            x = np.arange(len(subset))

            q025 = subset["null_q025"].to_numpy(dtype=float)
            q500 = subset["null_q500"].to_numpy(dtype=float)
            q975 = subset["null_q975"].to_numpy(dtype=float)
            observed = subset["observed"].to_numpy(dtype=float)
            pvalues = subset["empirical_pvalue_upper"].to_numpy(dtype=float)

            ax.fill_between(x, q025, q975, color="#c7ccd8", alpha=0.55, label="fitted-null 95% band")
            ax.plot(x, q500, color="#e68632", marker="x", lw=1.6, ls="--", label="fitted-null median")
            ax.plot(x, observed, color="#1f77b4", marker="o", lw=2.2, label="observed")

            for xi, yi, pv, upper in zip(x, observed, pvalues, q975):
                color = "#a32020" if yi > upper else "#245c2f"
                ax.annotate(
                    _format_pvalue(pv),
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                    color=color,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [CONDITION_LABELS.get(str(value), str(value)) for value in subset["condition"]],
                rotation=18,
                ha="right",
                fontsize=9,
            )
            if row_idx == 0:
                ax.set_title(f"{title}\n{subtitle}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{case_label}\ndiscrepancy", fontsize=10)
            ymax = max(float(np.nanmax(q975)), float(np.nanmax(observed))) * 1.22
            ax.set_ylim(0.0, ymax)
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.075,
        "Red p-values mark observed discrepancies above the fitted-null range; green p-values are inside the fitted-null range.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        choices=sorted(CASES),
        help=f"Configurations to audit. Default: {', '.join(DEFAULT_CASES)}",
    )
    parser.add_argument("--lags", default="1,7", help="Comma-separated lag list.")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--grid-size", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2.")
    if args.grid_size < 8:
        raise ValueError("--grid-size should be at least 8.")
    logging.getLogger().setLevel(logging.WARNING)

    cases = args.cases or DEFAULT_CASES
    lags = [int(part.strip()) for part in args.lags.split(",") if part.strip()]
    all_summary_rows: list[dict] = []
    all_draw_rows: list[dict] = []
    for offset, case in enumerate(cases):
        print(f"Running discrepancy diagnostic: {case} ({args.n_bootstrap} null paths)")
        summary_rows, draw_rows = _run_case(
            case,
            lags=lags,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + offset,
            grid_size=args.grid_size,
        )
        all_summary_rows.extend(summary_rows)
        all_draw_rows.extend(draw_rows)

    summary = pd.DataFrame(all_summary_rows)
    draws = pd.DataFrame(all_draw_rows)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False, float_format="%.8g")
    draws.to_csv(DRAWS_OUT, index=False, float_format="%.8g")
    _plot(summary, PLOT_OUT)

    display_metrics = [
        "interval_discrepancy_1d",
        "lag1_rect_discrepancy_2d",
        "lag7_rect_discrepancy_2d",
    ]
    display = summary[summary["metric"].isin(display_metrics)].copy()
    cols = [
        "case",
        "condition",
        "metric",
        "observed",
        "null_q500",
        "null_q975",
        "empirical_pvalue_upper",
        "observed_acf_lag",
    ]
    print(display[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"Saved summary: {SUMMARY_OUT}")
    print(f"Saved draws:   {DRAWS_OUT}")
    print(f"Saved plot:    {PLOT_OUT}")


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
