"""Rebuild the supplementary ablation tables from six retained trial CSVs."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {1: "weak", 2: "mid", 3: "nonlinear"}
METHODS = {"A": "A_global_mc", "B": "B_tangent_theory", "C": "C_tangent_mc"}
ALPHA = 0.05
BOOTSTRAP_REPETITIONS = 20_000
SEED = 2026082803


def read_trials(input_dir: Path, tag: str) -> tuple[pd.DataFrame, dict]:
    stem = input_dir / f"coord_ablation_{tag}"
    trials = pd.read_csv(stem.with_name(stem.name + "_trials.csv"))
    metadata = json.loads(stem.with_name(stem.name + "_metadata.json").read_text(encoding="utf-8"))
    n = len(trials)
    if (not n or n != metadata["replicates"] or n != metadata["completed_replicates"]
            or not np.array_equal(trials["replicate"].to_numpy(), np.arange(n))):
        raise ValueError(f"{tag}: expected one complete row per replicate in recorded order.")
    columns = [f"p_{scenario}_{method}_full"
               for scenario in ("null", "targeted", "uniform") for method in METHODS.values()]
    pvalues = trials[columns].to_numpy(dtype=float)
    if not np.isfinite(pvalues).all() or np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError(f"{tag}: p-values must be finite and between zero and one.")
    if "valid" in trials and not trials["valid"].eq(1).all():
        raise ValueError(f"{tag}: invalid trials cannot be included in rejection rates.")
    return trials, metadata


def paired_comparison(groups: list[pd.DataFrame], scenario: str,
                      rng: np.random.Generator) -> dict:
    pairs = [(group[f"p_{scenario}_{METHODS['A']}_full"].to_numpy() <= ALPHA,
              group[f"p_{scenario}_{METHODS['C']}_full"].to_numpy() <= ALPHA)
             for group in groups]
    if len({len(a) for a, c in pairs}) != 1:
        raise ValueError("The two strengths must contain equal numbers of refits.")
    a = np.concatenate([a for a, c in pairs])
    c = np.concatenate([c for a, c in pairs])
    left_only, right_only = int(np.sum(a & ~c)), int(np.sum(c & ~a))
    difference = c.astype(float) - a.astype(float)
    resampled_difference = np.zeros(BOOTSTRAP_REPETITIONS)
    for left, right in pairs:
        # Resampling the three paired outcomes is the original stratified
        # bootstrap; it preserves the dependence between A and C decisions.
        counts = np.array([np.sum(left & ~right), np.sum(right & ~left), np.sum(left == right)])
        resampled = rng.multinomial(len(left), counts / len(left), size=BOOTSTRAP_REPETITIONS)
        resampled_difference += resampled[:, 1] - resampled[:, 0]
    interval = np.quantile(resampled_difference / len(a), [0.025, 0.975])
    discordant = left_only + right_only
    return {
        "scenario": scenario, "left": "A", "right": "C", "n": len(a),
        "reject_left": float(a.mean()), "reject_right": float(c.mean()),
        "left_only": left_only, "right_only": right_only, "both": int(np.sum(a & c)),
        "difference": float(difference.mean()),
        "paired_mc_se": float(difference.std(ddof=1) / np.sqrt(len(a))),
        "stratified_paired_bootstrap_ci95": interval.tolist(),
        "mcnemar_two_sided_p": float(binomtest(right_only, discordant, 0.5).pvalue)
        if discordant else 1.0,
    }


def discrete_ks(pvalues: np.ndarray, denominator: int, rng: np.random.Generator) -> dict:
    """Retain the original null-grid checks and their place in the random stream."""
    ranks = np.rint(pvalues * denominator).astype(int)
    if np.any((ranks < 1) | (ranks > denominator)) or not np.allclose(
            pvalues, ranks / denominator, rtol=0, atol=1e-7):
        raise ValueError("A/C p-values must lie on the retained finite rank grid.")
    counts = np.bincount(ranks, minlength=denominator + 1)[1:]
    n = len(pvalues)
    grid = np.arange(1, denominator + 1) / denominator
    observed = float(np.max(np.abs(np.cumsum(counts) / n - grid)))
    samples = rng.multinomial(n, np.full(denominator, 1 / denominator),
                              size=BOOTSTRAP_REPETITIONS)
    distances = np.max(np.abs(samples.cumsum(axis=1) / n - grid), axis=1)
    return {"n": n, "grid_points": denominator, "ks_distance": observed,
            "simulations": BOOTSTRAP_REPETITIONS,
            "monte_carlo_p": float((1 + np.sum(distances >= observed - 1e-12))
                                   / (BOOTSTRAP_REPETITIONS + 1))}


def add_holm(rows: list[dict], source: str, destination: str) -> None:
    for row, adjusted in zip(rows, multipletests([row[source] for row in rows], method="holm")[1]):
        row[destination] = float(adjusted)


def summarize(input_dir: Path) -> dict:
    rng = np.random.default_rng(SEED)
    strengths, power, null, grid_checks = [], [], [], []
    for number, condition in CONDITIONS.items():
        groups, settings = [], []
        for strength in ("lo", "hi"):
            tag = f"cond{number}_{strength}"
            group, metadata = read_trials(input_dir, tag)
            groups.append(group)
            settings.append(metadata)
            row = {"condition": condition, "strength": strength, "n": len(group),
                   "seed": metadata["seed"], "extra_sigma1": metadata["extra_sigma1"],
                   "scale_factor": metadata["scale_factor"]}
            for scenario in ("targeted", "uniform"):
                for label in ("A", "C"):
                    row[f"{scenario}_{label}"] = float(
                        (group[f"p_{scenario}_{METHODS[label]}_full"] <= ALPHA).mean())
            strengths.append(row)
        if settings[0]["evaluation_paths"] != settings[1]["evaluation_paths"]:
            raise ValueError(f"{condition}: the two strengths use different rank grids.")
        denominator = int(settings[0]["evaluation_paths"]) + 1
        for scenario in ("targeted", "uniform"):
            power.append({"condition": condition, **paired_comparison(groups, scenario, rng)})
        for label, method in METHODS.items():
            pvalues = np.concatenate([group[f"p_null_{method}_full"].to_numpy() for group in groups])
            k, n = int(np.sum(pvalues <= ALPHA)), len(pvalues)
            target = ALPHA if label == "B" else np.floor(ALPHA * denominator) / denominator
            interval = binomtest(k, n).proportion_ci(method="wilson")
            null.append({"condition": condition, "method": label, "rejects": k, "n": n,
                         "rate": k / n, "wilson_ci95": [float(interval.low), float(interval.high)],
                         "null_probability": float(target),
                         "two_sided_binomial_p": float(binomtest(k, n, target).pvalue)})
            if label != "B":
                # These two checks occurred between each setting's bootstrap
                # calculations. Keep that order to reproduce the published CIs.
                grid_checks.append({"condition": condition, "method": label,
                                    **discrete_ks(pvalues, denominator, rng)})
    add_holm(power, "mcnemar_two_sided_p", "holm_p_6_pooled_full_comparisons")
    add_holm(null, "two_sided_binomial_p", "holm_p_9_pooled_null_checks")
    add_holm(grid_checks, "monte_carlo_p", "holm_p_6_discrete_uniformity_checks")
    return {"alpha": ALPHA, "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "analysis_seed": SEED, "strengths": strengths,
            "pooled_power_comparisons": power, "pooled_null_calibration": null,
            "discrete_rank_uniformity": grid_checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "cache/summaries/coordinate_ablation")
    parser.add_argument("--output", type=Path, default=ROOT / "output/tables/coordinate_ablation.json")
    args = parser.parse_args()
    result = summarize(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
