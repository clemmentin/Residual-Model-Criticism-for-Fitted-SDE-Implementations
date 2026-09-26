"""Recompute calibration tables from retained results without refitting models.

The global ECDF retains all 5,000 global p-values. Component KS calibration
uses the retained component KS distances, sample sizes and rank grids; it
does not reconstruct the unavailable per-replicate component p-values.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


from experiments.full_refit_statistics import (
    COMPONENT_PVALUE_ORDER,
    DEFAULT_ALPHAS,
    OUT_DIR,
    _alpha_table,
    _discrete_uniform_ks_mc_pvalue,
    _summary_table,
)


def summarize(input_dir: Path, out_dir: Path, mc_reps: int, seed: int) -> None:
    if mc_reps <= 0:
        raise ValueError("ks-mc-reps must be positive.")
    stem = "bi_synthetic_full_refit_internal_pvalue"
    pvalues = pd.read_csv(input_dir / f"{stem}_ecdf.csv")["pvalue"].to_numpy(dtype=float)
    components_name = "bi_synthetic_full_refit_component_pvalue_calibration.csv"
    components = pd.read_csv(input_dir / components_name)
    expected_columns = [f"p_{metric}_internal" for metric, _ in COMPONENT_PVALUE_ORDER]
    if len(pvalues) == 0 or sorted(components["column"]) != sorted(expected_columns):
        raise ValueError("Expected a nonempty global ECDF and all six component summaries.")
    components = components.set_index("column").loc[expected_columns].reset_index()
    if not components["n"].eq(len(pvalues)).all() or components["grid"].nunique() != 1:
        raise ValueError("Component sample sizes or rank grids do not match the global ECDF.")
    grid = int(components["grid"].iloc[0])
    distances = components["dks_stat"].to_numpy(dtype=float)
    # On this grid the KS distance has denominator n*grid/gcd(n,grid).
    units = distances * (len(pvalues) * grid // np.gcd(len(pvalues), grid))
    if (grid <= 0 or not np.isfinite(distances).all() or np.any(distances < 0)
            or np.any(distances > 1) or not np.allclose(units, np.rint(units), rtol=0, atol=1e-8)):
        raise ValueError("Retained KS distances must lie on the exact discrete grid.")
    summary = _summary_table(pvalues, grid, mc_reps, seed, "matched_null")
    alpha = _alpha_table(pvalues, DEFAULT_ALPHAS, grid)
    for index, distance in enumerate(distances):
        components.loc[index, "dks_mc_pvalue"] = _discrete_uniform_ks_mc_pvalue(
            float(distance), len(pvalues), grid, mc_reps,
            seed + 50_000 + 1009 * (index + 1),
        )
    components["dks_mc_reps"] = mc_reps
    # Finish all calculations before replacing any retained reporting table.
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in ((f"{stem}_summary.csv", summary),
                        (f"{stem}_alpha_table.csv", alpha),
                        (components_name, components)):
        table.to_csv(out_dir / name, index=False, float_format="%.17g")
        print(f"Wrote {out_dir / name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ks-mc-reps", type=int, default=100_000)
    parser.add_argument("--ks-mc-seed", type=int, default=20260711)
    args = parser.parse_args()
    summarize(args.input_dir, args.out_dir or args.input_dir, args.ks_mc_reps, args.ks_mc_seed)


if __name__ == "__main__":
    main()
