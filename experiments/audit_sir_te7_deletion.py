"""Recompute S_CE ranks after deleting only the frozen lag-seven energy feature."""

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

ROOT = Path(__file__).resolve().parents[1]


from experiments import run_sir_sde_native_country_audit as native
from experiments.audit_statistics import upper_rank


BATCH1 = ROOT / "cache/summaries/random_country_5000_reference_draws"
BATCH2 = (
    ROOT
    / "cache/summaries/sde_native_random_country_replication"
    / "random_country_batch2_frozen/evaluation"
)
REMOVED = "bracket_I_energy_acf7"


def _root(country: str) -> Path:
    return BATCH1 if country in {"BGR", "HUN", "SVK"} else BATCH2


def audit_country(country: str) -> dict[str, float | int | str]:
    country_dir = _root(country) / country
    draws = pd.read_csv(country_dir / "null_metric_draws.csv")
    components = pd.read_csv(country_dir / "component_summary.csv").set_index("metric")
    archived = pd.read_csv(country_dir / "layer_and_global_scores.csv").iloc[0]
    ids = pd.to_numeric(draws["bootstrap_id"], errors="raise").astype(int)
    pilot = draws.loc[ids.mod(2).eq(0)]
    evaluation = draws.loc[ids.mod(2).eq(1)]
    metrics = [metric for metric in native.PRIMARY_COMPONENTS if metric != REMOVED]
    eval_departures = []
    obs_departures = []
    for metric in metrics:
        spec = native.METRIC_SPECS[metric]
        pilot_values = pilot[metric].to_numpy(dtype=float)
        eval_values = evaluation[metric].to_numpy(dtype=float)
        center = float(np.median(pilot_values))
        scale = float(np.std(pilot_values, ddof=1))
        if not np.isfinite(scale) or scale <= 1e-10:
            raise ValueError(f"Degenerate pilot scale for {country}/{metric}: {scale}")
        eval_departures.append(native._departure(eval_values, center, scale, spec["tail"]))
        observed = float(components.loc[metric, "observed"])
        obs_departures.append(
            float(native._departure(np.array([observed]), center, scale, spec["tail"])[0])
        )
    null_score = np.max(np.column_stack(eval_departures), axis=1)
    observed_score = float(np.max(obs_departures))
    pvalue = upper_rank(observed_score, null_score)
    return {
        "country": country,
        "evaluation_n": int(len(null_score)),
        "archived_global_pvalue": float(archived["global_rank_pvalue"]),
        "without_te7_global_pvalue": pvalue,
        "removed_metric": REMOVED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--countries", nargs="+", default=["HUN", "LTU", "MDA", "SVN"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("Diagnostic output must be under repository tmp/.")
    if output.exists():
        raise FileExistsError(output)
    rows = [audit_country(country.upper()) for country in args.countries]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
