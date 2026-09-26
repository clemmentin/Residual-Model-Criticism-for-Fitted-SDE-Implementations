"""Build paper-facing country BI tables from archived per-country null draws.

This is a deterministic reporting step.  It does not fit a model or simulate a
new null bank.  The pilot/evaluation split, component tails, and rank rule match
the registered FIN/NOR/SWE score definition.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd


from experiments.country_bi_scoring import build_country_rows, build_single_component_profile


ROOT = Path(__file__).resolve().parents[1]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--case", default="recov28_frzgamma")
    parser.add_argument("--countries", nargs="+", default=["FIN", "NOR", "SWE"])
    return parser.parse_args()


def resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    input_root = resolve_from_root(args.input_root).resolve()
    input_root.mkdir(parents=True, exist_ok=True)

    component_frames: list[pd.DataFrame] = []
    global_rows: list[dict[str, object]] = []
    adjusted_rows: list[dict[str, float]] = []
    single_component_rows: list[dict[str, float | str]] = []
    for country in args.countries:
        country = country.upper()
        country_dir = input_root / f"{args.case}_val{country}"
        summary_path = country_dir / f"{args.artifact_prefix}_bi_summary.csv"
        draws_path = country_dir / f"{args.artifact_prefix}_bi_draws.csv"
        summary = pd.read_csv(summary_path)
        draws = pd.read_csv(draws_path)
        component_frames.append(summary)
        global_row, adjusted_row = build_country_rows(
            summary,
            draws,
            expected_split_size=250,
        )
        global_rows.append(global_row)
        adjusted_rows.append(adjusted_row)
        single_component_rows.append(
            build_single_component_profile(
                summary,
                draws,
                expected_split_size=250,
            )
        )

    component_path = input_root / f"{args.output_stem}_component_summary.csv"
    global_path = input_root / f"{args.output_stem}_global_scores.csv"
    adjusted_path = input_root / f"{args.output_stem}_max_adjusted_profile.csv"
    single_component_path = (
        input_root / f"{args.output_stem}_single_component_profile.csv"
    )
    component_frame = pd.concat(component_frames, ignore_index=True)
    global_frame = pd.DataFrame(global_rows)
    component_frame.to_csv(component_path, index=False)
    global_frame.to_csv(global_path, index=False)
    pd.DataFrame(adjusted_rows).to_csv(adjusted_path, index=False)
    pd.DataFrame(single_component_rows).to_csv(single_component_path, index=False)

    print(f"Wrote {component_path.relative_to(ROOT)}")
    print(f"Wrote {global_path.relative_to(ROOT)}")
    print(f"Wrote {adjusted_path.relative_to(ROOT)}")
    print(f"Wrote {single_component_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
