"""Reproduce the recorded BGR/HUN/SVK 5,000-path banks for figure data.

The observed paths and decisions were already viewed before this reproduction.
This script does not create new evidence.  It recreates the exact fitted-null
draws from the seeds in ``random_country_5000_metadata.json`` and
refuses to write unless every recorded score and decision is recovered.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


from experiments import run_sir_sde_native_country_audit as native


SOURCE_METADATA = ROOT / "cache" / "summaries" / "random_country_5000_metadata.json"
SOURCE_RESULTS = ROOT / "cache" / "summaries" / "random_country_5000.csv"
OUT_ROOT = ROOT / "cache" / "summaries" / "random_country_5000_reference_draws"
COUNTRIES = ("BGR", "HUN", "SVK")
N_BOOTSTRAP = 5000
BASE_SEED = 20260813
SEED_PURPOSE = "exploratory-5000-fitted-null"


def _close(left: float, right: float, *, atol: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= atol


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing reproduction: {OUT_ROOT}")
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    source_rows = {
        str(row.country): row
        for row in pd.read_csv(SOURCE_RESULTS).itertuples(index=False)
    }
    if tuple(metadata["countries"]) != COUNTRIES:
        raise ValueError("Recorded country order does not match the reproduction settings.")
    if int(metadata["null_bank"]["n_paths"]) != N_BOOTSTRAP:
        raise ValueError("Recorded null-bank size does not match the reproduction settings.")

    staging = OUT_ROOT.with_name(OUT_ROOT.name + "_staging")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    summary_rows: list[dict[str, object]] = []
    try:
        for country in COUNTRIES:
            print(f"=== Reproduce exploratory bank: {country} ===", flush=True)
            cfg = native._config(country)
            native.bi._validate_supported_config(cfg)
            data = native.load_cached_data(cfg)
            model = native._load_common_original9_model(cfg, data)
            country_seed = native._country_seed(BASE_SEED, country, SEED_PURPOSE)
            result = native._evaluate_country_paths(
                model,
                data,
                cfg,
                n_bootstrap=N_BOOTSTRAP,
                seed=country_seed,
            )
            expected = source_rows[country]
            score = result["score"]
            checks = {
                "country_seed": (country_seed, int(expected.country_seed)),
                "global_p": (score["global_rank_pvalue"], expected.global_p),
                "martingale_p": (score["martingale_rank_pvalue"], expected.martingale_p),
                "bracket_p": (score["bracket_rank_pvalue"], expected.bracket_p),
                "generator_p": (score["generator_rank_pvalue"], expected.generator_p),
                "global_score": (score["observed_global_score"], expected.global_score),
                "global_null_q500": (score["global_null_q500"], expected.global_null_q500),
                "global_null_q950": (score["global_null_q950"], expected.global_null_q950),
            }
            failures = [
                f"{name}: reproduced={left!r}, recorded={right!r}"
                for name, (left, right) in checks.items()
                if not _close(float(left), float(right))
            ]
            if str(score["dominant_metric"]) != str(expected.dominant_metric):
                failures.append(
                    "dominant_metric: "
                    f"reproduced={score['dominant_metric']!r}, recorded={expected.dominant_metric!r}"
                )
            if failures:
                raise ValueError(f"{country} reproduction mismatch: " + "; ".join(failures))

            country_dir = staging / country
            country_dir.mkdir(parents=True, exist_ok=False)
            draws = result["draws"].copy()
            draws.insert(0, "country", country)
            draws.to_csv(country_dir / "null_metric_draws.csv", index=False)
            component = result["component_summary"].copy()
            component.insert(0, "country", country)
            component.to_csv(country_dir / "component_summary.csv", index=False)
            score_row = {"country": country, **score}
            pd.DataFrame([score_row]).to_csv(
                country_dir / "layer_and_global_scores.csv", index=False
            )
            summary_rows.append(score_row)
            print(
                f"verified global p={float(score['global_rank_pvalue']):.12g}, "
                f"score={float(score['observed_global_score']):.12g}",
                flush=True,
            )

        pd.DataFrame(summary_rows).to_csv(staging / "country_scores.csv", index=False)
        metadata = {
            "analysis_role": (
                "exact deterministic reproduction of already-viewed exploratory fitted-null "
                "banks for figure construction; not new confirmatory evidence"
            ),
            "countries": list(COUNTRIES),
            "n_bootstrap": N_BOOTSTRAP,
            "pilot_n": N_BOOTSTRAP // 2,
            "evaluation_n": N_BOOTSTRAP // 2,
            "base_seed": BASE_SEED,
            "country_seed_derivation_purpose": SEED_PURPOSE,
            "source_metadata_path": str(SOURCE_METADATA.relative_to(ROOT)).replace("\\", "/"),
            "source_results_path": str(SOURCE_RESULTS.relative_to(ROOT)).replace("\\", "/"),
            "all_recorded_scores_reproduced": True,
        }
        (staging / "reproduction_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(OUT_ROOT)
    except Exception:
        # Preserve staging artifacts on failure for diagnosis; never silently retry.
        raise

    print(f"Completed verified reproduction: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
