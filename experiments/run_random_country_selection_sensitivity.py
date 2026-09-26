"""Retrospective window-sensitivity audit for the randomized country batch.

This audit cannot repair the outcome-dependent eligibility rule or restore
confirmatory status.  It asks a narrower numerical question: do the extreme
reference ranks at the published midpoint windows persist at nearby windows
and at calendar dates fixed identically across LTU, MDA, and SVN?

Run ``freeze`` before ``evaluate``.  The design records every window and seed
before any new reference bank is generated.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


from experiments import (
    run_sir_sde_native_country_audit as base,
    run_sir_sde_native_temporal_audit as temporal,
)
from experiments.audit_sir_beta_clipping import load_audit_data
import sir_data


COUNTRIES = ("LTU", "MDA", "SVN")
HORIZON = 60
N_BOOTSTRAP = 5000
BASE_SEED = 20260831
NEIGHBOR_WINDOWS = (
    ("midpoint_m120", -120),
    ("midpoint_m60", -60),
    ("midpoint_p60", 60),
)
CALENDAR_WINDOWS = (
    ("calendar_2021_06_01", "2021-06-01"),
    ("calendar_2022_06_01", "2022-06-01"),
    ("calendar_2023_06_01", "2023-06-01"),
)

OUT_ROOT = (
    ROOT
    / "cache"
    / "summaries"
    / "random_country_selection_sensitivity"
    / "fixed_calendar_and_neighbor_windows"
)
DESIGN_PATH = OUT_ROOT / "design.json"
STAGING_DIR = OUT_ROOT / "_evaluation_staging"
FINAL_DIR = OUT_ROOT / "evaluation"
ANCHOR_ROOT = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_random_country_replication"
    / "random_country_batch2_frozen"
    / "evaluation"
)


def _relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _window_seed(country: str, label: str) -> int:
    return base._country_seed(BASE_SEED, country, f"selection-sensitivity:{label}")


def _validate_window(data, cfg, start_pos: int, label: str) -> None:
    raw_lookback = int(sir_data.get_sir_control_lookback(cfg)) + (
        1 if data.ar_phi is not None else 0
    )
    if start_pos - raw_lookback + 1 < 0:
        raise ValueError(f"{cfg.VAL_COUNTRY}/{label} lacks its causal lookback.")
    if start_pos + HORIZON >= len(data.val_features_df):
        raise ValueError(f"{cfg.VAL_COUNTRY}/{label} exceeds the observed trajectory.")


def _schedule() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for country in COUNTRIES:
        cfg = base._config(country)
        data = load_audit_data(cfg)
        anchor = int(base.bi._start_position(data, cfg))

        _validate_window(data, cfg, anchor, "published_midpoint")
        rows.append(
            {
                "country": country,
                "window_label": "published_midpoint",
                "window_family": "existing_anchor",
                "start_pos": anchor,
                "start_date": str(data.val_features_df.index[anchor].date()),
                "end_date": str(data.val_features_df.index[anchor + HORIZON].date()),
                "offset_from_midpoint": 0,
                "null_seed": base._country_seed(20260814, country, "fitted-null"),
                "evaluation_action": "reuse_existing_5000_path_result",
            }
        )

        for label, offset in NEIGHBOR_WINDOWS:
            start_pos = anchor + int(offset)
            _validate_window(data, cfg, start_pos, label)
            rows.append(
                {
                    "country": country,
                    "window_label": label,
                    "window_family": "midpoint_neighbor",
                    "start_pos": start_pos,
                    "start_date": str(data.val_features_df.index[start_pos].date()),
                    "end_date": str(
                        data.val_features_df.index[start_pos + HORIZON].date()
                    ),
                    "offset_from_midpoint": int(offset),
                    "null_seed": _window_seed(country, label),
                    "evaluation_action": "generate_new_5000_path_result",
                }
            )

        index = data.val_features_df.index
        for label, date_text in CALENDAR_WINDOWS:
            timestamp = pd.Timestamp(date_text)
            matches = index.get_indexer([timestamp])
            if int(matches[0]) < 0:
                raise ValueError(f"{country}/{label}: date {date_text} is unavailable.")
            start_pos = int(matches[0])
            _validate_window(data, cfg, start_pos, label)
            rows.append(
                {
                    "country": country,
                    "window_label": label,
                    "window_family": "common_calendar_date",
                    "start_pos": start_pos,
                    "start_date": str(index[start_pos].date()),
                    "end_date": str(index[start_pos + HORIZON].date()),
                    "offset_from_midpoint": int(start_pos - anchor),
                    "null_seed": _window_seed(country, label),
                    "evaluation_action": "generate_new_5000_path_result",
                }
            )
    return rows


def _design() -> dict[str, object]:
    return {
        "role": (
            "retrospective selection-sensitivity audit; numerical robustness only; "
            "does not repair outcome-dependent eligibility, post-selection calibration, "
            "or confirmatory status"
        ),
        "countries": list(COUNTRIES),
        "horizon_increments": HORIZON,
        "n_bootstrap_per_new_window": N_BOOTSTRAP,
        "pilot_evaluation_split": "even/odd bootstrap_id, 2500 each",
        "base_seed": BASE_SEED,
        "neighbor_windows": [
            {"label": label, "offset_from_midpoint": offset}
            for label, offset in NEIGHBOR_WINDOWS
        ],
        "calendar_windows": [
            {"label": label, "fixed_start_date": date_text}
            for label, date_text in CALENDAR_WINDOWS
        ],
        "score_and_model": (
            "unchanged S_CE endpoint and common original-nine corrected fitted model"
        ),
        "schedule": _schedule(),
        "owid_snapshot": {
            "path": _relative(base.OWID_CACHE_PATH),
        },
        "model": {
            "path": _relative(base.COMMON_MODEL_PATH),
        },
    }


def freeze() -> None:
    if DESIGN_PATH.exists() or STAGING_DIR.exists() or FINAL_DIR.exists():
        raise FileExistsError(f"Selection-sensitivity artifacts already exist: {OUT_ROOT}")
    design = _design()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **design,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    DESIGN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Frozen retrospective sensitivity design: {DESIGN_PATH}", flush=True)
    print(pd.DataFrame(payload["schedule"]).to_string(index=False), flush=True)


def _load_design() -> dict[str, object]:
    if not DESIGN_PATH.exists():
        raise FileNotFoundError("Run the freeze phase before evaluation.")
    saved = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if not saved.get("schedule"):
        raise ValueError("The sensitivity design does not contain a window schedule.")
    return saved


def _existing_anchor_row(country: str, schedule_row: dict[str, object]) -> dict[str, object]:
    metadata_path = ANCHOR_ROOT / country / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["start_date"] != schedule_row["start_date"]:
        raise ValueError(f"Existing {country} anchor date does not match the design.")
    score = metadata["score"]
    return {
        "country": country,
        "window_label": "published_midpoint",
        "window_family": "existing_anchor",
        "start_date": metadata["start_date"],
        "end_date": metadata["end_date"],
        "offset_from_midpoint": 0,
        "n_bootstrap": int(metadata["n_bootstrap"]),
        "global_rank_pvalue": float(score["global_rank_pvalue"]),
        "martingale_rank_pvalue": float(score["martingale_rank_pvalue"]),
        "bracket_rank_pvalue": float(score["bracket_rank_pvalue"]),
        "generator_rank_pvalue": float(score["generator_rank_pvalue"]),
        "dominant_metric": str(score["dominant_metric"]),
        "observed_global_score": float(score["observed_global_score"]),
        "result_source": _relative(metadata_path),
    }


def evaluate() -> None:
    design = _load_design()
    if STAGING_DIR.exists() or FINAL_DIR.exists():
        raise FileExistsError("Sensitivity evaluation output already exists.")
    STAGING_DIR.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    schedule = pd.DataFrame(design["schedule"])

    for country in COUNTRIES:
        cfg = base._config(country)
        base.bi._validate_supported_config(cfg)
        data = load_audit_data(cfg)
        model = base._load_common_original9_model(cfg, data)
        country_schedule = schedule.loc[schedule["country"].eq(country)]

        anchor_spec = country_schedule.loc[
            country_schedule["window_family"].eq("existing_anchor")
        ].iloc[0].to_dict()
        rows.append(_existing_anchor_row(country, anchor_spec))

        for spec in country_schedule.loc[
            ~country_schedule["window_family"].eq("existing_anchor")
        ].itertuples(index=False):
            print(
                f"=== selection sensitivity: {country}/{spec.window_label} "
                f"{spec.start_date}--{spec.end_date} ===",
                flush=True,
            )
            result = temporal._evaluate_window(
                model,
                data,
                cfg,
                start_pos=int(spec.start_pos),
                n_bootstrap=N_BOOTSTRAP,
                seed=int(spec.null_seed),
            )
            window_dir = STAGING_DIR / country / str(spec.window_label)
            window_dir.mkdir(parents=True)
            result["draws"].to_csv(window_dir / "null_metric_draws.csv", index=False)
            result["component_summary"].to_csv(
                window_dir / "component_summary.csv", index=False
            )
            score = result["score"]
            metadata = {
                "country": country,
                "window_label": str(spec.window_label),
                "window_family": str(spec.window_family),
                "start_pos": int(spec.start_pos),
                "start_date": str(spec.start_date),
                "end_date": str(spec.end_date),
                "offset_from_midpoint": int(spec.offset_from_midpoint),
                "n_bootstrap": N_BOOTSTRAP,
                "null_seed": int(spec.null_seed),
                "model_path": _relative(base.COMMON_MODEL_PATH),
                "model_training_countries": list(base.DEVELOPMENT_COUNTRIES),
                "score": score,
                "observed_metrics": result["observed_metrics"],
            }
            (window_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            row = {
                "country": country,
                "window_label": str(spec.window_label),
                "window_family": str(spec.window_family),
                "start_date": str(spec.start_date),
                "end_date": str(spec.end_date),
                "offset_from_midpoint": int(spec.offset_from_midpoint),
                "n_bootstrap": N_BOOTSTRAP,
                "global_rank_pvalue": float(score["global_rank_pvalue"]),
                "martingale_rank_pvalue": float(score["martingale_rank_pvalue"]),
                "bracket_rank_pvalue": float(score["bracket_rank_pvalue"]),
                "generator_rank_pvalue": float(score["generator_rank_pvalue"]),
                "dominant_metric": str(score["dominant_metric"]),
                "observed_global_score": float(score["observed_global_score"]),
                "result_source": _relative(window_dir / "metadata.json"),
            }
            rows.append(row)
            print(
                pd.Series(
                    {
                        "global_p": row["global_rank_pvalue"],
                        "bracket_p": row["bracket_rank_pvalue"],
                        "dominant": row["dominant_metric"],
                    }
                ).to_string(),
                flush=True,
            )

    scores = pd.DataFrame(rows).sort_values(
        ["country", "window_family", "start_date"], kind="stable"
    )
    scores.to_csv(STAGING_DIR / "window_scores.csv", index=False)
    new_rows = scores.loc[~scores["window_family"].eq("existing_anchor")].copy()
    per_country = {}
    for country, group in new_rows.groupby("country", sort=False):
        values = group["global_rank_pvalue"].astype(float)
        per_country[country] = {
            "n_new_windows": int(len(group)),
            "n_below_0_05": int((values <= 0.05).sum()),
            "min_p": float(values.min()),
            "median_p": float(values.median()),
            "max_p": float(values.max()),
        }
    summary = {
        "role": design["role"],
        "n_existing_anchor_rows": int(
            scores["window_family"].eq("existing_anchor").sum()
        ),
        "n_new_window_rows": int(len(new_rows)),
        "n_new_rows_below_0_05": int(
            (new_rows["global_rank_pvalue"].astype(float) <= 0.05).sum()
        ),
        "per_country": per_country,
        "interpretation": (
            "Descriptive robustness only. Repeated and calendar windows are not "
            "independent tests, and the audit does not condition reference paths on "
            "the original eligibility rule."
        ),
    }
    (STAGING_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    metadata = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "output_files": [
            str(path.relative_to(STAGING_DIR)).replace("\\", "/")
            for path in sorted(STAGING_DIR.rglob("*"))
            if path.is_file()
        ],
    }
    (STAGING_DIR / "run_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    STAGING_DIR.replace(FINAL_DIR)
    print(scores.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "evaluate"), required=True)
    return parser.parse_args()


def main() -> None:
    phase = parse_args().phase
    if phase == "freeze":
        freeze()
    else:
        evaluate()


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
