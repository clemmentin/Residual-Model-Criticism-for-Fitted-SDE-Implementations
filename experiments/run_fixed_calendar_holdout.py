"""Run the fixed-calendar SIR holdout application.

The freeze phase uses only country metadata and source-data availability.  It
selects one country from the resulting ISO-sorted pool and records the fixed
calendar window before any diagnostic score is calculated.  The evaluate phase
then reuses the existing corrected common fit, SDE-native residual map and
split-calibrated centring--energy score.

Run from the project root:

    python experiments/run_fixed_calendar_holdout.py --phase freeze
    python experiments/run_fixed_calendar_holdout.py --phase evaluate
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

import config as config_module
import sir_data
from trajectory import get_signature_size  # noqa: E402

from experiments import run_sir_sde_native_country_audit as native


FREEZE_DATE = "2026-09-03"
WINDOW_START = pd.Timestamp("2021-01-01")
HORIZON = 60
WINDOW_END = WINDOW_START + pd.Timedelta(days=HORIZON)
HISTORY_START = WINDOW_START - pd.Timedelta(days=19)
MINIMUM_POPULATION = 1_000_000
SELECTION_SEED = 20260903
SELECTION_SIZE = 1
NULL_BANK_BASE_SEED = 20260904
NULL_BANK_PURPOSE = "fixed-calendar-holdout"
N_BOOTSTRAP = 5000
ALPHA = 0.05

DATA_PATH = ROOT / "cache" / "owid_covid_data.csv"
OUT_ROOT = (
    ROOT
    / "cache"
    / "summaries"
    / "fixed_calendar_holdout"
    / "causal_window"
)
FREEZE_PATH = OUT_ROOT / "holdout_freeze.json"
RESULTS_PATH = OUT_ROOT / "holdout_results.csv"
REFERENCE_DRAWS_PATH = OUT_ROOT / "holdout_reference_draws.csv"
METADATA_PATH = OUT_ROOT / "holdout_metadata.json"

EXCLUDED_TARGETS = (
    "AUT",
    "BEL",
    "BGR",
    "CHE",
    "DEU",
    "DNK",
    "ESP",
    "FIN",
    "FRA",
    "GBR",
    "HUN",
    "ITA",
    "LTU",
    "MDA",
    "NLD",
    "NOR",
    "SVK",
    "SVN",
    "SWE",
)

RAW_COLUMNS = [
    "iso_code",
    "continent",
    "date",
    "population",
    "total_cases",
    "new_cases_smoothed",
]


def _read_raw() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Required input file is missing: {DATA_PATH}")
    return pd.read_csv(
        DATA_PATH,
        usecols=RAW_COLUMNS,
        parse_dates=["date"],
        low_memory=False,
    )


def _input_metadata() -> dict[str, object]:
    return {
        "path": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
    }


def _fixed_dates() -> pd.DatetimeIndex:
    return pd.date_range(HISTORY_START, WINDOW_END, freq="D")


def _candidate_pool(raw: pd.DataFrame) -> tuple[list[str], dict[str, object]]:
    """Build the pool without using continuation values or diagnostic scores."""
    expected_dates = _fixed_dates()
    rows: list[dict[str, object]] = []
    pool: list[str] = []
    european_entities = 0
    population_eligible = 0

    europe = raw.loc[
        raw["continent"].eq("Europe")
        & raw["iso_code"].notna()
        & ~raw["iso_code"].astype(str).str.startswith("OWID")
    ]
    for iso, country_frame in europe.groupby("iso_code", sort=True):
        iso = str(iso)
        european_entities += 1
        country_frame = country_frame.sort_values("date")
        before_start = country_frame.loc[country_frame["date"] <= WINDOW_START]
        populations = before_start["population"].dropna()
        population = float(populations.iloc[-1]) if not populations.empty else None
        if population is None or population < MINIMUM_POPULATION:
            continue
        population_eligible += 1

        span = country_frame.loc[
            country_frame["date"].between(HISTORY_START, WINDOW_END)
        ]
        dates = pd.DatetimeIndex(span["date"])
        dates_complete = bool(
            len(span) == len(expected_dates)
            and dates.is_unique
            and dates.equals(expected_dates)
        )
        smoothed_complete = bool(
            dates_complete and span["new_cases_smoothed"].notna().all()
        )
        has_prewindow_anchor = bool(
            before_start["total_cases"].fillna(0).gt(0).any()
        )
        previously_used = iso in EXCLUDED_TARGETS
        eligible = bool(
            not previously_used
            and dates_complete
            and smoothed_complete
            and has_prewindow_anchor
        )
        rows.append(
            {
                "iso": iso,
                "population_at_window_start": population,
                "previously_used_target": previously_used,
                "history_and_window_dates_complete": dates_complete,
                "history_and_window_smoothed_incidence_complete": smoothed_complete,
                "positive_total_cases_anchor_by_window_start": has_prewindow_anchor,
                "eligible": eligible,
            }
        )
        if eligible:
            pool.append(iso)

    pool = sorted(pool)
    if not pool:
        raise ValueError("The fixed-calendar candidate pool is empty.")
    screening = {
        "european_entities": european_entities,
        "population_at_least_one_million": population_eligible,
        "previously_used_target_exclusions": list(EXCLUDED_TARGETS),
        "countries_after_population_screen": rows,
        "eligible_pool_iso_sorted": pool,
    }
    return pool, screening


def _draw_targets(pool: list[str]) -> list[str]:
    rng = np.random.default_rng(SELECTION_SEED)
    return [
        str(country)
        for country in rng.choice(
            np.asarray(pool), size=SELECTION_SIZE, replace=False
        ).tolist()
    ]


def _config(country: str) -> config_module.Config:
    return native._config(country)


def _freeze_payload(raw: pd.DataFrame) -> dict[str, object]:
    pool, screening = _candidate_pool(raw)
    selected = _draw_targets(pool)
    payload = {
        "schema_version": 1,
        "freeze_date": FREEZE_DATE,
        "status": "frozen_before_holdout_scores",
        "analysis_role": (
            "retrospective archived fixed-calendar holdout application; "
            "not prospective and not a preregistration of endpoint development"
        ),
        "data_source": _input_metadata(),
        "candidate_pool": {
            "rule": (
                "ISO codes in Europe with population at least one million "
                "measured by the window start, excluding all prior country-level "
                "diagnostic targets, and having complete daily source rows and "
                "nonmissing new_cases_smoothed values from the 20-state history "
                "through the fixed evaluation end"
            ),
            "no_continuation_value_rule": (
                "No incidence magnitude, nonzero-incidence condition, active-period "
                "condition, fitted-model diagnostic, or score was used."
            ),
            "prewindow_anchor_rule": (
                "The first nonzero total_cases anchor must occur on or before "
                "the fixed evaluation start."
            ),
            "screening": screening,
        },
        "randomization": {
            "library": "NumPy",
            "generator": "Generator(PCG64)",
            "seed": SELECTION_SEED,
            "sampling": "one target without replacement from the ISO-sorted pool",
            "selected_targets_in_draw_order": selected,
            "replacement_after_draw": False,
        },
        "window": {
            "history_start": str(HISTORY_START.date()),
            "evaluation_start_state": str(WINDOW_START.date()),
            "evaluation_end_state": str(WINDOW_END.date()),
            "continuation_dates": (
                f"{(WINDOW_START + pd.Timedelta(days=1)).date()} through "
                f"{WINDOW_END.date()}"
            ),
            "horizon_increments": HORIZON,
            "calendar_rule": (
                "common fixed calendar interval selected without inspecting the "
                "selected target's continuation"
            ),
            "target_conditioning_cutoff": str(WINDOW_START.date()),
        },
        "preprocessing": {
            "training_pipeline": "train_sde.prepare_sir_countries with its existing active-period default",
            "holdout_pipeline": (
                "same smoothed-incidence, cumulative-case and 28-day recursive "
                "active-case reconstruction, but retain the full causal target "
                "series instead of locating an active-period endpoint from the "
                "full series"
            ),
            "cumulative_case_anchor": (
                "first positive total_cases observation on or before "
                f"{WINDOW_START.date()}"
            ),
            "normalization": "frozen training-country normalization from the common checkpoint",
            "future_information_check": (
                "At each target date, S and I use incidence observations through "
                "that date only; no post-window active-period endpoint is used."
            ),
        },
        "fitted_model": {
            "case": "recov28_frzgamma",
            "training_countries": list(native.DEVELOPMENT_COUNTRIES),
            "model_path": str(native.COMMON_MODEL_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "fitting_rule": (
                "reuse the existing common corrected checkpoint fitted on the "
                "nine development countries; no target-specific fitting or model "
                "selection uses the holdout continuation"
            ),
            "solver": "projected Euler-Maruyama, dt=0.1, 10 substeps per day",
            "controls": "causal 20-state path signature, recomputed daily on every path",
        },
        "score": {
            "name": "S_CE",
            "implementation": "experiments/run_sir_sde_native_country_audit.py",
            "primary_components": list(native.PRIMARY_COMPONENTS),
            "component_specs": native.METRIC_SPECS,
            "residual": (
                "next normalized log-I state minus projected deterministic next "
                "state, divided by the square root of projected tangent Sigma_II"
            ),
            "pilot_paths": N_BOOTSTRAP // 2,
            "reference_paths": N_BOOTSTRAP // 2,
            "bank_split": "even bootstrap_id pilot, odd bootstrap_id reference",
            "pilot_standardization": "pilot median and sample standard deviation",
            "reference_rank": "(1 + number of reference departures >= observed departure)/(M + 1)",
            "decision_rule": f"reject the individual target when global rank <= {ALPHA}",
            "reporting_rule": (
                "report the selected target and its status; do not pool the one "
                "country into a rejection-rate claim"
            ),
            "null_bank_base_seed": NULL_BANK_BASE_SEED,
        },
        "non_evaluable_rule": (
            "A selected target is non-evaluable if the fixed history/window dates "
            "or required source values are unavailable, the causal reconstruction "
            "is nonfinite, or the required pre-window anchor is unavailable. "
            "No replacement target is allowed."
        ),
    }
    # The selected target and its full candidate pool are both recorded before
    # the evaluator is allowed to calculate any score.
    return payload


def freeze() -> None:
    if FREEZE_PATH.exists() or RESULTS_PATH.exists() or REFERENCE_DRAWS_PATH.exists():
        raise FileExistsError(
            f"Holdout artifacts already exist under {OUT_ROOT}; refusing to refreeze."
        )
    raw = _read_raw()
    payload = _freeze_payload(raw)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with FREEZE_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(f"Frozen holdout choices: {FREEZE_PATH}")
    print(
        "Candidate pool:",
        ", ".join(payload["candidate_pool"]["screening"]["eligible_pool_iso_sorted"]),
    )
    print(
        "Selected target:",
        ", ".join(payload["randomization"]["selected_targets_in_draw_order"]),
    )


def _read_freeze() -> dict[str, object]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError(
            f"Run the freeze phase first; missing {FREEZE_PATH}"
        )
    payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_holdout_scores":
        raise ValueError("Freeze file has an unexpected status.")
    return payload


def _build_lightweight_data(
    raw: pd.DataFrame,
    cfg: config_module.Config,
    country: str,
) -> config_module.TrainingData:
    """Build only the tensors needed to deserialize the frozen common fit."""
    training_physical = sir_data.prepare_sir_countries(
        raw,
        list(native.DEVELOPMENT_COUNTRIES),
        cfg.TRAINING_START_DATE,
        recovery_days=cfg.SIR_RECOVERY_DAYS,
        active_threshold=cfg.SIR_ACTIVE_THRESHOLD,
        incidence_source=cfg.SIR_INCIDENCE_SOURCE,
        incidence_blend_weight=cfg.SIR_INCIDENCE_BLEND_WEIGHT,
    )
    if set(training_physical) != set(native.DEVELOPMENT_COUNTRIES):
        missing = sorted(set(native.DEVELOPMENT_COUNTRIES) - set(training_physical))
        raise RuntimeError(f"Existing training reconstruction missed countries: {missing}")

    training_log = {iso: frame.copy() for iso, frame in training_physical.items()}
    for frame in training_log.values():
        frame["I"] = np.log(frame["I"])
    norm_mean, norm_std = sir_data.compute_norm_stats(training_log)

    target_physical = sir_data.prepare_sir_countries(
        raw,
        [country],
        cfg.TRAINING_START_DATE,
        recovery_days=cfg.SIR_RECOVERY_DAYS,
        active_threshold=cfg.SIR_ACTIVE_THRESHOLD,
        incidence_source=cfg.SIR_INCIDENCE_SOURCE,
        incidence_blend_weight=cfg.SIR_INCIDENCE_BLEND_WEIGHT,
        trim_to_active_period=False,
        anchor_cutoff=WINDOW_START,
    ).get(country)
    if target_physical is None:
        raise ValueError(f"Causal reconstruction did not produce {country}.")
    target_log = target_physical.copy()
    target_log["I"] = np.log(target_log["I"])
    target_norm = (target_log[["S", "I"]] - norm_mean) / (norm_std + 1e-8)

    normalized_training = [
        (frame[["S", "I"]] - norm_mean) / (norm_std + 1e-8)
        for frame in training_log.values()
    ]
    features_df = pd.concat(normalized_training, axis=0)
    raw_pieces = [frame.assign(country=iso) for iso, frame in training_physical.items()]
    raw_sir_df = pd.concat(raw_pieces, axis=0)
    signature_sizes = [
        get_signature_size(
            cfg.STATE_SIZE,
            cfg.SIGNATURE_DEPTH,
            augment_time=True,
            lead_lag=cfg.SIGNATURE_LEAD_LAG,
        )
        for _ in cfg.SIGNATURE_PATH_LENGTHS
    ]
    macro_size = config_module.get_recon_control_size(cfg)
    empty_ys = jnp.empty(
        (0, cfg.TRAINING_PATH_LENGTH, cfg.STATE_SIZE), dtype=jnp.float32
    )
    empty_controls = jnp.empty(
        (0, cfg.TRAINING_PATH_LENGTH, sum(signature_sizes) + macro_size),
        dtype=jnp.float32,
    )
    return config_module.TrainingData(
        train_set=config_module.TrainingDataSet(
            ys=empty_ys,
            controls=empty_controls,
        ),
        val_set=None,
        features_df=features_df,
        val_features_df=target_norm,
        macro_df=pd.DataFrame(),
        raw_sir_df=raw_sir_df,
        norm_mean=norm_mean,
        norm_std=norm_std,
        signature_sizes=signature_sizes,
        macro_size=macro_size,
        train_country_ids=np.zeros(0, dtype=np.int32),
        val_country_id=-1,
        n_countries=len(native.DEVELOPMENT_COUNTRIES),
        ar_phi=None,
    )


def _fixed_start_position(
    data: config_module.TrainingData,
) -> tuple[int | None, str | None]:
    index = pd.DatetimeIndex(data.val_features_df.index)
    expected_window = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    if not index.is_unique:
        return None, "reconstructed target dates are not unique"
    if WINDOW_START not in index:
        return None, f"fixed start date {WINDOW_START.date()} is missing"
    start_pos = int(index.get_loc(WINDOW_START))
    if start_pos < 19:
        return None, "the required 20-state pre-window history is missing"
    actual_window = index[start_pos : start_pos + HORIZON + 1]
    if not actual_window.equals(expected_window):
        return None, "the fixed history/window dates are incomplete or non-daily"
    values = data.val_features_df.loc[expected_window, ["S", "I"]].to_numpy()
    if not np.isfinite(values).all():
        return None, "the fixed history/window reconstructed states are nonfinite"
    return start_pos, None


def _result_row(
    country: str,
    result: dict[str, object],
    null_seed: int,
) -> dict[str, object]:
    score = result["score"]
    observed = result["observed_metrics"]
    row: dict[str, object] = {
        "country": country,
        "status": "evaluable",
        "evaluation_start": str(WINDOW_START.date()),
        "evaluation_end": str(WINDOW_END.date()),
        "horizon_increments": HORIZON,
        "case": "recov28_frzgamma",
        "model": str(native.COMMON_MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "pilot_n": score["pilot_n"],
        "reference_n": score["evaluation_n"],
        "null_bank_seed": null_seed,
        "observed_global_score": score["observed_global_score"],
        "global_rank_pvalue": score["global_rank_pvalue"],
        "decision_at_0_05": (
            "reject" if score["global_rank_pvalue"] <= ALPHA else "do_not_reject"
        ),
        "dominant_metric": score["dominant_metric"],
        "dominant_departure": score["dominant_departure"],
        "martingale_rank_pvalue": score["martingale_rank_pvalue"],
        "bracket_rank_pvalue": score["bracket_rank_pvalue"],
        "generator_rank_pvalue_diagnostic": score["generator_rank_pvalue"],
        "global_null_q500": score["global_null_q500"],
        "global_null_q950": score["global_null_q950"],
        "mean_null_clip_fraction": float(result["draws"]["clip_fraction"].mean()),
        "max_null_clip_fraction": float(result["draws"]["clip_fraction"].max()),
    }
    for metric in native.METRIC_SPECS:
        row[f"observed_{metric}"] = observed[metric]
    return row


def evaluate() -> None:
    if RESULTS_PATH.exists() or REFERENCE_DRAWS_PATH.exists() or METADATA_PATH.exists():
        raise FileExistsError(
            f"Final holdout outputs already exist under {OUT_ROOT}; refusing to rerun."
        )
    payload = _read_freeze()
    raw = _read_raw()
    selected = list(payload["randomization"]["selected_targets_in_draw_order"])
    if len(selected) != SELECTION_SIZE:
        raise ValueError("Unexpected number of selected targets in the freeze file.")

    result_rows: list[dict[str, object]] = []
    all_draws: list[pd.DataFrame] = []
    target_metadata: list[dict[str, object]] = []

    for country in selected:
        country = str(country)
        print(f"=== fixed-calendar holdout: {country} ===", flush=True)
        cfg = _config(country)
        data = _build_lightweight_data(raw, cfg, country)
        start_pos, reason = _fixed_start_position(data)
        if start_pos is None:
            result_rows.append(
                {
                    "country": country,
                    "status": "non_evaluable",
                    "status_detail": reason,
                    "evaluation_start": str(WINDOW_START.date()),
                    "evaluation_end": str(WINDOW_END.date()),
                    "horizon_increments": HORIZON,
                }
            )
            target_metadata.append(
                {"country": country, "status": "non_evaluable", "status_detail": reason}
            )
            continue

        model = native._load_common_original9_model(cfg, data)
        null_seed = native._country_seed(
            NULL_BANK_BASE_SEED, country, NULL_BANK_PURPOSE
        )
        result = native._evaluate_country_paths(
            model,
            data,
            cfg,
            n_bootstrap=N_BOOTSTRAP,
            seed=null_seed,
            start_pos=start_pos,
            horizon=HORIZON,
        )
        draws = result["draws"].copy()
        required_metrics = set(native.METRIC_SPECS)
        if len(draws) != N_BOOTSTRAP:
            raise AssertionError(
                f"Expected {N_BOOTSTRAP} reference paths, found {len(draws)}."
            )
        if not required_metrics.issubset(draws.columns):
            raise AssertionError("Reference paths do not contain the frozen score metrics.")
        if set(result["observed_metrics"]) != required_metrics:
            raise AssertionError("Observed path does not contain the frozen score metrics.")
        if int(result["score"]["pilot_n"]) != N_BOOTSTRAP // 2:
            raise AssertionError("Unexpected pilot-bank size.")
        if int(result["score"]["evaluation_n"]) != N_BOOTSTRAP // 2:
            raise AssertionError("Unexpected reference-bank size.")
        draws.insert(0, "country", country)
        all_draws.append(draws)
        result_rows.append(_result_row(country, result, null_seed))
        target_metadata.append(
            {
                "country": country,
                "status": "evaluable",
                "start_position": int(start_pos),
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "model_training_countries": list(native.DEVELOPMENT_COUNTRIES),
                "null_seed": null_seed,
                "observed_metrics": result["observed_metrics"],
                "score": result["score"],
            }
        )

    if len(result_rows) != len(selected):
        raise AssertionError("A selected target was silently dropped.")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(RESULTS_PATH, index=False, float_format="%.12g")
    if all_draws:
        pd.concat(all_draws, ignore_index=True).to_csv(
            REFERENCE_DRAWS_PATH, index=False, float_format="%.12g"
        )

    metadata = {
        "schema_version": 1,
        "status": "completed",
        "freeze_path": str(FREEZE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "data_source": _input_metadata(),
        "selected_targets_in_draw_order": selected,
        "window": {
            "evaluation_start_state": str(WINDOW_START.date()),
            "evaluation_end_state": str(WINDOW_END.date()),
            "horizon_increments": HORIZON,
        },
        "score": {
            "name": "S_CE",
            "primary_components": list(native.PRIMARY_COMPONENTS),
            "same_frozen_score_on_observed_and_reference": True,
            "n_bootstrap_requested": N_BOOTSTRAP,
            "pilot_n": N_BOOTSTRAP // 2,
            "reference_n": N_BOOTSTRAP // 2,
            "rank_rule": "plus-one upper rank after pilot standardization",
        },
        "targets": target_metadata,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(results_df.to_string(index=False), flush=True)
    print(f"Saved holdout outputs under {OUT_ROOT}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "evaluate"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "freeze":
        freeze()
    else:
        evaluate()


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
