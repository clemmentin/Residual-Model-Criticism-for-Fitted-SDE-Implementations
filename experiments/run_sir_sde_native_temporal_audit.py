"""Post-registration blocked temporal extension of the Nordic SDE audit.

This runner deliberately leaves the frozen 2026-07-18 country audit untouched.
It evaluates the same fitted model and the same SDE-native endpoint family on
four consecutive 60-increment blocks per Nordic country.  The blocks are
defined before evaluation by fixed integer offsets from each country's
registered audit origin: -120, -60, 0, and +60 increments.

The temporal windows are repeated measurements, not independent replicates.
Outputs from this runner are therefore descriptive temporal-robustness
evidence and must not be relabelled as part of the original confirmatory audit.
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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


from experiments import run_sir_sde_native_country_audit as base
from experiments._bi_audit_common import load_cached_data
import sir_data


COUNTRIES = base.CONFIRMATORY_COUNTRIES
HORIZON = 60
WINDOW_SPECS = (
    ("m2", -120),
    ("m1", -60),
    ("anchor", 0),
    ("p1", 60),
)
DEFAULT_OUT_ROOT = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_temporal_audit"
    / "nordic_four_block_B5000_frozen"
)
SCHEDULE_NAME = "window_schedule.csv"
ORIGINAL_SEED = 20260718
DEFAULT_SEED = 20260806


def _relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _window_seed(base_seed: int, country: str, label: str, offset: int) -> int:
    # Preserve the exact fitted-null bank for the already registered anchor.
    if offset == 0:
        return base._country_seed(ORIGINAL_SEED, country, "fitted-null")
    purpose = f"temporal-fitted-null:{label}:{offset:+d}"
    return base._country_seed(base_seed, country, purpose)


def _build_schedule(seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country in COUNTRIES:
        cfg = base._config(country)
        data = load_cached_data(cfg)
        anchor = int(base.bi._start_position(data, cfg))
        raw_lookback = int(sir_data.get_sir_control_lookback(cfg)) + (
            1 if data.ar_phi is not None else 0
        )
        for order, (label, offset) in enumerate(WINDOW_SPECS):
            start_pos = anchor + offset
            end_pos = start_pos + HORIZON
            if start_pos - raw_lookback + 1 < 0:
                raise ValueError(
                    f"{country}/{label} lacks the frozen control lookback."
                )
            if end_pos >= len(data.val_features_df):
                raise ValueError(f"{country}/{label} exceeds the observed trajectory.")
            rows.append(
                {
                    "country": country,
                    "window_order": order,
                    "window_label": label,
                    "offset_from_registered_start": offset,
                    "is_registered_anchor": bool(offset == 0),
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "start_date": str(data.val_features_df.index[start_pos].date()),
                    "end_date": str(data.val_features_df.index[end_pos].date()),
                    "horizon_increments": HORIZON,
                    "null_seed": _window_seed(seed, country, label, offset),
                }
            )
    schedule = pd.DataFrame(rows)
    for country, group in schedule.groupby("country", sort=False):
        ordered = group.sort_values("start_pos")
        starts = ordered["start_pos"].to_numpy(dtype=int)
        ends = ordered["end_pos"].to_numpy(dtype=int)
        if np.any(starts[1:] < ends[:-1]):
            raise ValueError(f"Increment sets overlap for {country}.")
        if not np.all(starts[1:] == ends[:-1]):
            raise ValueError(f"The fixed blocks are not consecutive for {country}.")
    return schedule


def freeze(out_root: Path, *, seed: int, n_bootstrap: int) -> Path:
    del n_bootstrap
    schedule_path = out_root / SCHEDULE_NAME
    if schedule_path.exists():
        raise FileExistsError(f"Temporal schedule already exists under {out_root}.")
    if (out_root / "temporal_windows").exists() or (
        out_root / "_temporal_windows_staging"
    ).exists():
        raise RuntimeError("Refusing to freeze beside temporal audit outputs.")
    schedule = _build_schedule(seed)
    out_root.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(schedule_path, index=False)
    print(f"Saved temporal schedule: {schedule_path}", flush=True)
    print(schedule.to_string(index=False), flush=True)
    return schedule_path


def _load_schedule(out_root: Path, *, seed: int) -> pd.DataFrame:
    schedule_path = out_root / SCHEDULE_NAME
    if not schedule_path.exists():
        raise FileNotFoundError("Temporal evaluation requires window_schedule.csv.")
    schedule = pd.read_csv(schedule_path)
    expected_schedule = _build_schedule(seed)
    pd.testing.assert_frame_equal(
        schedule,
        expected_schedule,
        check_dtype=False,
    )
    return schedule


def _evaluate_window(
    model,
    data,
    cfg,
    *,
    start_pos: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    context = base._make_context(model, data, cfg)
    horizon = HORIZON
    if start_pos + horizon >= len(data.val_features_df):
        raise ValueError("The frozen temporal window exceeds the country trajectory.")
    raw_lookback = int(context["raw_lookback"])
    if start_pos - raw_lookback + 1 < 0:
        raise ValueError("The frozen temporal window lacks its control lookback.")
    initial_history = jnp.asarray(
        data.val_features_df.iloc[
            start_pos - raw_lookback + 1 : start_pos + 1
        ].values
    )
    observed_future = jnp.asarray(
        data.val_features_df.iloc[start_pos + 1 : start_pos + horizon + 1].values
    )
    start_weekday = jnp.array(
        int(data.val_features_df.index[start_pos].weekday()), dtype=jnp.int32
    )

    evaluate_future = context["evaluate_future"]
    simulate_one = context["simulate_one"]

    @eqx.filter_jit
    def evaluate_observed(future):
        return evaluate_future(initial_history, future, start_weekday)

    @eqx.filter_jit
    def simulate_batch(noise):
        return jax.vmap(
            lambda one_noise: simulate_one(initial_history, one_noise, start_weekday)
        )(noise)

    @eqx.filter_jit
    def evaluate_batch(futures):
        return jax.vmap(
            lambda future: evaluate_future(initial_history, future, start_weekday)
        )(futures)

    observed_outputs = evaluate_observed(observed_future)
    noise = jax.random.normal(
        jax.random.PRNGKey(seed),
        shape=(n_bootstrap, horizon, int(context["n_substeps"])),
    )
    simulated_future, clipped = simulate_batch(noise)
    null_outputs = evaluate_batch(simulated_future)

    observed_z = np.asarray(observed_outputs[0])
    observed_dynkin = np.asarray(observed_outputs[1])
    null_z = np.asarray(null_outputs[0])
    null_dynkin = np.asarray(null_outputs[1])
    clip_fraction = np.asarray(clipped, dtype=float) / float(
        horizon * int(context["n_substeps"])
    )
    observed_metrics = base.compute_path_metrics(observed_z, observed_dynkin)
    draws = pd.DataFrame(
        [
            {
                "bootstrap_id": bootstrap_id,
                "clip_fraction": float(clip_fraction[bootstrap_id]),
                **base.compute_path_metrics(
                    null_z[bootstrap_id], null_dynkin[bootstrap_id]
                ),
            }
            for bootstrap_id in range(n_bootstrap)
        ]
    )
    score, component_summary = base.build_split_scores(observed_metrics, draws)
    return {
        "start_pos": int(start_pos),
        "start_date": str(data.val_features_df.index[start_pos].date()),
        "end_date": str(data.val_features_df.index[start_pos + horizon].date()),
        "horizon": horizon,
        "observed_metrics": observed_metrics,
        "draws": draws,
        "score": score,
        "component_summary": component_summary,
        "raw": {
            "observed_z_I": observed_z,
            "observed_dynkin_z": observed_dynkin,
            "observed_predicted_state": np.asarray(observed_outputs[2]),
            "observed_tangent_cov": np.asarray(observed_outputs[3]),
            "observed_training_plugin_var": np.asarray(observed_outputs[4]),
            "observed_generator_integral": np.asarray(observed_outputs[5]),
            "observed_generator_qv": np.asarray(observed_outputs[6]),
        },
    }


def _write_window(
    phase_dir: Path,
    row,
    result: dict[str, object],
    alignment: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame]:
    country = str(row.country)
    label = str(row.window_label)
    window_dir = phase_dir / country / label
    window_dir.mkdir(parents=True, exist_ok=False)

    id_columns = {
        "country": country,
        "window_label": label,
        "offset_from_registered_start": int(row.offset_from_registered_start),
        "is_registered_anchor": bool(row.is_registered_anchor),
        "start_date": str(row.start_date),
        "end_date": str(row.end_date),
    }
    draws = result["draws"].copy()
    for name, value in reversed(list(id_columns.items())):
        draws.insert(0, name, value)
    draws.to_csv(window_dir / "null_metric_draws.csv", index=False)

    component = result["component_summary"].copy()
    for name, value in reversed(list(id_columns.items())):
        component.insert(0, name, value)
    component.to_csv(window_dir / "component_summary.csv", index=False)

    score = {**id_columns, **result["score"]}
    pd.DataFrame([score]).to_csv(
        window_dir / "layer_and_global_scores.csv", index=False
    )

    alignment = alignment.copy()
    alignment.insert(1, "window_label", label)
    alignment.insert(2, "offset_from_registered_start", int(row.offset_from_registered_start))
    alignment.insert(3, "is_registered_anchor", bool(row.is_registered_anchor))
    alignment.insert(4, "window_start_date", str(row.start_date))
    alignment.insert(5, "window_end_date", str(row.end_date))
    alignment.to_csv(window_dir / "finite_step_alignment.csv", index=False)
    np.savez(window_dir / "observed_path_diagnostics.npz", **result["raw"])

    metadata = {
        **id_columns,
        "start_pos": int(row.start_pos),
        "end_pos": int(row.end_pos),
        "horizon_increments": int(result["horizon"]),
        "n_bootstrap": int(n_bootstrap),
        "null_seed": int(row.null_seed),
        "model_path": _relative(base.COMMON_MODEL_PATH),
        "model_training_countries": list(base.DEVELOPMENT_COUNTRIES),
        "observed_metrics": result["observed_metrics"],
        "score": score,
        "mean_null_clip_fraction": float(draws["clip_fraction"].mean()),
        "max_null_clip_fraction": float(draws["clip_fraction"].max()),
    }
    (window_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata, score, component, alignment


def evaluate(
    out_root: Path,
    *,
    seed: int,
    n_bootstrap: int,
    n_alignment_anchors: int,
    n_alignment_mc: int,
) -> Path:
    schedule = _load_schedule(out_root, seed=seed)
    final_dir = out_root / "temporal_windows"
    staging_dir = out_root / "_temporal_windows_staging"
    if final_dir.exists():
        raise FileExistsError(f"Temporal outputs already exist: {final_dir}")
    if staging_dir.exists():
        raise FileExistsError(
            f"A prior staging directory exists and must be inspected: {staging_dir}"
        )
    staging_dir.mkdir(parents=True)
    metadata_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    components: list[pd.DataFrame] = []
    alignments: list[pd.DataFrame] = []

    for country in COUNTRIES:
        cfg = base._config(country)
        base.bi._validate_supported_config(cfg)
        data = load_cached_data(cfg)
        model = base._load_common_original9_model(cfg, data)
        country_rows = schedule.loc[schedule["country"].eq(country)].sort_values(
            "window_order"
        )
        for row in country_rows.itertuples(index=False):
            print(
                f"=== temporal SDE audit: {country}/{row.window_label} "
                f"{row.start_date}--{row.end_date} ===",
                flush=True,
            )
            result = _evaluate_window(
                model,
                data,
                cfg,
                start_pos=int(row.start_pos),
                n_bootstrap=n_bootstrap,
                seed=int(row.null_seed),
            )
            alignment = base._alignment_rows(
                model,
                data,
                cfg,
                start_pos=int(row.start_pos),
                horizon=HORIZON,
                n_anchors=n_alignment_anchors,
                n_mc=n_alignment_mc,
                seed=int(row.null_seed),
            )
            metadata, score, component, alignment = _write_window(
                staging_dir,
                row,
                result,
                alignment,
                n_bootstrap=n_bootstrap,
            )
            metadata_rows.append(metadata)
            score_rows.append(score)
            components.append(component)
            alignments.append(alignment)
            print(
                pd.Series(
                    {
                        "global_p": score["global_rank_pvalue"],
                        "martingale_p": score["martingale_rank_pvalue"],
                        "bracket_p": score["bracket_rank_pvalue"],
                        "generator_p": score["generator_rank_pvalue"],
                        "dominant": score["dominant_metric"],
                    }
                ).to_string(),
                flush=True,
            )

    scores = pd.DataFrame(score_rows)
    scores.to_csv(staging_dir / "window_scores.csv", index=False)
    pd.concat(components, ignore_index=True).to_csv(
        staging_dir / "component_summaries.csv", index=False
    )
    alignment_all = pd.concat(alignments, ignore_index=True)
    alignment_all.to_csv(staging_dir / "finite_step_alignment_all.csv", index=False)

    extension = scores.loc[~scores["is_registered_anchor"]].copy()
    extension["distance_to_null_median"] = (
        extension["global_rank_pvalue"].astype(float) - 0.5
    ).abs()
    candidate = extension.sort_values(
        ["distance_to_null_median", "start_date", "country"],
        kind="stable",
    ).iloc[0]
    summary = {
        "n_windows": int(len(scores)),
        "n_registered_anchors": int(scores["is_registered_anchor"].sum()),
        "n_non_anchor_windows": int(len(extension)),
        "n_global_nonrejections_05_all": int(
            (scores["global_rank_pvalue"] >= 0.05).sum()
        ),
        "n_global_nonrejections_05_non_anchor": int(
            (extension["global_rank_pvalue"] >= 0.05).sum()
        ),
        "figure_candidate_rule": (
            "non-anchor window minimizing abs(global_rank_pvalue - 0.5), "
            "with start_date and country tie-breaks"
        ),
        "figure_candidate": {
            "country": str(candidate["country"]),
            "window_label": str(candidate["window_label"]),
            "start_date": str(candidate["start_date"]),
            "end_date": str(candidate["end_date"]),
            "global_rank_pvalue": float(candidate["global_rank_pvalue"]),
            "global_rejected_05": bool(candidate["global_rank_pvalue"] < 0.05),
            "dominant_metric": str(candidate["dominant_metric"]),
        },
        "dependence_note": (
            "Windows are repeated measurements within three countries; counts are "
            "descriptive and do not treat the 12 rows as independent replicates."
        ),
    }
    (staging_dir / "temporal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "countries": list(COUNTRIES),
        "window_labels": [label for label, _offset in WINDOW_SPECS],
        "n_bootstrap": int(n_bootstrap),
        "n_alignment_anchors": int(n_alignment_anchors),
        "n_alignment_mc": int(n_alignment_mc),
        "window_metadata": metadata_rows,
        "summary": summary,
    }
    (staging_dir / "run_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    staging_dir.replace(final_dir)
    print(json.dumps(summary, indent=2), flush=True)
    return final_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("freeze", "evaluate"),
        required=True,
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-alignment-anchors", type=int, default=8)
    parser.add_argument("--n-alignment-mc", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 4 or args.n_bootstrap % 2:
        raise ValueError("--n-bootstrap must be an even integer of at least four.")
    if args.n_alignment_anchors < 1 or args.n_alignment_mc < 8:
        raise ValueError("Alignment requires at least one anchor and eight MC paths.")
    if args.phase == "freeze":
        freeze(args.out_root, seed=args.seed, n_bootstrap=args.n_bootstrap)
    else:
        evaluate(
            args.out_root,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
            n_alignment_anchors=args.n_alignment_anchors,
            n_alignment_mc=args.n_alignment_mc,
        )


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
