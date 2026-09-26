"""Thirty-increment split-window sensitivity audit for Nordic SDE diagnostics.

Each of the twelve frozen 60-increment temporal-audit blocks is split into
two adjacent 30-increment halves.  The resulting 24 windows reuse the parent
block's fitted-null seed and the corresponding first/second 30-day innovation
slice from a 60-day random bank.  Each half is nevertheless simulated from
its own observed conditioning history, as required for a conditional audit.

This is a post-hoc horizon/power sensitivity analysis.  A short-window
non-rejection is not a new confirmation of model adequacy.
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


from experiments import (
    run_sir_sde_native_country_audit as base,
    run_sir_sde_native_temporal_audit as temporal,
)
from experiments._bi_audit_common import load_cached_data


COUNTRIES = base.CONFIRMATORY_COUNTRIES
HORIZON = 30
PARENT_HORIZON = 60
DEFAULT_SEED = temporal.DEFAULT_SEED
DEFAULT_N_BOOTSTRAP = 5000
PARENT_ROOT = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_temporal_audit"
    / "nordic_four_block_B5000_frozen"
)
PARENT_OUTPUT_DIR = PARENT_ROOT / "temporal_windows"
DEFAULT_OUT_ROOT = (
    ROOT
    / "cache"
    / "summaries"
    / "sde_native_short_window_audit"
    / "nordic_split30_B5000_frozen"
)
SCHEDULE_NAME = "short_window_schedule.csv"


def _relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def _build_schedule(*, seed: int, n_bootstrap: int) -> pd.DataFrame:
    if seed != DEFAULT_SEED or n_bootstrap != DEFAULT_N_BOOTSTRAP:
        raise ValueError(
            "The split-window protocol is tied to the frozen B=5000 parent bank."
        )
    parent_schedule = temporal._load_schedule(PARENT_ROOT, seed=seed)
    rows: list[dict[str, object]] = []
    for country in COUNTRIES:
        cfg = base._config(country)
        data = load_cached_data(cfg)
        country_parents = parent_schedule.loc[
            parent_schedule["country"].eq(country)
        ].sort_values("window_order")
        for parent in country_parents.itertuples(index=False):
            for half_order, (half_label, noise_start) in enumerate(
                (("a", 0), ("b", HORIZON))
            ):
                start_pos = int(parent.start_pos) + noise_start
                end_pos = start_pos + HORIZON
                rows.append(
                    {
                        "country": country,
                        "window_order": int(parent.window_order) * 2 + half_order,
                        "window_label": f"{parent.window_label}_{half_label}",
                        "parent_window_label": str(parent.window_label),
                        "parent_offset_from_registered_start": int(
                            parent.offset_from_registered_start
                        ),
                        "half": half_label,
                        "parent_is_registered_anchor": bool(
                            parent.is_registered_anchor
                        ),
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "start_date": str(
                            data.val_features_df.index[start_pos].date()
                        ),
                        "end_date": str(data.val_features_df.index[end_pos].date()),
                        "horizon_increments": HORIZON,
                        "parent_null_seed": int(parent.null_seed),
                        "parent_noise_slice_start": noise_start,
                        "parent_noise_slice_end": noise_start + HORIZON,
                    }
                )
    schedule = pd.DataFrame(rows)
    for country, group in schedule.groupby("country", sort=False):
        ordered = group.sort_values("start_pos")
        starts = ordered["start_pos"].to_numpy(dtype=int)
        ends = ordered["end_pos"].to_numpy(dtype=int)
        if np.any(starts[1:] < ends[:-1]):
            raise ValueError(f"Short-window increment sets overlap for {country}.")
        if not np.all(starts[1:] == ends[:-1]):
            raise ValueError(f"Short windows do not tile the parent blocks for {country}.")
    return schedule


def freeze(out_root: Path, *, seed: int, n_bootstrap: int) -> Path:
    schedule_path = out_root / SCHEDULE_NAME
    if schedule_path.exists():
        raise FileExistsError(f"Short-window schedule already exists under {out_root}.")
    if (out_root / "short_windows").exists() or (
        out_root / "_short_windows_staging"
    ).exists():
        raise RuntimeError("Refusing to freeze beside short-window outputs.")
    schedule = _build_schedule(seed=seed, n_bootstrap=n_bootstrap)
    out_root.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(schedule_path, index=False)
    print(f"Saved short-window schedule: {schedule_path}", flush=True)
    print(schedule.to_string(index=False), flush=True)
    return schedule_path


def _load_schedule(
    out_root: Path, *, seed: int, n_bootstrap: int
) -> pd.DataFrame:
    schedule_path = out_root / SCHEDULE_NAME
    if not schedule_path.exists():
        raise FileNotFoundError("Short-window evaluation requires short_window_schedule.csv.")
    schedule = pd.read_csv(schedule_path)
    expected = _build_schedule(seed=seed, n_bootstrap=n_bootstrap)
    pd.testing.assert_frame_equal(schedule, expected, check_dtype=False)
    return schedule


def _evaluate_short_window(
    model,
    data,
    cfg,
    *,
    start_pos: int,
    noise: jax.Array,
) -> dict[str, object]:
    context = base._make_context(model, data, cfg)
    n_bootstrap = int(noise.shape[0])
    expected_shape = (n_bootstrap, HORIZON, int(context["n_substeps"]))
    if tuple(noise.shape) != expected_shape:
        raise ValueError(f"Unexpected short-window noise shape {noise.shape}.")
    if start_pos + HORIZON >= len(data.val_features_df):
        raise ValueError("Short window exceeds the observed trajectory.")
    raw_lookback = int(context["raw_lookback"])
    if start_pos - raw_lookback + 1 < 0:
        raise ValueError("Short window lacks its causal control lookback.")

    initial_history = jnp.asarray(
        data.val_features_df.iloc[
            start_pos - raw_lookback + 1 : start_pos + 1
        ].values
    )
    observed_future = jnp.asarray(
        data.val_features_df.iloc[start_pos + 1 : start_pos + HORIZON + 1].values
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
    def simulate_batch(batch_noise):
        return jax.vmap(
            lambda one_noise: simulate_one(
                initial_history, one_noise, start_weekday
            )
        )(batch_noise)

    @eqx.filter_jit
    def evaluate_batch(futures):
        return jax.vmap(
            lambda future: evaluate_future(
                initial_history, future, start_weekday
            )
        )(futures)

    observed_outputs = evaluate_observed(observed_future)
    simulated_future, clipped = simulate_batch(noise)
    null_outputs = evaluate_batch(simulated_future)

    observed_z = np.asarray(observed_outputs[0])
    observed_dynkin = np.asarray(observed_outputs[1])
    null_z = np.asarray(null_outputs[0])
    null_dynkin = np.asarray(null_outputs[1])
    clip_fraction = np.asarray(clipped, dtype=float) / float(
        HORIZON * int(context["n_substeps"])
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
        "end_date": str(
            data.val_features_df.index[start_pos + HORIZON].date()
        ),
        "horizon": HORIZON,
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


def _verify_observed_parent_slice(row, result: dict[str, object]) -> None:
    parent_path = (
        PARENT_OUTPUT_DIR
        / str(row.country)
        / str(row.parent_window_label)
        / "observed_path_diagnostics.npz"
    )
    start = int(row.parent_noise_slice_start)
    stop = int(row.parent_noise_slice_end)
    with np.load(parent_path) as parent:
        checks = {
            "observed_z_I": parent["observed_z_I"][start:stop],
            "observed_dynkin_z": parent["observed_dynkin_z"][start:stop],
        }
    for key, expected in checks.items():
        current = np.asarray(result["raw"][key])
        if not np.allclose(current, expected, rtol=0.0, atol=1e-8):
            max_diff = float(np.max(np.abs(current - expected)))
            raise ValueError(
                f"Observed parent-slice identity failed for {row.country}/"
                f"{row.window_label}/{key}; max diff={max_diff}"
            )


def _write_window(
    phase_dir: Path,
    row,
    result: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    country = str(row.country)
    label = str(row.window_label)
    window_dir = phase_dir / country / label
    window_dir.mkdir(parents=True, exist_ok=False)
    ids = {
        "country": country,
        "window_label": label,
        "parent_window_label": str(row.parent_window_label),
        "half": str(row.half),
        "parent_is_registered_anchor": bool(row.parent_is_registered_anchor),
        "start_date": str(row.start_date),
        "end_date": str(row.end_date),
    }
    draws = result["draws"].copy()
    for name, value in reversed(list(ids.items())):
        draws.insert(0, name, value)
    draws.to_csv(window_dir / "null_metric_draws.csv", index=False)
    component = result["component_summary"].copy()
    for name, value in reversed(list(ids.items())):
        component.insert(0, name, value)
    component.to_csv(window_dir / "component_summary.csv", index=False)
    score = {**ids, **result["score"]}
    pd.DataFrame([score]).to_csv(
        window_dir / "layer_and_global_scores.csv", index=False
    )
    np.savez(window_dir / "observed_path_diagnostics.npz", **result["raw"])
    metadata = {
        **ids,
        "start_pos": int(row.start_pos),
        "end_pos": int(row.end_pos),
        "horizon_increments": HORIZON,
        "n_bootstrap": int(len(draws)),
        "parent_null_seed": int(row.parent_null_seed),
        "parent_noise_slice": [
            int(row.parent_noise_slice_start),
            int(row.parent_noise_slice_end),
        ],
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
    return metadata, score, component


def evaluate(out_root: Path, *, seed: int, n_bootstrap: int) -> Path:
    schedule = _load_schedule(out_root, seed=seed, n_bootstrap=n_bootstrap)
    final_dir = out_root / "short_windows"
    staging_dir = out_root / "_short_windows_staging"
    if final_dir.exists():
        raise FileExistsError(f"Short-window outputs already exist: {final_dir}")
    if staging_dir.exists():
        raise FileExistsError(
            f"A prior short-window staging directory exists: {staging_dir}"
        )
    staging_dir.mkdir(parents=True)
    metadata_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    components: list[pd.DataFrame] = []
    for country in COUNTRIES:
        cfg = base._config(country)
        base.bi._validate_supported_config(cfg)
        data = load_cached_data(cfg)
        model = base._load_common_original9_model(cfg, data)
        country_rows = schedule.loc[schedule["country"].eq(country)]
        for parent_label, parent_rows in country_rows.groupby(
            "parent_window_label", sort=False
        ):
            parent_rows = parent_rows.sort_values("window_order")
            parent_seed = int(parent_rows.iloc[0]["parent_null_seed"])
            context = base._make_context(model, data, cfg)
            full_noise = jax.random.normal(
                jax.random.PRNGKey(parent_seed),
                shape=(n_bootstrap, PARENT_HORIZON, int(context["n_substeps"])),
            )
            for row in parent_rows.itertuples(index=False):
                print(
                    f"=== short SDE audit: {country}/{row.window_label} "
                    f"{row.start_date}--{row.end_date} ===",
                    flush=True,
                )
                noise = full_noise[
                    :,
                    int(row.parent_noise_slice_start) : int(
                        row.parent_noise_slice_end
                    ),
                    :,
                ]
                result = _evaluate_short_window(
                    model,
                    data,
                    cfg,
                    start_pos=int(row.start_pos),
                    noise=noise,
                )
                _verify_observed_parent_slice(row, result)
                metadata, score, component = _write_window(
                    staging_dir,
                    row,
                    result,
                )
                metadata_rows.append(metadata)
                score_rows.append(score)
                components.append(component)
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
    scores["distance_to_null_median"] = (
        scores["global_rank_pvalue"].astype(float) - 0.5
    ).abs()
    nonrejected = scores.loc[scores["global_rank_pvalue"] >= 0.05]
    eligible = nonrejected if not nonrejected.empty else scores
    representative = eligible.sort_values(
        ["distance_to_null_median", "start_date", "country"], kind="stable"
    ).iloc[0]
    summary = {
        "n_windows": int(len(scores)),
        "n_global_rejections_05": int(
            (scores["global_rank_pvalue"] < 0.05).sum()
        ),
        "n_bracket_rejections_05": int(
            (scores["bracket_rank_pvalue"] < 0.05).sum()
        ),
        "n_martingale_rejections_05": int(
            (scores["martingale_rank_pvalue"] < 0.05).sum()
        ),
        "n_generator_rejections_05": int(
            (scores["generator_rank_pvalue"] < 0.05).sum()
        ),
        "representative_rule": (
            "among global non-rejections choose p closest to 0.5; if none, "
            "choose least-extreme global rank"
        ),
        "representative_window": {
            "country": str(representative["country"]),
            "window_label": str(representative["window_label"]),
            "parent_window_label": str(
                representative["parent_window_label"]
            ),
            "start_date": str(representative["start_date"]),
            "end_date": str(representative["end_date"]),
            "global_rank_pvalue": float(
                representative["global_rank_pvalue"]
            ),
            "global_rejected_05": bool(
                representative["global_rank_pvalue"] < 0.05
            ),
            "dominant_metric": str(representative["dominant_metric"]),
        },
        "interpretation": (
            "post-hoc horizon sensitivity; non-rejection can reflect lower "
            "30-day power and is not confirmatory adequacy"
        ),
    }
    (staging_dir / "short_window_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
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
    parser.add_argument("--phase", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap != DEFAULT_N_BOOTSTRAP:
        raise ValueError("The frozen short-window protocol requires B=5000.")
    if args.seed != DEFAULT_SEED:
        raise ValueError("The frozen short-window protocol requires seed=20260806.")
    if args.phase == "freeze":
        freeze(args.out_root, seed=args.seed, n_bootstrap=args.n_bootstrap)
    else:
        evaluate(args.out_root, seed=args.seed, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
