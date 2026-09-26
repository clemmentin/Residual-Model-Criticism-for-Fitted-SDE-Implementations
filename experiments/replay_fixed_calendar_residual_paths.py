"""Recover the already-reported CZE/GRC residual and log-I paths for figures.

Reuses the original fit, reconstruction, window and 5,000-path seeds. The
replayed metric banks must reproduce the saved results before paths are saved.
No fitting, target selection or new test is performed.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import equinox as eqx
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]

from experiments import run_fixed_calendar_holdout as holdout
from experiments import run_sir_sde_native_country_audit as native


OUT = ROOT / "output/sir_figure_data/residuals"
SOURCES = {
    "CZE": ("causal_window", "holdout"),
    "GRC": ("second_holdout", "second_holdout"),
}


def recover_log_i(model, data, cfg, start: int, seed: int, n_paths: int, expected_z):
    """Replay states with the original simulator, leaving the frozen runner intact."""
    context = native._make_context(model, data, cfg)
    lookback = int(context["raw_lookback"])
    history = jnp.asarray(data.val_features_df.iloc[start-lookback+1:start+1].values)
    weekday = jnp.array(int(data.val_features_df.index[start].weekday()), dtype=jnp.int32)

    @eqx.filter_jit
    def simulate(noise):
        return jax.vmap(lambda one: context["simulate_one"](history, one, weekday))(noise)[0]

    @eqx.filter_jit
    def residuals(futures):
        return jax.vmap(lambda one: context["evaluate_future"](history, one, weekday)[0])(futures)

    noise = jax.random.normal(jax.random.PRNGKey(seed),
                              shape=(n_paths, 60, int(context["n_substeps"])))
    future = simulate(noise)
    np.testing.assert_allclose(np.asarray(residuals(future)), expected_z, rtol=0, atol=1e-10)
    mean, scale = float(data.norm_mean[1]), float(data.norm_std[1])
    observed = data.val_features_df.iloc[start:start+61, 1].to_numpy(float)*scale+mean
    reference = np.asarray(future, dtype=float)[1::2, :, 1]*scale+mean
    reference = np.column_stack([np.full(len(reference), observed[0]), reference])
    assert observed.shape == (61,) and reference.shape == (2500, 61)
    return observed, reference


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = holdout._read_raw()
    summaries = []
    for country, (folder, prefix) in SOURCES.items():
        source = ROOT / "cache/summaries/fixed_calendar_holdout" / folder
        recorded = pd.read_csv(source / f"{prefix}_results.csv").iloc[0]
        bank = pd.read_csv(source / f"{prefix}_reference_draws.csv")
        cfg = holdout._config(country)
        data = holdout._build_lightweight_data(raw, cfg, country)
        start, reason = holdout._fixed_start_position(data)
        if start is None:
            raise ValueError(reason)
        model = native._load_common_original9_model(cfg, data)
        seed = int(recorded["null_bank_seed"])
        print(f"Replaying {country}: seed={seed}, {len(bank)} paths", flush=True)
        result = native._evaluate_country_paths(
            model, data, cfg, n_bootstrap=len(bank), seed=seed,
            start_pos=start, horizon=60, retain_null_z=True,
        )
        metric_errors = {}
        for metric in native.METRIC_SPECS:
            expected = bank[metric].to_numpy(float)
            actual = result["draws"][metric].to_numpy(float)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                result["observed_metrics"][metric], recorded[f"observed_{metric}"],
                rtol=0, atol=1e-10,
            )
            metric_errors[metric] = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(
            result["score"]["global_rank_pvalue"], recorded["global_rank_pvalue"],
            rtol=0, atol=1e-12,
        )
        observed = np.asarray(result["raw"]["observed_z_I"], dtype=float)
        reference = np.asarray(result["raw"]["null_z_I"], dtype=float)[1::2]
        assert observed.shape == (60,) and reference.shape == (2500, 60)
        # Verify against the previously plotted paths before adding state arrays.
        cache_path = OUT / f"{country}_residual_paths.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as old:
                np.testing.assert_allclose(observed, old["observed_z_I"], rtol=0, atol=1e-10)
                np.testing.assert_allclose(reference, old["evaluation_z_I"], rtol=0, atol=1e-10)
        observed_log_i, reference_log_i = recover_log_i(
            model, data, cfg, start, seed, len(bank), result["raw"]["null_z_I"],
        )
        state_dates = data.val_features_df.index[start:start+61].strftime("%Y-%m-%d").to_numpy(dtype="U10")
        assert state_dates[0] == "2021-01-01" and state_dates[-1] == "2021-03-02"
        np.savez_compressed(
            cache_path,
            observed_z_I=observed, evaluation_z_I=reference,
            dates=np.asarray(result["dates"].astype(str), dtype="U10"),
            observed_log_I=observed_log_i, evaluation_log_I=reference_log_i,
            state_dates=state_dates,
        )
        summary = {
            "country": country, "seed": seed, "pilot_n": 2500,
            "evaluation_n": 2500, "reference_subset": "odd bootstrap_id",
            "source": str(source.relative_to(ROOT)).replace("\\", "/"),
            "max_metric_bank_error": max(metric_errors.values()),
            "global_rank_pvalue": float(result["score"]["global_rank_pvalue"]),
            "state_variable": "log(I), after reversing training normalization",
            "state_dates": [str(state_dates[0]), str(state_dates[-1])],
            "state_points": len(state_dates),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (OUT / "replay_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
