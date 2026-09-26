"""Formal corrected SIR audit using a finite-step tangent-covariance map.

The existing registered six-component score is retained.  What changes is the
fixed path-to-coordinate map: the scalar log-I residual is divided by the
finite-step marginal variance obtained from the full 2x2 tangent covariance
recursion

    Sigma_{l+1} = F_l Sigma_l F_l.T + h g_l g_l.T,
    F_l = I + h J_b(y_l, c).

This is the geometry-aware corrected audit, not a retraining step and not a
new registered confirmation.  The old deterministic plug-in map remains in
the archived registered and corrected-control outputs for comparison.
"""

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

import config as config_module
from experiments import run_sir_bi_bootstrap as bi, run_sir_tangent_map_comparison as tangent
from experiments.country_bi_scoring import build_country_rows


BASE_COUNTRIES = ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "CHE", "GBR"]
COUNTRIES = ("FIN", "NOR", "SWE")
CORRECTED_MODEL_ROOT = ROOT / "models" / "country_cv" / "corrected_fixednorm_matchedsolver"
DEFAULT_OUT_DIR = ROOT / "cache" / "summaries" / "corrected_country_bi" / "tangent_geometry"


def _run_country(
    val_country: str,
    n_bootstrap: int,
    seed: int,
    out_root: Path,
    model_root: Path,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, float]]:
    cfg = tangent._config(val_country, model_root)
    bi._validate_supported_config(cfg)
    data = tangent.load_cached_data(cfg)
    model, model_path = bi._load_model(cfg, data)
    start_pos = bi._start_position(data, cfg)
    start_date = str(data.val_features_df.index[start_pos].date())
    horizon = min(cfg.FORECAST_HORIZON, len(data.val_features_df) - 1 - start_pos)
    end_date = str(data.val_features_df.index[start_pos + horizon].date())
    experiment = config_module.get_experiment_tag(cfg)

    # The imported implementation returns both maps from the same path bank.
    # Only the tangent-I coordinate is used for the formal geometry-aware audit.
    observed_plugin, observed_tangent, obs_cov01, obs_cov11 = tangent._observed_residuals(
        model, data, cfg, start_pos
    )
    (
        plugin_raw,
        tangent_raw,
        simulated_cov01,
        simulated_cov11,
        clip_fraction,
    ) = tangent._simulated_residuals(
        model, data, cfg, start_pos, n_bootstrap, seed
    )

    observed_tangent_metrics = bi.compute_brownian_inversion_metrics_from_residuals(
        observed_tangent
    )
    geometry_case = "recov28_frzgamma_tangent_geometry"
    geometry_draws = tangent._metric_draws(
        geometry_case,
        val_country,
        experiment,
        start_date,
        tangent_raw,
        clip_fraction,
    )
    geometry_summary = tangent._metric_rows(
        geometry_case,
        val_country,
        experiment,
        start_date,
        observed_tangent_metrics,
        geometry_draws,
    )
    geometry_global, geometry_profile = build_country_rows(
        geometry_summary, geometry_draws
    )

    # Keep a paired old-map number in the metadata, but do not use it for the
    # geometry-aware score or its rank.
    plugin_metrics = bi.compute_brownian_inversion_metrics_from_residuals(observed_plugin)
    plugin_draws = tangent._metric_draws(
        "recov28_frzgamma_plugin_control",
        val_country,
        experiment,
        start_date,
        plugin_raw,
        clip_fraction,
    )
    plugin_summary = tangent._metric_rows(
        "recov28_frzgamma_plugin_control",
        val_country,
        experiment,
        start_date,
        plugin_metrics,
        plugin_draws,
    )
    plugin_global, _ = build_country_rows(plugin_summary, plugin_draws)

    out_dir = out_root / f"recov28_frzgamma_val{val_country}"
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry_summary.to_csv(out_dir / "geometry_bi_summary.csv", index=False)
    geometry_draws.to_csv(out_dir / "geometry_bi_draws.csv", index=False)
    pd.DataFrame([geometry_global]).to_csv(
        out_dir / "geometry_bi_global_score.csv", index=False
    )
    pd.DataFrame([geometry_profile]).to_csv(
        out_dir / "geometry_bi_max_adjusted_profile.csv", index=False
    )
    np.savez(
        out_dir / "geometry_raw_diagnostics.npz",
        observed_tangent_I=np.asarray(observed_tangent),
        observed_plugin_I=np.asarray(observed_plugin),
        observed_cov_SI=np.asarray(obs_cov01),
        observed_cov_II=np.asarray(obs_cov11),
        simulated_cov_SI=np.asarray(simulated_cov01),
        simulated_cov_II=np.asarray(simulated_cov11),
    )

    metadata = {
        "role": "post-hoc geometry-aware corrected SIR audit; not registered confirmation",
        "case": "recov28_frzgamma",
        "val_country": val_country,
        "experiment": experiment,
        "start_date": start_date,
        "end_date": end_date,
        "audit_horizon_increments": int(horizon),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "model_path": str(model_path),
        "map_name": "finite_step_tangent_covariance_I_marginal",
        "map_definition": {
            "recursion": "Sigma[l+1] = F[l] Sigma[l] F[l].T + h g[l] g[l].T",
            "local_drift_map": "F[l] = I + h J_b(y[l], c[l])",
            "initial_covariance": "Sigma[0] = 0",
            "primary_coordinate": "z_I = (Delta Y_I - m_I) / sqrt(Sigma_II)",
            "full_covariance_recursion_used": True,
            "full_vector_whitening_used": False,
            "covariance_role": "finite-step marginal geometry for the audited log-I coordinate",
            "tangent_covariance_is_exact_conditional_moment": False,
        },
        "conditioning_and_solver": {
            "substep_dt": float(cfg.SDE_SUBSTEP_DT),
            "num_substeps": int(cfg.SDE_NUM_SUBSTEPS),
            "controls_recomputed_on_each_null_path": True,
            "projection_each_null_substep": True,
        },
        "observed_metrics": observed_tangent_metrics,
        "plugin_control_comparison": {
            "observed_smax": float(plugin_global["observed_smax"]),
            "smax_empirical_pvalue": float(plugin_global["smax_empirical_pvalue"]),
            "dominant_metric": str(plugin_global["dominant_metric"]),
        },
        "geometry_diagnostics": {
            "mean_simulated_cov_SI": float(np.mean(simulated_cov01)),
            "mean_simulated_cov_II": float(np.mean(simulated_cov11)),
            "mean_observed_cov_SI": float(np.mean(obs_cov01)),
            "mean_observed_cov_II": float(np.mean(obs_cov11)),
            "mean_clip_fraction": float(np.mean(clip_fraction)),
            "max_clip_fraction": float(np.max(clip_fraction)),
        },
    }
    (out_dir / "geometry_bi_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    row = {
        "val_country": val_country,
        "map_name": metadata["map_name"],
        "geometry_observed_smax": float(geometry_global["observed_smax"]),
        "geometry_smax_empirical_pvalue": float(
            geometry_global["smax_empirical_pvalue"]
        ),
        "geometry_dominant_metric": str(geometry_global["dominant_metric"]),
        "plugin_observed_smax": float(plugin_global["observed_smax"]),
        "plugin_smax_empirical_pvalue": float(
            plugin_global["smax_empirical_pvalue"]
        ),
        "plugin_dominant_metric": str(plugin_global["dominant_metric"]),
        "mean_simulated_cov_SI": float(np.mean(simulated_cov01)),
        "mean_simulated_cov_II": float(np.mean(simulated_cov11)),
        "mean_clip_fraction": float(np.mean(clip_fraction)),
    }
    return row, geometry_summary, geometry_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", nargs="+", default=list(COUNTRIES), choices=COUNTRIES)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-root", type=Path, default=CORRECTED_MODEL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2")

    rows = []
    summaries = []
    profiles = []
    for country in args.val:
        print(f"=== geometry-aware corrected audit: {country} ===", flush=True)
        row, summary, profile = _run_country(
            val_country=country.upper(),
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            out_root=args.out_dir,
            model_root=args.model_root,
        )
        rows.append(row)
        summaries.append(summary)
        profiles.append(profile)
        print(pd.Series(row).to_string(), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.out_dir / "geometry_comparison.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.out_dir / "geometry_component_summary.csv", index=False
    )
    pd.DataFrame([dict(x) for x in profiles]).to_csv(
        args.out_dir / "geometry_max_adjusted_profile.csv", index=False
    )

    print()
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[1])
    main()
