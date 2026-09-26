"""Command-line entry point; legacy imports re-export the single implementations.

New code should import data, training and evaluation helpers from their modules.
"""

import argparse
import logging

from brownian_inversion import (
    compute_brownian_inversion_metrics,
)
from config import (
    Config,
    TrainingData,
    TrainingDataSet,
    get_cache_path,
    get_experiment_tag,
    get_model_save_path,
    get_recon_control_size,
)
from losses import loss as sde_loss
from sir_data import (
    OWID_MIRROR,
    OWID_URL,
    _explicit_recon_control_frame,
    _explicit_recon_control_vector_jax,
    _safe_lag_corr,
    _safe_lag_corr_jax,
    _select_incidence_array,
    _weekday_adjust_incidence,
    apply_temporal_prewhiten_frame,
    compute_norm_stats,
    create_sir_controls,
    download_owid_data,
    estimate_temporal_prewhiten_phi,
    fit_sir_baseline,
    get_oracle_targets,
    get_paths_from_dataframe,
    get_realized_drift_targets,
    get_sir_control_lookback,
    load_or_create_training_data,
    prepare_sir_countries,
)
from sir_evaluation import (
    forecast_sir,
    run_rolling_validation_sir,
    save_brownian_inversion_metrics,
    save_experiment_summary,
    summarize_experiment,
)
from sir_experiments import (
    main,
    run_ablation_suite,
    run_depth_ablation,
    run_depth_width_ablation,
    run_recovery_days_ablation,
    run_sigma_ablation_suite,
    run_signature_ablation_suite,
    run_special_experiments,
)
from sir_training import (
    assert_model_normalization,
    get_sir_optimizer,
    make_trainable_filter_spec,
    run_detailed_validation,
    validate_training_config,
)
from trajectory import get_signature_size
from visualization import (
    plot_brownian_inversion_validation,
    plot_rolling_validation_sir,
    plot_sir_forecast,
    plot_sir_parameters,
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    plt.style.use("seaborn-v0_8-darkgrid")
    parser = argparse.ArgumentParser(
        description="Train SIR Neural SDE on OWID COVID-19 data."
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download OWID data and rebuild training set.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if a saved model exists.",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=None,
        help="Override number of ensemble models.",
    )
    parser.add_argument(
        "--recovery-days",
        type=float,
        default=None,
        help="Override infectious period T_rec (days) for active-case estimation.",
    )
    parser.add_argument(
        "--incidence-source",
        choices=["smoothed", "raw", "weekday_adjusted", "blend_raw_smoothed"],
        default="smoothed",
        help="Incidence series used to reconstruct SIR active cases.",
    )
    parser.add_argument(
        "--incidence-blend-weight",
        type=float,
        default=0.0,
        help="Raw-incidence weight for --incidence-source blend_raw_smoothed.",
    )
    parser.add_argument(
        "--recovery-days-ablation",
        action="store_true",
        help="Run recovery-days experiments and save a comparison CSV.",
    )
    parser.add_argument(
        "--recovery-days-values",
        type=str,
        default="7,14,21,28",
        help="Comma-separated recovery-days values for --recovery-days-ablation.",
    )
    parser.add_argument(
        "--ablation-suite",
        action="store_true",
        help="Run none/zero/shuffle experiments and save a comparison CSV.",
    )
    parser.add_argument(
        "--signature-ablation-suite",
        action="store_true",
        help="Run depth/lead-lag/window ablation experiments.",
    )
    parser.add_argument(
        "--per-country-beta",
        action="store_true",
        help="Use per-country NLS β_base/γ_base instead of a single global fit.",
    )
    parser.add_argument(
        "--country-sigma",
        action="store_true",
        help="Enable learnable per-country scaling of the diffusion coefficient.",
    )
    parser.add_argument(
        "--state-dependent-sigma",
        action="store_true",
        help="Use a neural state/control-dependent diffusion coefficient.",
    )
    parser.add_argument(
        "--temporal-prewhiten",
        action="store_true",
        help="Apply AR(1) pre-whitening to signature-control paths.",
    )
    parser.add_argument(
        "--explicit-recon-controls",
        choices=["none", "acf_weekday"],
        default="none",
        help="Append causal reconstruction-history controls after signatures.",
    )
    parser.add_argument(
        "--explicit-control-window",
        type=int,
        default=20,
        help="Rolling window for explicit reconstruction-history controls.",
    )
    parser.add_argument(
        "--depth-ablation",
        action="store_true",
        help="Run depth-only ablation: depth in {1,2,3,4,5}.",
    )
    parser.add_argument(
        "--depth-width-ablation",
        action="store_true",
        help="Run depth×width factorial ablation: {3,4} × {32,64,128}.",
    )
    parser.add_argument(
        "--sigma-ablation",
        action="store_true",
        help="Sweep log_sigma init values [-4.0, -3.5, -3.0, -2.5, -2.0].",
    )
    parser.add_argument(
        "--special-experiments",
        action="store_true",
        help="Run freeze-γ+T_rec=28 and T_rec=28+d4_w64 experiments.",
    )
    parser.add_argument(
        "--zscore-wd-weight",
        type=float,
        default=0.0,
        help="Weight for 1D Wasserstein regularization of Brownian z-scores toward N(0,1).",
    )
    args = parser.parse_args()

    config = Config()
    if args.recovery_days is not None:
        config.SIR_RECOVERY_DAYS = float(args.recovery_days)
    config.SIR_INCIDENCE_SOURCE = args.incidence_source
    config.SIR_INCIDENCE_BLEND_WEIGHT = float(args.incidence_blend_weight)
    if args.per_country_beta:
        config.PER_COUNTRY_BETA_GAMMA = True
    if args.country_sigma:
        config.SIGMA_COUNTRY_SCALE = True
    if args.state_dependent_sigma:
        config.DIFFUSION_STATE_DEPENDENT = True
    if args.temporal_prewhiten:
        config.USE_TEMPORAL_PREWHITEN = True
    config.EXPLICIT_RECON_CONTROLS = args.explicit_recon_controls
    config.EXPLICIT_RECON_CONTROL_WINDOW = int(args.explicit_control_window)
    config.ZSCORE_WD_WEIGHT = float(args.zscore_wd_weight)

    recovery_days_values = [
        float(value.strip())
        for value in args.recovery_days_values.split(",")
        if value.strip()
    ]

    if args.recovery_days_ablation:
        run_recovery_days_ablation(
            config,
            recovery_days_values=recovery_days_values,
            force_data_refresh=args.refresh_data,
            force_train=args.force_train,
            num_models_override=args.num_models,
        )
    elif args.depth_width_ablation:
        run_depth_width_ablation(
            config,
            force_data_refresh=args.refresh_data,
            num_models_override=args.num_models,
        )
    elif args.depth_ablation:
        run_depth_ablation(
            config,
            force_data_refresh=args.refresh_data,
            num_models_override=args.num_models,
        )
    elif args.sigma_ablation:
        run_sigma_ablation_suite(
            config,
            force_data_refresh=args.refresh_data,
            force_train=args.force_train,
            num_models_override=args.num_models,
        )
    elif args.special_experiments:
        run_special_experiments(
            config,
            force_data_refresh=args.refresh_data,
            force_train=args.force_train,
            num_models_override=args.num_models,
        )
    elif args.signature_ablation_suite:
        run_signature_ablation_suite(
            config,
            force_data_refresh=args.refresh_data,
            force_train=args.force_train,
            num_models_override=args.num_models,
        )
    elif args.ablation_suite:
        run_ablation_suite(
            config,
            force_data_refresh=args.refresh_data,
            force_train=args.force_train,
            num_models_override=args.num_models,
        )
    else:
        main(
            config,
            force_data_refresh=args.refresh_data,
            force_train=args.force_train,
            num_models_override=args.num_models,
        )
