"""Checks for failures found when running the standalone source release."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import jax.numpy as jnp

from experiments import bi_synthetic_full_refit_internal_pvalue_calibration as calibration
from experiments import run_linear_geometry_mechanism as linear
from experiments import run_fixed_calendar_holdout as holdout
from experiments import run_sir_trec_gamma_decoupling as decoupling
from experiments import summarize_full_refit_calibration as reporting
from experiments import run_sir_swd_kernel_diagnostic as swd
import config as config_module
import losses
import sir_data


class ReleaseRegressionTests(unittest.TestCase):
    def test_linear_runner_creates_output_directory_and_writes_paired_trials(self):
        design = json.loads(linear.DESIGN.read_text(encoding="utf-8"))
        design["model"].update(kappa_grid=[0.0, 0.5], L_grid=[1, 2])
        design["alternatives"] = design["alternatives"][:2]
        design["monte_carlo"].update(outer_replicates=4, reference_paths_per_replicate=9, block_size=2)
        design["analysis"]["primary_family_size"] = 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path = root / "design.json"
            design_path.write_text(json.dumps(design), encoding="utf-8")
            output = root / "not_created" / "results"
            with patch.object(linear, "DESIGN", design_path), patch.object(linear, "OUT", output):
                linear.run()
            trials = pd.read_csv(output / "trials.csv")
            self.assertEqual(len(trials), 32)
            self.assertEqual(len(pd.read_csv(output / "summary.csv")), 8)
            control = trials.loc[trials.kappa.eq(0) | trials.L.eq(1)]
            np.testing.assert_array_equal(control.p_instantaneous, control.p_finite_step)

    @staticmethod
    def raw_sir():
        dates = pd.date_range("2020-03-01", periods=100)
        cases = np.r_[np.zeros(10), np.linspace(10, 50, 90)]
        return pd.concat([
            pd.DataFrame({"iso_code": country, "date": dates, "population": 1_000_000,
                          "total_cases": np.cumsum(cases), "new_cases_smoothed": cases})
            for country in (*holdout.native.DEVELOPMENT_COUNTRIES, "CZE", "GRC")
        ], ignore_index=True)

    def test_fixed_calendar_data_keeps_inactive_dates_for_both_holdouts(self):
        raw = self.raw_sir()
        for country in ("CZE", "GRC"):
            with self.subTest(country=country):
                config = holdout._config(country)
                config.TRAINING_START_DATE = "2020-03-01"
                data = holdout._build_lightweight_data(raw, config, country)
                self.assertEqual(len(data.val_features_df), 100)
                self.assertEqual(data.val_features_df.index[0], pd.Timestamp("2020-03-01"))

    def test_holdout_anchor_cannot_use_a_future_positive_total(self):
        result = sir_data.prepare_sir_countries(
            self.raw_sir(), ["CZE"], "2020-03-01",
            trim_to_active_period=False, anchor_cutoff="2020-03-05",
        )
        self.assertEqual(result, {})

    def test_country_evaluation_respects_explicit_calendar_window(self):
        frame = pd.DataFrame({"S": np.sin(np.arange(80) / 7), "I": np.cos(np.arange(80) / 9)},
                             index=pd.date_range("2021-01-01", periods=80))

        def evaluate_future(history, future, weekday):
            return (future[:, 0], future[:, 1:2], future, future, future, future, future)

        def simulate_one(history, noise, weekday):
            return jnp.column_stack((noise[:, 0], noise[:, 0] ** 2)), jnp.array(0)

        native = holdout.native
        context = {"raw_lookback": 3, "n_substeps": 1,
                   "evaluate_future": evaluate_future, "simulate_one": simulate_one}
        with patch.object(native, "_make_context", return_value=context), \
                patch.object(native.bi, "_start_position", side_effect=AssertionError("Unexpected midpoint lookup")):
            result = native._evaluate_country_paths(
                None, SimpleNamespace(val_features_df=frame), config_module.Config(),
                n_bootstrap=8, seed=123, start_pos=20, horizon=12)
            self.assertEqual(result["start_date"], "2021-01-21")
            self.assertEqual(result["end_date"], "2021-02-02")
            self.assertEqual(result["horizon"], 12)
            np.testing.assert_allclose(result["raw"]["observed_z_I"], frame.S.iloc[21:33], atol=1e-7)
            with self.assertRaises(ValueError):
                native._evaluate_country_paths(
                    None, SimpleNamespace(val_features_df=frame), config_module.Config(),
                    n_bootstrap=8, seed=123, start_pos=70, horizon=12)

    def test_default_active_period_and_causal_prefix_are_preserved(self):
        raw = self.raw_sir()
        kwargs = dict(countries=["CZE"], start_date="2020-03-01", recovery_days=28)
        active = sir_data.prepare_sir_countries(raw, **kwargs)["CZE"]
        full = sir_data.prepare_sir_countries(raw, **kwargs, trim_to_active_period=False,
                                              anchor_cutoff="2020-03-31")["CZE"]
        pd.testing.assert_frame_equal(active, full.loc[active.index])
        changed = raw.copy()
        changed.loc[changed.date.gt("2020-05-15"), "new_cases_smoothed"] *= 100
        rebuilt = sir_data.prepare_sir_countries(changed, **kwargs, trim_to_active_period=False,
                                                 anchor_cutoff="2020-03-31")["CZE"]
        pd.testing.assert_frame_equal(full.loc[:"2020-05-15"], rebuilt.loc[:"2020-05-15"])

    def test_discrete_ks_counts_equal_distances_as_ties(self):
        # For n=5 and grid=10, every possible sample has KS distance >= 1/10.
        # Float CDF subtraction used to represent this same distance unevenly.
        observed = calibration._discrete_uniform_ks_stat(np.array([1, 3, 5, 7, 9]) / 10, 10)
        self.assertEqual(observed, 0.1)
        for batch_size in (37, 1000):
            self.assertEqual(calibration._discrete_uniform_ks_mc_pvalue(
                observed, 5, 10, 1000, 123, batch_size=batch_size), 1.0)

    def test_ks_rejects_nonfinite_or_off_grid_inputs(self):
        for values in ([np.nan], [np.inf], [0.125], [0.0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                calibration._discrete_uniform_ks_stat(np.array(values), 10)
        with self.assertRaises(ValueError):
            calibration._discrete_uniform_ks_mc_pvalue(0.1, 5, 10, 10, 123, batch_size=0)

    def test_checkpoint_preserves_non_decimal_rank_grid(self):
        ranks = np.array([1, 2, 7]) / 13
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            calibration._append_csv_checkpoint(pd.DataFrame({"p": ranks}), path, reset=True)
            saved = pd.read_csv(path).p.to_numpy()
            np.testing.assert_allclose(saved, ranks, rtol=0, atol=1e-15)
            self.assertAlmostEqual(calibration._discrete_uniform_ks_stat(saved, 13),
                                   calibration._discrete_uniform_ks_stat(ranks, 13))

    def test_retained_results_rebuild_tables_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem = "bi_synthetic_full_refit_internal_pvalue"
            pd.DataFrame({"pvalue": np.array([1, 3, 5, 7, 9]) / 10}).to_csv(
                root / f"{stem}_ecdf.csv", index=False)
            components_name = "bi_synthetic_full_refit_component_pvalue_calibration.csv"
            components = pd.DataFrame([
                {"column": f"p_{metric}_internal", "component": label, "n": 5, "grid": 10,
                 "dks_stat": 0.1, "dks_mc_pvalue": 0.9, "size_05": 0.0}
                for metric, label in calibration.COMPONENT_PVALUE_ORDER
            ])
            components.to_csv(root / components_name, index=False)
            original = (root / components_name).read_bytes()
            reporting.summarize(root, root / "rebuilt", 64, 123)
            result = pd.read_csv(root / "rebuilt" / components_name)
            np.testing.assert_array_equal(result.dks_mc_pvalue, np.ones(6))
            self.assertTrue(result.size_05.eq(0).all())
            alpha = pd.read_csv(root / "rebuilt" / f"{stem}_alpha_table.csv")
            self.assertEqual(int(alpha.loc[alpha.alpha.eq(0.05), "reject_count"].iloc[0]), 0)
            self.assertEqual((root / components_name).read_bytes(), original)

    def test_nonfinite_statistics_cannot_become_significant_ranks(self):
        columns = list(calibration.COMPONENT_METRICS)
        bank = pd.DataFrame(np.arange(18).reshape(3, 6), columns=columns, dtype=float)
        for source in ("observed", "pilot", "evaluation"):
            with self.subTest(source=source):
                observed = dict.fromkeys(columns, 0.0)
                pilot = bank.copy()
                evaluation = bank.copy()
                if source == "observed":
                    observed[columns[0]] = np.nan
                elif source == "pilot":
                    pilot.iloc[0, 0] = np.nan
                else:
                    evaluation.iloc[0, 0] = np.nan
                with patch.object(calibration, "residuals_for_path", return_value=np.zeros(12)), \
                        patch.object(calibration, "simulate_fitted_null_residuals", return_value=np.zeros((3, 12))), \
                        patch.object(calibration, "diagnostic_components", return_value=observed), \
                        patch.object(calibration, "diagnostics_matrix", side_effect=[pilot, evaluation]), \
                        self.assertRaises(FloatingPointError):
                    calibration._split_bank_audit({}, np.zeros(12), np.random.default_rng(1), 3, 3)

    def test_resume_requires_matching_row_and_metadata_seeds(self):
        config = calibration.InternalPvalueConfig()
        trials = pd.DataFrame({"replicate": [0, 1], "outer_seed": [1234, 1235]})
        with tempfile.TemporaryDirectory() as directory:
            paths = calibration._output_paths(Path(directory), "matched_null")
            metadata = {"config": asdict(config), "run": {"seed": 1234}}
            paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
            calibration._validate_existing_config(paths, trials, config, 1234)
            with self.assertRaisesRegex(ValueError, "seed"):
                calibration._validate_existing_config(paths, trials, config, 2234)
            metadata["run"]["seed"] = 2234
            paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed"):
                calibration._validate_existing_config(paths, trials, config, 1234)
            paths["metadata"].unlink()
            trials = trials.assign(pilot_paths=config.pilot_paths, evaluation_paths=config.evaluation_paths)
            calibration._validate_existing_config(paths, trials, config, 1234)
            with self.assertRaisesRegex(ValueError, "seed"):
                calibration._validate_existing_config(paths, trials, config, 2234)

    def test_missing_forecast_input_does_not_overwrite_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.csv", root / "output.csv"
            pd.DataFrame([{"case": "baseline", "experiment": "baseline"}]).to_csv(source, index=False)
            output.write_text("existing valid results\n", encoding="utf-8")
            with patch.object(decoupling, "SUMMARY_DIR", root), patch.object(decoupling, "SOURCE_PATH", source), \
                    patch.object(decoupling, "OUT_PATH", output), self.assertRaises(FileNotFoundError):
                decoupling.main()
            self.assertEqual(output.read_text(encoding="utf-8"), "existing valid results\n")

    def test_forecast_summary_requires_finite_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict.fromkeys(decoupling.SUMMARY_COLUMNS, 0.5)
            row["forecast_smape"] = np.nan
            pd.DataFrame([row]).to_csv(root / "summary_baseline.csv", index=False)
            with patch.object(decoupling, "SUMMARY_DIR", root), self.assertRaises(ValueError):
                decoupling._load_summary_metrics("baseline")

    def test_swd_validation_preserves_historical_one_day_loss(self):
        config = config_module.Config()
        config.TRAINING_PATH_LENGTH = 3
        config.FAST_EULER_LOSS = False
        data = SimpleNamespace(val_set=SimpleNamespace(
            ys=np.zeros((1, 3, 2)), controls=np.zeros((1, 3, 1))))
        diagnostics = {"loss_nll": -1.0, "loss_mse": 0.0, "bi_z_std": 1.0}
        with patch.object(losses, "loss", return_value=(-1.0, diagnostics)) as loss:
            swd._validation_nll(None, config, data)
        self.assertIs(loss.call_args.kwargs["fast_euler"], True)


if __name__ == "__main__":
    unittest.main()
