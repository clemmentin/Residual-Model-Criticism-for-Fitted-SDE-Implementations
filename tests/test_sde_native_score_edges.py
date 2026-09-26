"""Regression checks for degenerate correlations and invalid Monte Carlo scores."""
import unittest

import numpy as np
import pandas as pd

from experiments.run_sir_sde_native_country_audit import (
    METRIC_SPECS, PHI_NAMES, PRIMARY_COMPONENTS,
    _lag_corr, _upper_rank, build_split_scores, compute_path_metrics,
)
from experiments.run_ce_conditional_fit_pilot import ce_features, scores, compare_score_laws


class ScoreEdgeTests(unittest.TestCase):
    @staticmethod
    def finite_banks():
        ids = np.arange(40)
        draws = pd.DataFrame({"bootstrap_id": ids})
        observed = {}
        for index, metric in enumerate(METRIC_SPECS):
            values = np.linspace(-1.0, 1.0, len(ids)) + index * 0.1
            draws[metric] = values
            observed[metric] = float(np.median(values[ids % 2 == 0]))
        return observed, draws

    def test_degenerate_lag_pairs_have_zero_score_component(self):
        cases = [
            (np.zeros(60), 1),
            (np.ones(60), 7),
            (np.array([0.0, 0.0, 0.0, 1.0]), 1),
            (np.array([1.0, 0.0, 0.0, 0.0]), 1),
            (np.arange(4) * 1e-14, 1),
            (np.arange(8), 7),
            (np.arange(7), 7),
        ]
        for values, lag in cases:
            with self.subTest(values=values, lag=lag):
                self.assertEqual(_lag_corr(values, lag), 0.0)
        self.assertAlmostEqual(_lag_corr(np.arange(10), 1), 1.0)

    def test_invalid_lag_inputs_are_not_treated_as_zero_correlation(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _lag_corr(np.array([0.0, bad, 0.0, 0.0]), 1)
        for lag in (0, -1):
            with self.subTest(lag=lag), self.assertRaises(ValueError):
                _lag_corr(np.arange(10), lag)

    def test_constant_energy_path_can_be_scored(self):
        z = np.tile([-1.0, 1.0], 30)
        observed = compute_path_metrics(z, np.zeros((len(z), len(PHI_NAMES))))
        self.assertTrue(np.isfinite(list(observed.values())).all())
        self.assertEqual(observed["bracket_I_energy_acf1"], 0.0)
        self.assertEqual(observed["bracket_I_energy_acf7"], 0.0)
        _, draws = self.finite_banks()
        result, _ = build_split_scores(observed, draws)
        self.assertTrue(np.isfinite(result["observed_global_score"]))
        self.assertGreaterEqual(result["global_rank_pvalue"], 1 / 21)
        self.assertLessEqual(result["global_rank_pvalue"], 1.0)

    def test_vectorized_features_match_scalar_convention(self):
        paths = np.vstack([
            np.zeros(60), np.ones(60), np.tile([-1.0, 1.0], 30),
            np.r_[np.zeros(59), 1.0],
            np.sqrt(1.0 + np.arange(60) * 1e-14),
            np.linspace(-1.25, 1.75, 60),
        ])
        expected = [
            [compute_path_metrics(z, np.zeros((60, len(PHI_NAMES))))[m]
             for m in PRIMARY_COMPONENTS]
            for z in paths
        ]
        np.testing.assert_allclose(ce_features(paths), expected, rtol=1e-13, atol=1e-13)
        self.assertEqual(ce_features(np.arange(8.0)[None, :])[0, -1], 0.0)

    def test_nonfinite_observed_components_cannot_produce_a_rank(self):
        observed, draws = self.finite_banks()
        for metric in METRIC_SPECS:
            for bad in (np.nan, np.inf, -np.inf):
                invalid = {**observed, metric: bad}
                with self.subTest(metric=metric, bad=bad):
                    with self.assertRaisesRegex(ValueError, "Non-finite observed metric"):
                        build_split_scores(invalid, draws)

    def test_nonfinite_pilot_and_reference_components_are_rejected(self):
        observed, draws = self.finite_banks()
        for row in (0, 1):
            for bad in (np.nan, np.inf, -np.inf):
                invalid = draws.copy()
                invalid.loc[row, "bracket_I_energy_acf1"] = bad
                with self.subTest(row=row, bad=bad):
                    with self.assertRaisesRegex(ValueError, "Non-finite null metric"):
                        build_split_scores(observed, invalid)

    def test_rank_rejects_invalid_scores_and_retains_finite_ties(self):
        self.assertEqual(_upper_rank(2.0, np.array([1.0, 2.0, 2.0, 3.0])), 4 / 5)
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(observed=bad), self.assertRaises(ValueError):
                _upper_rank(bad, np.arange(4.0))
            with self.subTest(reference=bad), self.assertRaises(ValueError):
                _upper_rank(2.0, np.array([1.0, bad, 3.0]))
        for reference in (np.array([]), np.ones((2, 2))):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                _upper_rank(2.0, reference)

    def test_vectorized_scoring_and_comparison_reject_nonfinite_values(self):
        centers, scales = np.zeros(6), np.ones(6)
        for bad in (np.nan, np.inf, -np.inf):
            features = np.zeros((2, 6))
            features[0, 4] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scores(features, centers, scales)
            for held, reference in ((np.array([bad, 0.0]), np.array([0.0, 1.0])),
                                    (np.array([0.0, 1.0]), np.array([0.0, bad]))):
                with self.assertRaises(ValueError):
                    compare_score_laws(held, reference, 2500, 1)
        with self.assertRaises(ValueError):
            scores(np.zeros((2, 6)), centers, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
