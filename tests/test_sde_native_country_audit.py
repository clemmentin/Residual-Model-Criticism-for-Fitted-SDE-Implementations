import unittest

import numpy as np
import pandas as pd

from experiments.run_sir_sde_native_country_audit import (
    CONFIRMATORY_COUNTRIES,
    DEVELOPMENT_COUNTRIES,
    METRIC_SPECS,
    PHI_NAMES,
    PRIMARY_COMPONENTS,
    _country_seed,
    _endpoint_settings,
    build_split_scores,
    compute_path_metrics,
)
from experiments.country_bi_scoring import (
    COMPONENTS as BI_COMPONENTS,
    build_country_rows as build_parity_rows,
    build_country_rows_from_banks,
)


class SDENativeCountryAuditTests(unittest.TestCase):
    def test_explicit_banks_match_the_existing_even_odd_split(self):
        draw_ids = np.arange(40)
        draws = pd.DataFrame({"bootstrap_id": draw_ids})
        observed = {}
        for index, metric in enumerate(BI_COMPONENTS):
            values = np.linspace(-1.0, 1.0, len(draw_ids)) + index * 0.1
            draws[metric] = values
            observed[metric] = float(np.median(values[draw_ids % 2 == 0]))
        summary = pd.DataFrame(
            {
                "case": ["test"] * len(BI_COMPONENTS),
                "val_country": ["FIN"] * len(BI_COMPONENTS),
                "metric": list(BI_COMPONENTS),
                "observed": [observed[metric] for metric in BI_COMPONENTS],
            }
        )
        parity_global, parity_profile = build_parity_rows(summary, draws)
        explicit_global, explicit_profile = build_country_rows_from_banks(
            summary,
            draws.loc[draw_ids % 2 == 0],
            draws.loc[draw_ids % 2 == 1],
        )
        self.assertEqual(parity_global, explicit_global)
        self.assertEqual(parity_profile, explicit_profile)

    def test_country_partition_and_endpoint_settings_are_explicit(self):
        self.assertEqual(len(DEVELOPMENT_COUNTRIES), 9)
        self.assertEqual(CONFIRMATORY_COUNTRIES, ("FIN", "NOR", "SWE"))
        self.assertTrue(set(DEVELOPMENT_COUNTRIES).isdisjoint(CONFIRMATORY_COUNTRIES))
        settings = _endpoint_settings(n_bootstrap=500, seed=20260718)
        self.assertEqual(settings["primary_global_components"], list(PRIMARY_COMPONENTS))
        self.assertEqual(len(PRIMARY_COMPONENTS), 6)

    def test_path_metric_order_matches_frozen_specification(self):
        z = np.linspace(-1.25, 1.75, 60)
        dynkin = np.sin(np.linspace(0.0, 4.0, 60))[:, None]
        self.assertEqual(dynkin.shape[1], len(PHI_NAMES))
        metrics = compute_path_metrics(z, dynkin)
        self.assertEqual(tuple(metrics), tuple(METRIC_SPECS))
        self.assertTrue(np.isfinite(list(metrics.values())).all())

    def test_diagnostic_generator_does_not_enter_primary_global_max(self):
        draw_ids = np.arange(20)
        draws = pd.DataFrame({"bootstrap_id": draw_ids})
        observed = {}
        base = np.linspace(-1.0, 1.0, len(draw_ids))
        for index, metric in enumerate(METRIC_SPECS):
            values = base + index * 0.1
            draws[metric] = values
            pilot = values[draw_ids % 2 == 0]
            observed[metric] = float(np.median(pilot))
        for metric, spec in METRIC_SPECS.items():
            if spec["score_role"] == "diagnostic":
                observed[metric] += 100.0

        score, summary = build_split_scores(observed, draws)
        self.assertEqual(score["observed_global_score"], 0.0)
        self.assertEqual(score["global_rank_pvalue"], 1.0)
        self.assertIn(score["dominant_metric"], PRIMARY_COMPONENTS)
        diagnostic = summary.loc[summary["score_role"].eq("diagnostic")]
        self.assertTrue(diagnostic["primary_global_max_adjusted_pvalue"].isna().all())
        self.assertLess(score["generator_rank_pvalue"], 1.0)

    def test_country_seed_is_stable_and_purpose_separated(self):
        first = _country_seed(20260718, "DEU", "fitted-null")
        self.assertEqual(first, _country_seed(20260718, "DEU", "fitted-null"))
        self.assertNotEqual(first, _country_seed(20260718, "DEU", "alignment"))
        self.assertNotEqual(first, _country_seed(20260718, "FRA", "fitted-null"))


if __name__ == "__main__":
    unittest.main()
