"""Check paired aggregation against the counts retained with the manuscript."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments import summarize_coordinate_ablation as reporting


ROOT = Path(__file__).resolve().parents[1]


class CoordinateAblationSummaryTests(unittest.TestCase):
    def write_trials(self, directory: Path) -> dict:
        evidence = json.loads((ROOT / "docs/results/coordinate_ablation_counts.json").read_text(
            encoding="utf-8"))
        for tag, run in evidence["runs"].items():
            metadata = run["metadata"]
            n = metadata["replicates"]
            columns = [f"p_{scenario}_{method}_full"
                       for scenario in ("null", "targeted", "uniform")
                       for method in reporting.METHODS.values()]
            # The bootstrap resamples paired decision counts, so their trial
            # ordering and the unused score magnitudes do not affect its result.
            frame = pd.DataFrame(100 / 201, index=range(n), columns=columns)
            frame.insert(0, "replicate", np.arange(n))
            for label, record in run["null"].items():
                column = f"p_null_{reporting.METHODS[label]}_full"
                frame.loc[:record["rejects"] - 1, column] = 2 / 201
            for comparison in run["comparisons"]:
                if comparison["coord"] != "full":
                    continue
                scenario = comparison["scenario"]
                both, a_only, c_only = (comparison[key] for key in ("both", "left_only", "right_only"))
                a = frame.columns.get_loc(f"p_{scenario}_{reporting.METHODS['A']}_full")
                c = frame.columns.get_loc(f"p_{scenario}_{reporting.METHODS['C']}_full")
                frame.iloc[:both + a_only, a] = 2 / 201
                frame.iloc[:both, c] = 2 / 201
                frame.iloc[both + a_only:both + a_only + c_only, c] = 2 / 201
            stem = directory / f"coord_ablation_{tag}"
            frame.to_csv(stem.with_name(stem.name + "_trials.csv"), index=False)
            stem.with_name(stem.name + "_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8")
        return evidence

    def test_reported_intervals_and_both_holm_families_from_paired_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = self.write_trials(root)
            result = reporting.summarize(root)
        self.assertEqual(len(result["strengths"]), 6)
        for name in ("pooled_power_comparisons", "pooled_null_calibration"):
            self.assertEqual(len(result[name]), len(evidence[name]))
            for actual, expected in zip(result[name], evidence[name]):
                for field, value in actual.items():
                    if isinstance(value, str):
                        self.assertEqual(value, expected[field])
                    else:
                        np.testing.assert_allclose(value, expected[field], rtol=0, atol=1e-14,
                                                   err_msg=f"{name}: {field}")

    def test_bootstrap_keeps_strengths_and_method_pairs_together(self):
        a, c = (f"p_targeted_{reporting.METHODS[label]}_full" for label in ("A", "C"))
        groups = [pd.DataFrame({a: [0.5] * 10, c: [0.01] * 10}),
                  pd.DataFrame({a: [0.01] * 10, c: [0.5] * 10})]
        result = reporting.paired_comparison(groups, "targeted", np.random.default_rng(1))
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["stratified_paired_bootstrap_ci95"], [0, 0])
        self.assertEqual(result["mcnemar_two_sided_p"], 1)

    def test_partial_duplicate_and_nonfinite_trials_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_trials(root)
            path = root / "coord_ablation_cond1_lo_trials.csv"
            original = pd.read_csv(path)
            duplicate = original.copy()
            duplicate.loc[1, "replicate"] = 0
            nonfinite = original.copy()
            nonfinite.loc[0, "p_null_A_global_mc_full"] = np.nan
            for frame in (original.iloc[:-1], duplicate, nonfinite):
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    reporting.read_trials(root, "cond1_lo")


if __name__ == "__main__":
    unittest.main()
