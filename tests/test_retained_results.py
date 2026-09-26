"""Check retained numerical results and replay the feedback tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd



ROOT = Path(__file__).resolve().parents[1]


class RetainedResultsTests(unittest.TestCase):
    def run_feedback_summary(self, script: str, output: Path) -> dict:
        result = subprocess.run(
            [sys.executable, str(ROOT / "experiments/feedback_weight_selection" / script),
             "--out-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_feedback_final_tests_and_level_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = self.run_feedback_summary("summarize.py", output)
            fits = pd.read_csv(output / "fits.csv")
        self.assertEqual(len(fits), 16)
        self.assertTrue(fits.selected.eq("pooled").all())
        metrics = {row["metric"]: row for row in result["metrics"]}
        self.assertEqual([metrics[name]["count"] for name in
                          ("selection", "null_rejection", "alternative_rejection")], [16, 1, 4])
        self.assertAlmostEqual(result["marginal_null_upper"], 0.048453125)

    def test_joint_feedback_exclusion_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_feedback_summary("joint_point_check.py", Path(directory))
        self.assertEqual(len(result["bounds"]), 5)
        self.assertEqual(result["conditional_simulation_confidence"], 0.995)
        self.assertTrue(all(row["excluded"] for row in result["bounds"]))

    def test_nonlinear_interval_bounds_support_displayed_rounding(self):
        source = ROOT / "experiments/nonlinear_weight_selection/separation"
        points = pd.read_csv(source / "point_bounds.csv")
        bounds = pd.read_csv(source / "bound_refinement.csv")
        lower = points.loc[points.paths.eq(2097152) & points.ridge.eq(0), "gap_lower"].max()
        interval = bounds.loc[bounds.paths.eq(2097152) & bounds.ridge.eq(0)
                              & bounds.partition.eq("split_32")].iloc[0]
        self.assertGreaterEqual(lower, 0.02127)
        self.assertLessEqual(interval.cdf_upper, 0.02853)
        self.assertLessEqual(interval.rejection_deviation_upper, 0.00453)


if __name__ == "__main__":
    unittest.main()
