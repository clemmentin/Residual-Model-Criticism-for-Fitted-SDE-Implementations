"""Small checks for the diagnostic counting and derivative conventions."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from experiments.audit_sir_beta_clipping import bank_spec, summary


class BetaAuditTests(unittest.TestCase):
    def test_strict_negatives_are_distinct_from_numerical_zeros(self):
        values = np.array([[[-1., 0., 2.]], [[3., 4., 5.]]])
        result = summary(values)
        self.assertEqual(result["negative_n"], 1)
        self.assertEqual(result["exact_zero_n"], 1)
        self.assertEqual(result["n_substeps"], 6)
        self.assertEqual(result["paths_with_nonpositive"], 1)
        self.assertAlmostEqual(result["nonpositive_fraction"], 1 / 3)

    def test_single_observed_path(self):
        result = summary(np.ones((60, 10)))
        self.assertEqual(result["n_substeps"], 600)
        self.assertEqual(result["paths_with_nonpositive"], 0)

    def test_alignment_draws_are_individual_one_day_paths(self):
        values = np.ones((8, 512, 10))
        values[0, 0, 0] = -1
        # Companion views flatten anchor and draw, not the microstep axis.
        result = summary(values.reshape(-1, 1, 10))
        self.assertEqual(result["n_substeps"], 40960)
        self.assertEqual(result["paths_with_nonpositive"], 1)

    def test_maximum_derivative_convention(self):
        derivative = jax.vmap(jax.grad(lambda x: jnp.maximum(x, 0.)))(jnp.array([-1., 0., 1.]))
        np.testing.assert_array_equal(derivative, [0., .5, 1.])

    def test_bank_specs_preserve_historical_seeds_and_sizes(self):
        self.assertEqual(bank_spec("FIN")[1:], (500, 20260718, "fitted-null"))
        self.assertEqual(bank_spec("MDA")[1:], (5000, 20260814, "fitted-null"))
        self.assertEqual(bank_spec("BGR")[1:], (5000, 20260813, "exploratory-5000-fitted-null"))


if __name__ == "__main__":
    unittest.main()
