import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from config import Config
from neural_sde import NeuralSDE
from sir_data import (
    _explicit_recon_control_vector_jax,
    _explicit_recon_control_frame,
    get_sir_control_lookback,
)
from sir_training import (
    make_trainable_filter_spec,
    validate_training_config,
)


def _small_model() -> NeuralSDE:
    return NeuralSDE(
        signature_sizes=[4],
        macro_size=0,
        state_size=2,
        key=jax.random.PRNGKey(0),
        norm_mean=jnp.array([0.7, -6.0]),
        norm_std=jnp.array([0.2, 2.0]),
        freeze_gamma=True,
    )


class TrainingInvariantTests(unittest.TestCase):
  def test_optimizer_cannot_change_normalization_arrays(self):
    model = _small_model()
    spec = make_trainable_filter_spec(model)
    trainable, frozen = eqx.partition(model, spec)
    optimizer = optax.adamw(1e-3, weight_decay=1e-2)
    state = optimizer.init(trainable)

    def objective(candidate):
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(candidate) if leaf is not None]
        return sum(jnp.sum(leaf**2) for leaf in leaves)

    grads = eqx.filter_grad(objective)(trainable)
    updates, _ = optimizer.update(grads, state, trainable)
    updated = eqx.combine(eqx.apply_updates(trainable, updates), frozen)

    for name in (
        "norm_mean",
        "norm_std",
        "norm_whiten_matrix",
        "norm_dewhiten_matrix",
    ):
        np.testing.assert_array_equal(np.asarray(getattr(updated, name)), np.asarray(getattr(model, name)))
    self.assertFalse(np.array_equal(np.asarray(updated.log_sigma), np.asarray(model.log_sigma)))

  def test_explicit_pathwise_controls_match_frame_builder(self):
    cfg = Config(EXPLICIT_RECON_CONTROLS="acf_weekday", EXPLICIT_RECON_CONTROL_WINDOW=20)
    index = pd.date_range("2021-01-01", periods=30, freq="D")
    values = np.column_stack(
        (np.linspace(-1.0, 1.0, 30), np.sin(np.linspace(0.0, 4.0, 30)))
    )
    frame = pd.DataFrame(values, index=index, columns=["S", "I"])
    valid_ts = index[get_sir_control_lookback(cfg) - 1 :]
    expected = _explicit_recon_control_frame(frame, cfg, valid_ts).iloc[-1].values
    actual = _explicit_recon_control_vector_jax(
        jnp.asarray(frame.values), cfg, jnp.array(index[-1].weekday())
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


  def test_integration_horizon_must_equal_one_day(self):
    cfg = Config(SDE_SUBSTEP_DT=0.2, SDE_NUM_SUBSTEPS=4)
    with self.assertRaisesRegex(ValueError, "must equal one day"):
        validate_training_config(cfg)


if __name__ == "__main__":
    unittest.main()
