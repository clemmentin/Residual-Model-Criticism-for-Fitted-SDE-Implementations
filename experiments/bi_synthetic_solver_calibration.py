"""Fitted x solver finite-grid calibration for a rank-one 2D controlled SDE.

This experiment fills the cell that no existing synthetic study covers.  The
5,000-replicate full-refit calibration has fitting but no numerical solver, and
``finite_step_geometry_synthetic.py`` has a multi-substep solver but a known
oracle simulator with no fitting.  The audited SIR object has both, so neither
study instantiates the conditioning set ``A`` of the rank-calibration theorem,
which explicitly includes solver settings.

Design
------
The truth is a two-dimensional controlled SDE whose instantaneous diffusion is
exactly rank one::

    dX1 = (-a1*X1 + kappa*X2        + <controls>) dt
    dX2 = (-a2*X2 + 0.12*tanh(1.5*X1) + <controls>) dt + sigma dW

Noise enters only the second coordinate.  With a single Euler step the first
coordinate has zero transition support; with ``n_substeps >= 2`` and
``kappa != 0`` the drift Jacobian propagates injected noise into it, so the
finite-step transition covariance is full rank.  This is the mechanism of
Proposition ``linear-finite-support`` and the tangent recursion, now with a
fitted model inside the Monte Carlo loop.

The X2 -> X1 coupling is linear on purpose.  With a ``tanh`` coupling at this
noise level the tangent recursion overstates the propagated coordinate-1
variance by about 31 percent, because the linearization is evaluated on the
zero-noise path while the actual excursions leave the linear range of the
``tanh``.  That would write an energy deficit into the coordinate map itself
and confound the calibration question.  A linear coupling puts the design in
the regime the audited SIR application reports (tangent/direct variance ratios
0.998 and 0.999) and matches the linear setting of the proposition; the
nonlinearity kept in ``b2`` and in the controls still leaves the fitted model
something to get wrong.

Everything is fixed-shape JAX so the whole replicate loop can be ``vmap``-ed
over trials; on a single trial the workload is interpreter-bound and a GPU is
useless, but batched over hundreds of trials the per-substep matrices become
large enough for the accelerator to matter.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import optax

STATE_DIM = 2
LOOKBACK = 7
PERIOD = 20
CONTROL_DIM = 8          # last_inc x2, mean_inc x2, level_mean x2, sin, cos
INPUT_DIM = STATE_DIM + CONTROL_DIM

# Truth-drift constants.  Only `kappa` is swept; the rest are fixed so that the
# swept axis is unambiguously the propagation strength.
A1 = 0.22
A2 = 0.30
CROSS_21 = 0.12          # X1 -> X2 drift coupling, held fixed
CTRL_MEAN_INC = 0.18
CTRL_LEVEL = 0.04
CTRL_SIN = 0.06
CTRL_COS = -0.025
SIGMA_TRUE = 0.42
INIT_SCALE = 0.7


# ---------------------------------------------------------------------------
# Predictable controls
# ---------------------------------------------------------------------------

def controls_from_window(window: jnp.ndarray, step_index: jnp.ndarray) -> jnp.ndarray:
    """Controls available before the transition out of the window's last row.

    ``window`` is ``(LOOKBACK + 1, STATE_DIM)``: the state at the transition
    origin plus the ``LOOKBACK`` states before it.  The window is always full,
    because controls are only ever evaluated at observation indices
    ``k >= LOOKBACK``.  That is what removes the data-dependent branch in the
    original scalar implementation and makes the whole thing vmap-able.
    """
    increments = jnp.diff(window, axis=0)          # (LOOKBACK, STATE_DIM)
    last_inc = increments[-1]                       # (STATE_DIM,)
    mean_inc = jnp.mean(increments, axis=0)         # (STATE_DIM,)
    level_mean = jnp.mean(window, axis=0)           # (STATE_DIM,)
    phase = 2.0 * math.pi * jnp.mod(step_index, PERIOD) / PERIOD
    return jnp.concatenate(
        [last_inc, mean_inc, level_mean, jnp.stack([jnp.sin(phase), jnp.cos(phase)])]
    )


def gather_window(path: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """Rows ``k - LOOKBACK ... k`` of ``path`` as a fixed-shape slice."""
    return jax.lax.dynamic_slice(path, (k - LOOKBACK, 0), (LOOKBACK + 1, STATE_DIM))


# ---------------------------------------------------------------------------
# Truth drift and its Jacobian
# ---------------------------------------------------------------------------

def true_drift(state: jnp.ndarray, controls: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """Instantaneous drift of the generating SDE.

    Controls are held fixed across the substeps of one observation step, which
    matches how the audited SIR simulator treats its signature controls.
    """
    x1, x2 = state[0], state[1]
    mean_inc_1, mean_inc_2 = controls[2], controls[3]
    level_1, level_2 = controls[4], controls[5]
    sin_phase, cos_phase = controls[6], controls[7]

    # The X2 -> X1 coupling is deliberately LINEAR.  Proposition
    # `linear-finite-support` is stated for a linear SDE, so a linear coupling
    # makes the propagated coordinate-1 variance an exact consequence of the
    # tangent recursion rather than a linearization of it.  Keeping the
    # nonlinearity in b2 and in the controls still leaves the fitted model
    # something to get wrong.
    b1 = (
        -A1 * x1
        + kappa * x2
        + CTRL_MEAN_INC * mean_inc_1
        + CTRL_SIN * sin_phase
        + CTRL_COS * cos_phase
    )
    b2 = (
        -A2 * x2
        + CROSS_21 * jnp.tanh(1.5 * x1)
        + CTRL_MEAN_INC * mean_inc_2
        + CTRL_LEVEL * level_2
        + CTRL_SIN * sin_phase
    )
    _ = level_1
    return jnp.stack([b1, b2])


def true_drift_jacobian(
    state: jnp.ndarray, controls: jnp.ndarray, kappa: float
) -> jnp.ndarray:
    """d b / d state, the object the tangent recursion propagates."""
    return jax.jacobian(true_drift, argnums=0)(state, controls, kappa)


# Instantaneous diffusion: noise enters coordinate 2 only, so the matrix is
# exactly rank one and its range never contains coordinate 1.
def diffusion_matrix(sigma: jnp.ndarray | float) -> jnp.ndarray:
    return jnp.array([[0.0, 0.0], [0.0, 1.0]]) * (sigma ** 2)


# ---------------------------------------------------------------------------
# One observation step: zero-noise path + tangent covariance recursion
# ---------------------------------------------------------------------------

def step_moments(
    drift_fn,
    state: jnp.ndarray,
    sigma: jnp.ndarray | float,
    n_substeps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Effective one-step drift and transition covariance under the solver.

    ``drift_fn`` maps a state to an instantaneous drift with the controls
    already closed over; controls are held fixed across the substeps of one
    observation step, matching how the audited SIR simulator treats its
    signature controls.

    Integrates the deterministic (zero-noise) path over ``n_substeps`` Euler
    steps of size ``dt = 1 / n_substeps`` while propagating

        Sigma_{j+1} = (I + J_j dt) Sigma_j (I + J_j dt)^T + Sigma_diff dt,

    which is the discrete tangent recursion of the paper.  With
    ``n_substeps == 1`` this returns ``Sigma = Sigma_diff``, whose first
    diagonal entry is exactly zero: the instantaneous zero-support case.

    The same function serves the true generator and the fitted model, so
    observed and reference paths are guaranteed to pass through one map.
    """
    dt = 1.0 / n_substeps
    sigma_diff = diffusion_matrix(sigma)
    eye = jnp.eye(STATE_DIM)
    jacobian_fn = jax.jacobian(drift_fn)

    def body(carry, _):
        y, cov = carry
        transition = eye + jacobian_fn(y) * dt
        cov = transition @ cov @ transition.T + sigma_diff * dt
        y = y + drift_fn(y) * dt
        return (y, cov), None

    (y_final, cov), _ = jax.lax.scan(
        body, (state, jnp.zeros((STATE_DIM, STATE_DIM))), None, length=n_substeps
    )
    return y_final - state, cov


@partial(jax.jit, static_argnames=("n_substeps",))
def finite_step_moments(
    state: jnp.ndarray,
    controls: jnp.ndarray,
    kappa: float,
    sigma: jnp.ndarray | float,
    n_substeps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Finite-step moments of the *true* generator."""
    return step_moments(
        lambda y: true_drift(y, controls, kappa), state, sigma, n_substeps
    )


NOISE_DIRECTION = jnp.array([0.0, 1.0])   # rank-one: coordinate 2 only


def advance_observation_step(
    drift_fn,
    state: jnp.ndarray,
    noise: jnp.ndarray,
    sigma: jnp.ndarray | float,
) -> jnp.ndarray:
    """Advance one observation step with noise injected at every substep.

    ``noise`` is ``(n_substeps,)`` standard normal draws; each substep adds
    ``sigma * sqrt(dt) * noise[j]`` to coordinate 2 only.  The substep count is
    taken from the length of ``noise``.
    """
    n_substeps = noise.shape[0]
    dt = 1.0 / n_substeps
    sqrt_dt = math.sqrt(dt)

    def body(y, eps):
        y = y + drift_fn(y) * dt
        return y + NOISE_DIRECTION * (sigma * sqrt_dt * eps), None

    y_final, _ = jax.lax.scan(body, state, noise)
    return y_final


@partial(jax.jit, static_argnames=("n_substeps",))
def simulate_observation_step(
    state: jnp.ndarray,
    controls: jnp.ndarray,
    noise: jnp.ndarray,
    kappa: float,
    sigma: jnp.ndarray | float,
    n_substeps: int,
) -> jnp.ndarray:
    """One observation step of the *true* generator."""
    del n_substeps  # taken from noise.shape
    return advance_observation_step(
        lambda y: true_drift(y, controls, kappa), state, noise, sigma
    )


def zero_noise_drift(drift_fn, state: jnp.ndarray, n_substeps: int) -> jnp.ndarray:
    """Effective one-step drift only, without the tangent covariance.

    Training uses this together with the *unpropagated* plug-in variance
    accumulator ``sum_j sigma^2 dt``, which is what the audited SIR training
    loss does.  Skipping the Jacobian here is not an approximation of
    convenience: the paper treats the unpropagated accumulator and the tangent
    covariance as two admissible finite-step scale constructions, and the audit
    coordinate below uses the tangent one.
    """
    dt = 1.0 / n_substeps

    def body(y, _):
        return y + drift_fn(y) * dt, None

    y_final, _ = jax.lax.scan(body, state, None, length=n_substeps)
    return y_final - state


# ---------------------------------------------------------------------------
# Truth path simulation on the observation grid
# ---------------------------------------------------------------------------

ZERO_CONTROLS = jnp.zeros((CONTROL_DIM,))


def simulate_truth_path(
    key: jax.Array,
    n_steps: int,
    kappa: float,
    n_substeps: int,
) -> jnp.ndarray:
    """One observation-grid path of the generating SDE, shape ``(n_steps+1, 2)``.

    The first ``LOOKBACK`` transitions are a burn-in run with zero controls.
    That keeps every shape static -- the original scalar benchmark branched on
    a partially filled history instead, which is exactly what blocks ``vmap``.
    Controls are only ever consumed at observation indices ``k >= LOOKBACK``,
    where the window is full, so the burn-in convention never enters the audit.
    """
    key_init, key_noise = jax.random.split(key)
    x0 = jax.random.normal(key_init, (STATE_DIM,)) * INIT_SCALE
    noise = jax.random.normal(key_noise, (n_steps, n_substeps))

    def burn_body(y, eps):
        y = advance_observation_step(
            lambda z: true_drift(z, ZERO_CONTROLS, kappa), y, eps, SIGMA_TRUE
        )
        return y, y

    y_burn, burn_states = jax.lax.scan(burn_body, x0, noise[:LOOKBACK])
    window = jnp.concatenate([x0[None, :], burn_states], axis=0)

    def main_body(carry, inputs):
        win, k = carry
        eps = inputs
        controls = controls_from_window(win, k)
        y_next = advance_observation_step(
            lambda z: true_drift(z, controls, kappa), win[-1], eps, SIGMA_TRUE
        )
        win = jnp.concatenate([win[1:], y_next[None, :]], axis=0)
        return (win, k + 1), y_next

    _, main_states = jax.lax.scan(
        main_body, (window, jnp.array(LOOKBACK)), noise[LOOKBACK:]
    )
    return jnp.concatenate([window, main_states], axis=0)


def path_transitions(path: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Audited transitions of one path: origin states, controls, increments."""
    n_steps = path.shape[0] - 1
    ks = jnp.arange(LOOKBACK, n_steps)

    def one(k):
        window = gather_window(path, k)
        return path[k], controls_from_window(window, k), path[k + 1] - path[k]

    return jax.vmap(one)(ks)


# ---------------------------------------------------------------------------
# Fitted transition model
# ---------------------------------------------------------------------------

WIDTH_MAX = 16
CANDIDATE_WIDTHS = (8, 16)
S_LOSS_SCALE = 1.0        # weight on the deterministic drift term, as in SIR
SIGMA_FLOOR = 1e-5


def width_mask(width: int) -> jnp.ndarray:
    """Hidden-unit mask making a narrow candidate exactly representable.

    Masking *after* the activation is exact: a masked unit contributes zero
    regardless of its pre-activation, which is identical to zeroing its output
    weights.  Both candidates therefore share one parameter shape and one
    ``vmap`` axis, so model selection is a ``where`` instead of a shape switch.
    """
    return (jnp.arange(WIDTH_MAX) < width).astype(jnp.float32)


def init_params(key: jax.Array) -> dict[str, jnp.ndarray]:
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (INPUT_DIM, WIDTH_MAX)) * 0.15,
        "b1": jnp.zeros((WIDTH_MAX,)),
        "w2": jax.random.normal(k2, (WIDTH_MAX, STATE_DIM)) * 0.10,
        "b2": jnp.zeros((STATE_DIM,)),
        "log_sigma": jnp.array(math.log(0.45), dtype=jnp.float32),
    }


def fitted_drift(
    params: dict[str, jnp.ndarray],
    mask: jnp.ndarray,
    state: jnp.ndarray,
    controls: jnp.ndarray,
    in_mean: jnp.ndarray,
    in_scale: jnp.ndarray,
) -> jnp.ndarray:
    x = (jnp.concatenate([state, controls]) - in_mean) / in_scale
    hidden = jnp.tanh(x @ params["w1"] + params["b1"]) * mask
    return hidden @ params["w2"] + params["b2"]


def fitted_sigma(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.exp(params["log_sigma"]) + SIGMA_FLOOR


def _fitted_drift_fn(params, mask, controls, in_mean, in_scale):
    return lambda y: fitted_drift(params, mask, y, controls, in_mean, in_scale)


def fitted_effective_drift(
    params, mask, state, controls, in_mean, in_scale, n_substeps: int
) -> jnp.ndarray:
    """Solver-integrated one-step drift of the fitted model (no Jacobian)."""
    return zero_noise_drift(
        _fitted_drift_fn(params, mask, controls, in_mean, in_scale), state, n_substeps
    )


def fitted_tangent_moments(
    params, mask, state, controls, in_mean, in_scale, n_substeps: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solver-integrated drift *and* tangent covariance of the fitted model."""
    return step_moments(
        _fitted_drift_fn(params, mask, controls, in_mean, in_scale),
        state,
        fitted_sigma(params),
        n_substeps,
    )


# ---------------------------------------------------------------------------
# Training: solver-matched transition loss
# ---------------------------------------------------------------------------

# Adam's state transformation is kept independent of the step size so that the
# learning-rate argument actually controls optimization.
OPTIMIZER = optax.scale_by_adam()


def standardize_inputs(states: jnp.ndarray, controls: jnp.ndarray):
    x = jnp.concatenate([states, controls], axis=1)
    mean = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0)
    return mean, jnp.where(std < 1e-6, 1.0, std)


VARIANCE_MODELS = ("tangent", "plugin")


def transition_loss(
    params, mask, states, controls, increments, in_mean, in_scale,
    n_substeps: int, weight_decay: float, variance_model: str = "tangent",
) -> jnp.ndarray:
    """Gaussian transition loss matched to the multi-substep simulator.

    Mirrors the audited SIR objective: a deterministic drift term plus a
    transition NLL on the noise-carrying coordinate, with the mean given by the
    solver-integrated drift.  The deterministic term is not decorative -- it is
    the only training signal the noise-free coordinate ever receives, since the
    NLL touches coordinate 2 alone.  Dropping it leaves coordinate 1's drift
    unconstrained and the audit degenerates completely.

    ``variance_model`` selects which of the paper's two admissible finite-step
    scale constructions trains the volatility:

    ``"tangent"``
        the propagated transition variance from the tangent recursion, i.e. the
        same construction the audit coordinate uses.  This is the *aligned*
        configuration.

    ``"plugin"``
        the unpropagated accumulator ``sum_j sigma^2 dt = sigma^2``.  When drift
        contraction over one observation step is appreciable, this trains sigma
        to absorb the contraction and the audit then applies it a second time,
        so residuals are inflated by a factor that does *not* vanish with more
        training data.  The audited SIR application reports the two
        constructions agreeing to 0.998 and 0.999, which is why the choice is
        immaterial there; this arm is the controlled case where it is not.
    """
    if variance_model not in VARIANCE_MODELS:
        raise ValueError(f"variance_model must be one of {VARIANCE_MODELS}")

    if variance_model == "tangent":
        eff, cov = jax.vmap(
            lambda s, c: fitted_tangent_moments(
                params, mask, s, c, in_mean, in_scale, n_substeps
            )
        )(states, controls)
        var = jnp.maximum(cov[:, 1, 1], 1e-10)
    else:
        eff = jax.vmap(
            lambda s, c: fitted_effective_drift(
                params, mask, s, c, in_mean, in_scale, n_substeps
            )
        )(states, controls)
        var = fitted_sigma(params) ** 2

    resid = increments - eff
    nll = 0.5 * resid[:, 1] ** 2 / var + 0.5 * jnp.log(var)
    l2 = jnp.sum(params["w1"] ** 2) + jnp.sum(params["w2"] ** 2)
    return jnp.mean(nll) + S_LOSS_SCALE * jnp.mean(resid ** 2) + weight_decay * l2


def train_candidate(
    key, mask, states, controls, increments, in_mean, in_scale,
    n_substeps: int, epochs: int, learning_rate: float, weight_decay: float,
    variance_model: str = "tangent",
):
    """Full-batch Adam on one candidate width.  Returns params and final loss."""
    params = init_params(key)
    opt_state = OPTIMIZER.init(params)

    def body(_, carry):
        prm, opt = carry
        grads = jax.grad(transition_loss)(
            prm, mask, states, controls, increments, in_mean, in_scale,
            n_substeps, weight_decay, variance_model,
        )
        updates, opt = OPTIMIZER.update(grads, opt, prm)
        updates = jax.tree_util.tree_map(lambda u: -learning_rate * u, updates)
        return optax.apply_updates(prm, updates), opt

    params, _ = jax.lax.fori_loop(0, epochs, body, (params, opt_state))
    return params


# ---------------------------------------------------------------------------
# Fitted-null reference bank
# ---------------------------------------------------------------------------

def simulate_reference_paths(
    key, params, mask, in_mean, in_scale, prefix, n_paths, n_steps, n_substeps: int
) -> jnp.ndarray:
    """Reference paths from the fitted model, conditioned on the observed prefix.

    Controls are recomputed along every simulated path, so the reference bank
    reproduces the complete fitted implementation rather than replaying the
    observed control sequence.
    """
    horizon = n_steps - LOOKBACK
    noise = jax.random.normal(key, (n_paths, horizon, n_substeps))
    sigma = fitted_sigma(params)

    def one_path(eps):
        def body(carry, e):
            win, k = carry
            controls = controls_from_window(win, k)
            y_next = advance_observation_step(
                _fitted_drift_fn(params, mask, controls, in_mean, in_scale),
                win[-1], e, sigma,
            )
            return (jnp.concatenate([win[1:], y_next[None, :]], axis=0), k + 1), y_next

        _, states = jax.lax.scan(body, (prefix, jnp.array(LOOKBACK)), eps)
        return jnp.concatenate([prefix, states], axis=0)

    return jax.vmap(one_path)(noise)


def path_residuals(
    params, mask, in_mean, in_scale, path, n_substeps: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Raw finite-step residual vectors and their tangent covariances.

    Returns ``(resid, cov)`` with shapes ``(horizon, 2)`` and
    ``(horizon, 2, 2)``.  All three audit arms are functions of exactly these
    two arrays, so observed and reference paths cannot diverge in the map.
    """
    states, controls, increments = path_transitions(path)
    eff, cov = jax.vmap(
        lambda s, c: fitted_tangent_moments(
            params, mask, s, c, in_mean, in_scale, n_substeps
        )
    )(states, controls)
    return increments - eff, cov
