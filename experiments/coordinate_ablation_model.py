"""Extra model pieces for the A/B/C coordinate-construction ablation.

Reuses ``bi_synthetic_solver_calibration`` (the fitted x solver 2D controlled
SDE) for the truth generator, the fitted network, and the tangent recursion.
This module adds only what that file does not already provide:

``A``  global whitening
    One covariance matrix, estimated once from an independent pilot bank,
    applied unchanged at every step of every path.  Contrast with the
    per-step ``fitted_tangent_moments`` covariance already in the base module.

``B``  theoretical calibration of the tangent coordinate
    The same per-step tangent whitening as the paper's coordinate ``C``, but
    the resulting energy statistic is referred to its working chi-square
    approximation (treating the ``K`` per-step whitened coordinates as iid)
    instead of a matched-simulation rank.

Deviations
    Targeted (extra noise loaded onto the *un*forced coordinate only) and
    uniform (a scalar multiple of the true diffusion, loaded the same way in
    every direction and every state).  Both act on the *frozen fitted*
    simulator, not the truth generator, so held-out and reference paths never
    differ by a refitting gap -- only by the deviation itself.

Nonlinear-coupling condition (design condition 3)
    ``bi_synthetic_solver_calibration.true_drift`` deliberately keeps the
    propagating coupling (``kappa``) linear so the tangent recursion is exact
    for it; its own docstring records that switching to a ``tanh`` coupling
    overstates the propagated coordinate-1 variance by about 31 percent.  This
    module exposes that same switch on the *fitted* side: the fitted network
    is unconstrained by construction, so what actually needs to leave the
    linear regime is the truth generator the network is trying to match.
    ``nonlinear_kappa=True`` swaps the truth drift's ``kappa * x2`` term for
    ``kappa * tanh(coupling_scale * x2)`` at a noise level large enough that
    realized excursions leave the linearization neighbourhood.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy import stats as jstats


from experiments import bi_synthetic_solver_calibration as model

STATE_DIM = model.STATE_DIM


# ---------------------------------------------------------------------------
# Reproducible seeding
# ---------------------------------------------------------------------------

def condition_key(base_seed: int, condition_id: int) -> jax.Array:
    """Stable PRNG key for one condition of one probe.

    Do NOT derive seeds from `hash(label)`: Python randomizes string hashing
    per process (PYTHONHASHSEED), so the same label yields a different seed on
    every run -- verified, three runs produced three disjoint seed triples.
    Probe outputs seeded that way are not reproducible and, worse, are not
    comparable across runs, which silently invalidates any before/after
    comparison of probe results.

    `condition_id` must be a fixed small integer assigned per condition and
    never reused or renumbered, so that editing a display label (e.g. its
    whitespace) cannot change which experiment was run.
    """
    if not isinstance(condition_id, int) or condition_id < 0:
        raise ValueError(f"condition_id must be a non-negative int, got {condition_id!r}")
    return jax.random.fold_in(jax.random.PRNGKey(base_seed), condition_id)

# ---------------------------------------------------------------------------
# Condition-3 truth drift: nonlinear propagating coupling
# ---------------------------------------------------------------------------

COUPLING_SCALE = 0.6  # tanh(0.6 * x2) leaves the linear range within ~1 sd of noise


def true_drift_nonlinear_kappa(
    state: jnp.ndarray, controls: jnp.ndarray, kappa: float
) -> jnp.ndarray:
    """Truth drift with the coordinate-2 -> coordinate-1 coupling made nonlinear.

    Identical to ``model.true_drift`` except the propagating term is
    ``kappa * tanh(COUPLING_SCALE * x2)`` in place of ``kappa * x2``.  The
    tangent recursion linearizes this coupling at the zero-noise path on every
    substep; the gap between that linearization and the coupling's actual
    curvature over a realized excursion is exactly what condition 3 is meant
    to expose.
    """
    x1, x2 = state[0], state[1]
    mean_inc_1, mean_inc_2 = controls[2], controls[3]
    level_1, level_2 = controls[4], controls[5]
    sin_phase, cos_phase = controls[6], controls[7]

    b1 = (
        -model.A1 * x1
        + kappa * jnp.tanh(COUPLING_SCALE * x2)
        + model.CTRL_MEAN_INC * mean_inc_1
        + model.CTRL_SIN * sin_phase
        + model.CTRL_COS * cos_phase
    )
    b2 = (
        -model.A2 * x2
        + model.CROSS_21 * jnp.tanh(1.5 * x1)
        + model.CTRL_MEAN_INC * mean_inc_2
        + model.CTRL_LEVEL * level_2
        + model.CTRL_SIN * sin_phase
    )
    _ = level_1
    return jnp.stack([b1, b2])


def truth_drift_fn(controls: jnp.ndarray, kappa: float, nonlinear_kappa: bool):
    base = true_drift_nonlinear_kappa if nonlinear_kappa else model.true_drift
    return lambda y: base(y, controls, kappa)


def simulate_truth_path_general(
    key: jax.Array,
    n_steps: int,
    kappa: float,
    n_substeps: int,
    nonlinear_kappa: bool,
) -> jnp.ndarray:
    """``model.simulate_truth_path`` parameterized by the coupling regime."""
    drift = true_drift_nonlinear_kappa if nonlinear_kappa else model.true_drift
    key_init, key_noise = jax.random.split(key)
    x0 = jax.random.normal(key_init, (STATE_DIM,)) * model.INIT_SCALE
    noise = jax.random.normal(key_noise, (n_steps, n_substeps))

    def burn_body(y, eps):
        y = model.advance_observation_step(
            lambda z: drift(z, model.ZERO_CONTROLS, kappa), y, eps, model.SIGMA_TRUE
        )
        return y, y

    y_burn, burn_states = jax.lax.scan(burn_body, x0, noise[: model.LOOKBACK])
    window = jnp.concatenate([x0[None, :], burn_states], axis=0)

    def main_body(carry, inputs):
        win, k = carry
        eps = inputs
        controls = model.controls_from_window(win, k)
        y_next = model.advance_observation_step(
            lambda z: drift(z, controls, kappa), win[-1], eps, model.SIGMA_TRUE
        )
        win = jnp.concatenate([win[1:], y_next[None, :]], axis=0)
        return (win, k + 1), y_next

    _, main_states = jax.lax.scan(
        main_body, (window, jnp.array(model.LOOKBACK)), noise[model.LOOKBACK :]
    )
    return jnp.concatenate([window, main_states], axis=0)


# ---------------------------------------------------------------------------
# Fast finite-step moments: analytic Jacobian instead of jax.jacobian
# ---------------------------------------------------------------------------

def fitted_drift_jacobian(
    params, mask, state: jnp.ndarray, controls: jnp.ndarray, in_mean, in_scale
) -> jnp.ndarray:
    """Closed-form d(fitted_drift)/d(state) for the one-hidden-layer tanh net.

    ``model.step_moments`` calls ``jax.jacobian(drift_fn)`` once per substep,
    which at audit time dominates: roughly ``(n_pilot + n_eval + 3) * horizon
    * n_substeps`` calls per replicate.  For this architecture the Jacobian is
    three small products, so the AD machinery buys nothing.

    With ``h = tanh(x W1 + b1) * mask`` and ``out = h W2 + b2``, where
    ``x = (concat(state, controls) - mean) / scale``:

        J[i, m] = sum_j W2[j, i] * (mask[j] * sech2[j]) * W1[m, j] / scale[m]

    Verified against ``jax.jacobian`` to ~1e-8 (float32 round-off), including
    with non-trivial ``in_mean`` / ``in_scale``.
    """
    x = (jnp.concatenate([state, controls]) - in_mean) / in_scale
    pre = x @ params["w1"] + params["b1"]
    gate = (1.0 - jnp.tanh(pre) ** 2) * mask                  # (WIDTH_MAX,)
    w1_state = params["w1"][: STATE_DIM, :]                    # (STATE_DIM, WIDTH_MAX)
    return (params["w2"].T * gate) @ w1_state.T / in_scale[:STATE_DIM]


def fast_step_moments(
    params, mask, state, controls, in_mean, in_scale, sigma, n_substeps: int
):
    """``model.step_moments`` with the analytic Jacobian substituted in.

    Same recursion, same zero-noise path, same returned ``(drift, cov)`` --
    only the per-substep Jacobian evaluation differs.
    """
    dt = 1.0 / n_substeps
    sigma_diff = model.diffusion_matrix(sigma)
    eye = jnp.eye(STATE_DIM)
    drift_fn = model._fitted_drift_fn(params, mask, controls, in_mean, in_scale)

    def body(carry, _):
        y, cov = carry
        jac = fitted_drift_jacobian(params, mask, y, controls, in_mean, in_scale)
        transition = eye + jac * dt
        cov = transition @ cov @ transition.T + sigma_diff * dt
        y = y + drift_fn(y) * dt
        return (y, cov), None

    (y_final, cov), _ = jax.lax.scan(
        body, (state, jnp.zeros((STATE_DIM, STATE_DIM))), None, length=n_substeps
    )
    return y_final - state, cov


def fast_path_residuals(params, mask, in_mean, in_scale, path, n_substeps: int):
    """``model.path_residuals`` using ``fast_step_moments``."""
    states, controls, increments = model.path_transitions(path)
    sigma = model.fitted_sigma(params)
    eff, cov = jax.vmap(
        lambda s, c: fast_step_moments(
            params, mask, s, c, in_mean, in_scale, sigma, n_substeps
        )
    )(states, controls)
    return increments - eff, cov


# ---------------------------------------------------------------------------
# A: global (state-independent) whitening covariance
# ---------------------------------------------------------------------------

def global_pilot_covariance(
    params, mask, in_mean, in_scale, pilot_paths: jnp.ndarray, n_substeps: int
) -> jnp.ndarray:
    """One pooled 2x2 covariance from every transition in a pilot bank.

    Uses the raw increment residual against the fitted zero-noise drift (no
    per-step tangent propagation), pooled across every path and every
    transition -- the generic "how jittery is the room on average" estimate
    that a diagnostic with no finite-step construction would use.
    """
    def one_path(path):
        states, controls, increments = model.path_transitions(path)
        eff = jax.vmap(
            lambda s, c: model.fitted_effective_drift(
                params, mask, s, c, in_mean, in_scale, n_substeps
            )
        )(states, controls)
        return increments - eff

    resid = jax.vmap(one_path)(pilot_paths)  # (n_pilot, horizon, 2)
    flat = resid.reshape((-1, STATE_DIM))
    mean = jnp.mean(flat, axis=0)
    centred = flat - mean
    cov = (centred.T @ centred) / flat.shape[0]
    # Floor the diagonal so a pilot bank with a near-zero coordinate-1 spread
    # (weak-propagation condition 1) still gives an invertible matrix; this
    # mirrors the SIGMA_FLOOR convention already used for the fitted sigma.
    cov = cov + jnp.eye(STATE_DIM) * (model.SIGMA_FLOOR ** 2)
    return cov


def cholesky_whiten(residual: jnp.ndarray, covariance: jnp.ndarray) -> jnp.ndarray:
    """Analytic 2x2 Cholesky whitening, shared by fixed and per-step covariances."""
    a = jnp.maximum(covariance[..., 0, 0], 1e-10)
    l11 = jnp.sqrt(a)
    l21 = covariance[..., 1, 0] / l11
    l22 = jnp.sqrt(jnp.maximum(covariance[..., 1, 1] - l21 * l21, 1e-10))
    z1 = residual[..., 0] / l11
    z2 = (residual[..., 1] - l21 * z1) / l22
    return jnp.stack((z1, z2), axis=-1)


def energy_from_z(z: jnp.ndarray) -> jnp.ndarray:
    """(full, coord1, coord2) mean-squared energy from a (horizon, 2) z array."""
    per_coordinate = jnp.mean(z * z, axis=-2)
    return jnp.concatenate((jnp.mean(per_coordinate, keepdims=True), per_coordinate))


# ---------------------------------------------------------------------------
# B: theoretical (working chi-square) calibration of the tangent coordinate
# ---------------------------------------------------------------------------

def theoretical_pvalue(energy: jnp.ndarray, horizon: int) -> jnp.ndarray:
    """Upper-tail chi-square p-value for a mean-squared-z energy statistic.

    Treats each of the ``horizon`` (full) or ``horizon`` (per-coordinate)
    whitened values entering ``energy`` as iid standard normal -- the
    "textbook table" a user would consult if they trusted the tangent
    linearization's own reported scale exactly, with no matched-simulation
    check.  ``energy[0]`` (full) pools both coordinates, so its reference
    degrees of freedom is ``2 * horizon``; the two per-coordinate entries use
    ``horizon``.
    """
    dof = jnp.array([2 * horizon, horizon, horizon], dtype=jnp.float64)
    stat = energy.astype(jnp.float64) * dof
    return jstats.chi2.sf(stat, dof)


# ---------------------------------------------------------------------------
# Deviations, applied to the frozen fitted simulator
# ---------------------------------------------------------------------------

def advance_fitted_step(
    drift_fn, state: jnp.ndarray, noise: jnp.ndarray, loading: jnp.ndarray
) -> jnp.ndarray:
    """One observation step with noise ``(n_substeps, noise_dim)`` mapped by
    a fixed ``(state_dim, noise_dim)`` loading held across substeps."""
    n_substeps = noise.shape[0]
    dt = 1.0 / n_substeps
    sqrt_dt = math.sqrt(dt)

    def body(y, eps):
        y = y + drift_fn(y) * dt
        return y + sqrt_dt * (loading @ eps), None

    y_final, _ = jax.lax.scan(body, state, noise)
    return y_final


def simulate_fitted_path_with_deviation(
    key: jax.Array,
    params,
    mask,
    in_mean,
    in_scale,
    prefix: jnp.ndarray,
    n_steps: int,
    n_substeps: int,
    loading_fn,
):
    """One held-out path from the frozen fitted simulator under a deviation.

    ``loading_fn(base_sigma) -> (state_dim, noise_dim)`` builds the loading
    matrix; the null case passes the base model's rank-one loading unchanged.
    """
    horizon = n_steps - model.LOOKBACK
    sigma = model.fitted_sigma(params)
    loading = loading_fn(sigma)
    noise_dim = loading.shape[1]
    noise = jax.random.normal(key, (horizon, n_substeps, noise_dim))

    def body(carry, eps):
        win, k = carry
        controls = model.controls_from_window(win, k)
        drift = model._fitted_drift_fn(params, mask, controls, in_mean, in_scale)
        y_next = advance_fitted_step(drift, win[-1], eps, loading)
        return (jnp.concatenate([win[1:], y_next[None, :]], axis=0), k + 1), y_next

    _, states = jax.lax.scan(body, (prefix, jnp.array(model.LOOKBACK)), noise)
    return jnp.concatenate([prefix, states], axis=0)


def null_loading(sigma: jnp.ndarray) -> jnp.ndarray:
    """The base model's own rank-one loading: noise into coordinate 2 only."""
    return jnp.array([[0.0], [1.0]]) * sigma


def targeted_loading(extra_sigma1: float):
    """Deviation 1: small independent noise added to the unforced coordinate,
    on top of the base loading into coordinate 2.  Two independent noise
    channels, so the coordinate-2 marginal is untouched."""
    def build(sigma: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[extra_sigma1, 0.0], [0.0, sigma]])
    return build


def uniform_scale_loading(scale: float):
    """Deviation 2: the whole (rank-one) loading multiplied by one constant,
    same factor in every state and every direction it already reaches."""
    def build(sigma: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[0.0], [1.0]]) * (sigma * scale)
    return build
