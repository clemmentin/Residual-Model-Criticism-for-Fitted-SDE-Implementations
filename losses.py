import jax
import jax.numpy as jnp
import jax.scipy.stats
from numpy.typing import NDArray
from typing import Tuple, Dict, Any


_MASK_EPS = 1e-8
_STD_FLOOR = 1e-4
_VAR_FLOOR = 1e-8
_PHYS_LOWER_BOUNDS = jnp.array([1e-6, -20.0])
_PHYS_UPPER_BOUNDS = jnp.array([1.0 - 1e-6, 0.0])


def _prepare_observations_and_mask(
    ys: NDArray, observed_states: NDArray, keep_steps: int
) -> Tuple[NDArray, NDArray, NDArray]:
    ys_prev = ys[:, :-1, :]
    obs_next = observed_states[:, 1:, :]

    max_steps = ys_prev.shape[1]
    step_indices = jnp.arange(max_steps)
    mask = jnp.where(step_indices < keep_steps, 1.0, 0.0)
    return ys_prev, obs_next, mask


def _masked_time_average(values: NDArray, mask: NDArray) -> NDArray:
    """Average over batch first, then take a masked mean over time."""
    values_batch_mean = jnp.mean(values, axis=0) if values.ndim > 1 else values
    denom = jnp.sum(mask) + _MASK_EPS
    return jnp.sum(values_batch_mean * mask) / denom


def _compute_transition_statistics(
    model: Any, ys: NDArray, ts: NDArray, controls: NDArray,
    fast_euler: bool = False,
    country_ids: NDArray = None,
    substep_dt: float = 0.1,
    num_substeps: int = 10,
) -> Tuple[NDArray, NDArray]:
    """One-step transition moments via Euler integration.

    fast_euler=True  : single Euler step at dt=1 day — 10x faster, O(dt²) error.
    fast_euler=False : 10 micro-steps at dt=0.1 day  — original behaviour.

    Returns (predicted_next, predicted_std).
    """

    def get_step_prediction_fast(y_p, t_p, c_p, cid):
        del t_p
        drift = model._calculate_drift(y_p, c_p, country_idx=cid)
        diffusion = model._calculate_diffusion(y_p, c_p, country_idx=cid)
        y_final = y_p + drift                       # dt = 1.0 day
        var_final = diffusion ** 2                  # Var over dt = 1.0 day
        return y_final, jnp.sqrt(jnp.maximum(var_final, 1e-8))

    def get_step_prediction_full(y_p, t_p, c_p, cid):
        del t_p
        dt = substep_dt
        num_steps = num_substeps

        # Physical clip bounds for projected EM (matches inference _clip_physical)
        _nm = model.norm_mean
        _ns = model.norm_std

        def _clip_norm(y):
            """Denormalise → clip to K → re-normalise."""
            phys = y * _ns + _nm
            phys = jnp.clip(phys, _PHYS_LOWER_BOUNDS, _PHYS_UPPER_BOUNDS)
            return (phys - _nm) / _ns

        def step_fn(carry, _):
            y_curr, var_curr = carry
            drift = model._calculate_drift(y_curr, c_p, country_idx=cid)
            diffusion = model._calculate_diffusion(y_curr, c_p, country_idx=cid)
            y_next = _clip_norm(y_curr + drift * dt)
            var_next = var_curr + (diffusion ** 2) * dt
            return (y_next, var_next), None

        (y_final, var_final), _ = jax.lax.scan(
            step_fn,
            (y_p, jnp.zeros_like(y_p)),
            jnp.arange(num_steps),
        )
        return y_final, jnp.sqrt(jnp.maximum(var_final, 1e-8))

    get_step_prediction = get_step_prediction_fast if fast_euler else get_step_prediction_full

    _cids = country_ids if country_ids is not None else jnp.full((ys.shape[0],), -1, dtype=jnp.int32)
    batch_predict = jax.vmap(
        jax.vmap(get_step_prediction, in_axes=(0, 0, 0, None)), in_axes=(0, None, 0, 0)
    )
    return batch_predict(ys[:, :-1, :], ts[:-1], controls[:, :-1, :], _cids)


def _compute_huber_loss(
    predicted: NDArray, observed: NDArray, mask: NDArray
) -> NDArray:
    residuals = predicted - observed

    delta = 1.0
    abs_r = jnp.abs(residuals)
    quadratic = jnp.minimum(abs_r, delta)
    linear = abs_r - quadratic
    huber_elementwise = 0.5 * quadratic ** 2 + delta * linear

    huber_per_step = jnp.mean(huber_elementwise, axis=-1)  # (B, T)
    return _masked_time_average(huber_per_step, mask)


def _compute_stochastic_nll(
    predicted_mean: NDArray,
    predicted_std: NDArray,
    observed: NDArray,
    mask: NDArray,
    stochastic_idx: int,
) -> Tuple[NDArray, NDArray]:
    """Gaussian transition NLL for the stochastic dimension."""
    obs = observed[..., stochastic_idx]
    mu = predicted_mean[..., stochastic_idx]
    std = jnp.maximum(predicted_std[..., stochastic_idx], _STD_FLOOR)

    z_scores = (obs - mu) / std
    nll = 0.5 * z_scores ** 2 + jnp.log(std) + 0.5 * jnp.log(2 * jnp.pi)

    loss_nll = _masked_time_average(nll, mask)
    return loss_nll, z_scores


def _compute_zscore_wd_loss(z_scores: NDArray, keep_steps: int) -> NDArray:
    z_valid_flat = z_scores[:, :keep_steps].flatten()
    z_valid_flat = jnp.nan_to_num(z_valid_flat, nan=0.0, posinf=5.0, neginf=-5.0)

    n_total = z_valid_flat.shape[0]
    z_sorted = jnp.sort(z_valid_flat)
    probs = (jnp.arange(n_total) + 0.5) / n_total
    target_quantiles = jax.scipy.stats.norm.ppf(probs)
    return jnp.mean(jnp.abs(z_sorted - target_quantiles))


def _masked_moment(values: NDArray, mask: NDArray, center: NDArray = None) -> NDArray:
    mean = _masked_time_average(values, mask)
    if center is None:
        return mean
    dev = jnp.mean((values - center) ** 2, axis=0) if values.ndim > 1 else (values - center) ** 2
    return _masked_time_average(dev, mask)


def loss(
    model: Any,
    ts: NDArray,
    ys: NDArray,
    controls: NDArray,
    targets: NDArray,
    training_phase: int = 1,   # unused — kept for API compatibility
    keep_steps: int = 1000,
    fast_euler: bool = False,
    s_loss_scale: float = 1e4,
    zscore_wd_weight: float = 0.0,
    country_ids: NDArray = None,
    substep_dt: float = 0.1,
    num_substeps: int = 10,
) -> Tuple[NDArray, Dict[str, NDArray]]:
    """
    One-step transition loss on observed states.

    Deterministic dimensions use Huber loss on the next state.
    The stochastic dimension (log-I) uses a Gaussian transition NLL.

    targets is expected to be the observed state path aligned with ys.
    fast_euler: if True, uses a single dt=1 Euler approximation. This is an
    explicitly approximate mode and need not match projected substep inference.
    """
    del training_phase

    ys_prev, obs_next, mask = _prepare_observations_and_mask(ys, targets, keep_steps)

    predicted_next, predicted_std = _compute_transition_statistics(
        model,
        ys,
        ts,
        controls,
        fast_euler=fast_euler,
        country_ids=country_ids,
        substep_dt=substep_dt,
        num_substeps=num_substeps,
    )

    state_dim = ys.shape[-1]
    stochastic_idx = state_dim - 1

    if state_dim > 1:
        deterministic_loss = _compute_huber_loss(
            predicted_next[..., :-1],
            obs_next[..., :-1],
            mask,
        )
    else:
        deterministic_loss = jnp.zeros(())

    transition_nll, z_scores = _compute_stochastic_nll(
        predicted_next,
        predicted_std,
        obs_next,
        mask,
        stochastic_idx,
    )
    zscore_wd_loss = _compute_zscore_wd_loss(z_scores, keep_steps)

    # Scale up S loss: daily ΔS ~ 1e-4 in norm space, Huber ≈ 0.5*(1e-4)² ≈ 5e-9
    # NLL is O(1). s_loss_scale (from caller / Config.S_LOSS_SCALE) brings S loss to comparable magnitude.
    total_loss = s_loss_scale * deterministic_loss + transition_nll + zscore_wd_weight * zscore_wd_loss

    transition_step = predicted_next - ys_prev
    sigma_values = predicted_std[..., stochastic_idx]
    drift_values = transition_step[..., stochastic_idx]
    drift_mean = _masked_moment(drift_values, mask)
    sigma_mean = _masked_moment(sigma_values, mask)
    bi_z_mean = _masked_moment(z_scores, mask)
    bi_z_var = _masked_moment(z_scores, mask, bi_z_mean)

    diagnostics = {
        "drift_std": jnp.sqrt(jnp.maximum(_masked_moment(drift_values, mask, drift_mean), 0.0)),
        "drift_mean": drift_mean,
        "drift_max": jnp.max(drift_values),
        "drift_min": jnp.min(drift_values),
        "loss_mse": deterministic_loss,
        "loss_nll": transition_nll,
        "loss_zwd": zscore_wd_loss,
        "sigma_mean": sigma_mean,
        "base_sigma_mean": jnp.zeros(()),
        "bi_z_mean": bi_z_mean,
        "bi_z_std": jnp.sqrt(jnp.maximum(bi_z_var, 0.0)),
        "loss_bi": transition_nll,
    }

    return total_loss, diagnostics
