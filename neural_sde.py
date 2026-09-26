import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import logging
from numpy.typing import NDArray
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical SIR baseline constants (centres of the neural perturbation bands)
_SIR_BETA_BASE = 0.30
_SIR_BETA_SCALE = 0.25
_SIR_GAMMA_BASE = 0.07
_SIR_GAMMA_SCALE = 0.05

# Numerical floor for log-infected: Z = log(I) ∈ [Z_LOG_I_MIN, 0].
# Prevents underflow in exp(Z) and keeps I ∈ (e^{-20}, 1) as claimed in the paper.
_Z_LOG_I_MIN: float = -20.0


class DriftNet(eqx.Module):
    """
    Neural perturbation network for time-varying SIR parameters.

    Input : [y_norm (state_size), signatures (total_sig_size)]
    Output: [raw_δβ(t), raw_δγ(t)]  — unconstrained; tanh scaling applied in caller.

    Default MLP initialisation is intentionally used: at init all weights are
    near zero so tanh(raw) ≈ 0, giving β ≈ β_base, γ ≈ γ_base.
    """

    mlp: eqx.nn.MLP

    def __init__(self, in_size: int, key: NDArray, width: int = 32, depth: int = 2):
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=2,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, x: NDArray) -> NDArray:
        return self.mlp(x)  # raw — tanh scaling applied in _calculate_drift


class SignatureLinearDrift(eqx.Module):
    """
    Signature-linear perturbation: β(t), γ(t) are *linear* functionals of
    the path signature — no hidden layers, no state input.

        raw = W @ sig + b        (W: 2×sig_dim, b: 2)
        β = β_base + β_scale · tanh(raw[0])
        γ = γ_base + γ_scale · tanh(raw[1])

    Theoretical motivation: the Universal Approximation Theorem for
    signatures guarantees that linear functionals of truncated signatures
    are already universal approximators of continuous path functionals.
    The nonlinearity is captured *inside* the signature, not in the model.
    """

    weight: NDArray   # (2, sig_dim)
    bias: NDArray     # (2,)
    sig_dim: int = eqx.field(static=True)

    def __init__(self, sig_dim: int, key: NDArray):
        self.sig_dim = sig_dim
        # Small init so tanh(W@sig + b) ≈ 0 → β ≈ β_base at start
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (2, sig_dim)) * 0.01
        self.bias = jnp.zeros(2)

    def __call__(self, sig: NDArray) -> NDArray:
        """sig: (sig_dim,) → (2,) raw perturbations."""
        return self.weight @ sig + self.bias


class DiffusionNet(eqx.Module):
    """
    State-dependent volatility: σ(y, controls).

    Input : [y_norm (state_size), signatures (total_sig_size)]
    Output: scalar log_sigma — σ(y,c) = exp(log_sigma).

    Small MLP (width=16, depth=1) to avoid overfitting the noise structure.
    Initialized so output ≈ log_sigma_init (default -3 → σ ≈ 0.05).
    """

    mlp: eqx.nn.MLP
    log_sigma_init: float = eqx.field(static=True)

    def __init__(self, in_size: int, key: NDArray, log_sigma_init: float = -3.0):
        self.log_sigma_init = log_sigma_init
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=16,
            depth=1,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, x: NDArray) -> NDArray:
        # At init, MLP output ≈ 0 → σ ≈ exp(log_sigma_init)
        return self.log_sigma_init + self.mlp(x)[0]


class StepInterpolation(eqx.Module):
    ts: NDArray
    ys: NDArray

    def evaluate(self, t: float) -> NDArray:
        t0 = self.ts[0]
        dt = self.ts[1] - self.ts[0]
        idx = jnp.clip(jnp.floor((t - t0) / dt).astype(int), 0, len(self.ts) - 1)
        return self.ys[idx]


class NeuralSDE(eqx.Module):
    """
    Neural SDE — baseline-perturbation SIR architecture.

    Drift  :  β(t) = β_base  + β_scale  * tanh(DriftNet[0])
              γ(t) = γ_base  + γ_scale  * tanh(DriftNet[1])
              dS/dt = −β(t) · S · I
              dZ/dt =  β(t)·S − γ(t) − 0.5·σ²   (Z = log I, Itô)

    Diffusion: σ = exp(log_sigma)  — learned scalar volatility.

    State  : [S, Z=log(I)]  z-score normalised for network input.
    """

    drift_net: DriftNet           # None when signature_linear_drift=True
    sig_linear_drift: SignatureLinearDrift  # None when signature_linear_drift=False
    diffusion_net: DiffusionNet    # None when constant σ mode
    log_sigma: NDArray
    use_diffusion_net: bool = eqx.field(static=True)
    signature_linear_drift: bool = eqx.field(static=True)
    sigma_country_scale: bool = eqx.field(static=True)
    n_countries: int = eqx.field(static=True)
    country_log_scale: NDArray
    use_per_country_beta: bool = eqx.field(static=True)
    beta_country_bases: tuple = eqx.field(static=True)   # per-country NLS β estimates (python tuple, immutable)
    gamma_country_bases: tuple = eqx.field(static=True)  # per-country NLS γ estimates
    norm_mean: NDArray
    norm_std: NDArray
    # Transformation matrices for the normalised/whitened state space.
    # For z-score (default): W = diag(1/std), W_inv = diag(std)
    # For ZCA pre-whitening : W and W_inv are full 2×2 matrices.
    # Encoding (physical → model):  z = W   @ (x − mean)
    # Decoding (model → physical):  x = W_inv @ z + mean
    norm_whiten_matrix: NDArray    # W    shape (state, state)
    norm_dewhiten_matrix: NDArray  # W_inv shape (state, state)
    beta_base: float = eqx.field(static=True)
    gamma_base: float = eqx.field(static=True)
    gamma_scale: float = eqx.field(static=True)  # 0.0 when FREEZE_GAMMA=True

    state_size: int = eqx.field(static=True)
    macro_size: int = eqx.field(static=True)
    signature_sizes: List[int] = eqx.field(static=True)
    total_signature_size: int = eqx.field(static=True)

    def __init__(
        self,
        signature_sizes: List[int],
        macro_size: int,
        state_size: int,
        key: NDArray,
        norm_mean: NDArray = None,
        norm_std: NDArray = None,
        beta_base: float = _SIR_BETA_BASE,
        gamma_base: float = _SIR_GAMMA_BASE,
        **kwargs,
    ):
        self.state_size = int(state_size)
        self.macro_size = macro_size
        self.signature_sizes = signature_sizes
        self.total_signature_size = sum(signature_sizes)
        self.norm_mean = jnp.array(norm_mean if norm_mean is not None else [0.0, 0.0])
        _std = jnp.array(norm_std if norm_std is not None else [1.0, 1.0])
        self.norm_std = _std
        self.beta_base = float(beta_base)
        self.gamma_base = float(gamma_base)
        self.gamma_scale = 0.0 if kwargs.get("freeze_gamma", False) else _SIR_GAMMA_SCALE

        # Transformation matrices.  Default reproduces z-score behaviour exactly:
        #   W     = diag(1/std)  →  z = (x−mean)/std
        #   W_inv = diag(std)    →  x = z*std + mean
        _dw = kwargs.get("norm_dewhiten_matrix", None)
        _fw = kwargs.get("norm_whiten_matrix", None)
        self.norm_dewhiten_matrix = jnp.array(_dw) if _dw is not None else jnp.diag(_std)
        self.norm_whiten_matrix   = jnp.array(_fw) if _fw is not None else jnp.diag(1.0 / _std)

        net_in_size = self.state_size + self.total_signature_size + macro_size
        key, diff_key = jax.random.split(key)
        _drift_width = kwargs.get("drift_width", 32)
        self.signature_linear_drift = kwargs.get("signature_linear_drift", False)
        if self.signature_linear_drift:
            self.drift_net = None
            self.sig_linear_drift = SignatureLinearDrift(self.total_signature_size, key)
        else:
            self.drift_net = DriftNet(net_in_size, key, width=_drift_width, depth=2)
            self.sig_linear_drift = None
        _log_sigma_init = kwargs.get("log_sigma_init", -2.5)
        self.log_sigma = jnp.array(_log_sigma_init)

        self.use_diffusion_net = kwargs.get("use_diffusion_net", False)
        if self.use_diffusion_net:
            self.diffusion_net = DiffusionNet(net_in_size, diff_key, log_sigma_init=-3.0)
        else:
            self.diffusion_net = None

        # Level 2: per-country σ scaling
        self.sigma_country_scale = kwargs.get("sigma_country_scale", False)
        _n_countries = kwargs.get("n_countries", 0)
        self.n_countries = _n_countries
        if self.sigma_country_scale and _n_countries > 0:
            self.country_log_scale = jnp.zeros(_n_countries)
        else:
            self.country_log_scale = jnp.zeros(0)

        # Per-country NLS β/γ bases (stored as static python tuples so JAX traces them as constants)
        self.use_per_country_beta = kwargs.get("use_per_country_beta", False)
        _bcb = kwargs.get("beta_country_bases", None)
        _gcb = kwargs.get("gamma_country_bases", None)
        self.beta_country_bases = tuple(float(x) for x in _bcb) if _bcb is not None else ()
        self.gamma_country_bases = tuple(float(x) for x in _gcb) if _gcb is not None else ()

    # ── Geometry ───────────────────────────────────────────────────────────────

    brownian_size: int = eqx.field(static=True, default=1)

    @property
    def sigma(self) -> NDArray:
        """Learned scalar volatility: σ = exp(log_sigma)."""
        return jnp.exp(self.log_sigma)

    def _raw_drift_params(self, y: NDArray, controls: NDArray) -> NDArray:
        if self.signature_linear_drift:
            sig = controls[:self.total_signature_size]
            return self.sig_linear_drift(sig)

        net_input = jnp.concatenate([y, controls])
        return self.drift_net(net_input)

    def _get_beta_gamma(
        self,
        y: NDArray,
        controls: NDArray,
        country_idx=None,
    ) -> tuple[NDArray, NDArray]:
        """Return the current SIR beta/gamma implied by the drift model."""
        raw = self._raw_drift_params(y, controls)

        if self.use_per_country_beta and country_idx is not None and len(self.beta_country_bases) > 0:
            beta_country_bases = jnp.array(self.beta_country_bases)
            gamma_country_bases = jnp.array(self.gamma_country_bases)
            safe_idx = jnp.clip(country_idx, 0, len(self.beta_country_bases) - 1)
            beta_base = jnp.where(country_idx >= 0, beta_country_bases[safe_idx], self.beta_base)
            gamma_base = jnp.where(country_idx >= 0, gamma_country_bases[safe_idx], self.gamma_base)
        else:
            beta_base = self.beta_base
            gamma_base = self.gamma_base

        beta = beta_base + _SIR_BETA_SCALE * jnp.tanh(raw[0])
        gamma = gamma_base + self.gamma_scale * jnp.tanh(raw[1])
        return jnp.maximum(beta, 0.0), jnp.maximum(gamma, 1e-2)

    def _scalar_sigma(self, y: NDArray, controls: NDArray, country_idx=None) -> NDArray:
        """Return physical log-I volatility before normalization whitening."""
        if self.use_diffusion_net:
            net_input = jnp.concatenate([y, controls])
            sigma = jnp.exp(self.diffusion_net(net_input))
        else:
            sigma = self.sigma

        if self.sigma_country_scale and country_idx is not None and self.n_countries > 0:
            safe_idx = jnp.clip(country_idx, 0, self.n_countries - 1)
            scale = jnp.where(
                country_idx >= 0,
                jnp.exp(self.country_log_scale[safe_idx]),
                1.0,
            )
            sigma = sigma * scale

        return sigma

    # ── Drift ──────────────────────────────────────────────────────────────────

    def _calculate_drift(self, y: NDArray, controls: NDArray, country_idx=None) -> NDArray:
        """
        Log-Normal SIR drift.
        controls vector includes: [Signatures..., Macro_Controls...]
        country_idx: scalar int; if use_per_country_beta is True uses per-country NLS base.
        """
        beta, gamma = self._get_beta_gamma(y, controls, country_idx=country_idx)
        sigma = self._scalar_sigma(y, controls, country_idx=country_idx)

        # Decode: x = W_inv @ z + mean  (works for both z-score and ZCA)
        x_phys = self.norm_dewhiten_matrix @ y + self.norm_mean
        S = jnp.clip(x_phys[0], 1e-6, 1.0 - 1e-6)
        Z = jnp.clip(x_phys[1], _Z_LOG_I_MIN, 0.0)
        I = jnp.exp(Z)

        # Physical derivatives (Itô SIR log-normal)
        dS = -beta * S * I
        dZ = beta * S - gamma - 0.5 * (sigma ** 2)

        # Encode: f_z = W @ f_x  (maps physical drift back to model space)
        return self.norm_whiten_matrix @ jnp.array([dS, dZ])

    def _calculate_diffusion(self, y: NDArray, controls: NDArray, country_idx: NDArray = None) -> NDArray:
        """
        Diffusion coefficient. Two modes:
        - Constant: σ = exp(log_sigma)
        - State-dependent: σ(y, c) = exp(DiffusionNet(y, c))
        Optional per-country scaling (Level 2): σ_c = σ * exp(country_log_scale[c]).
        No noise on S (index 0).
        """
        sigma = self._scalar_sigma(y, controls, country_idx=country_idx)

        # Level 2: per-country σ scaling

        # Physical diffusion vector: [0, σ]  (no noise on S).
        # Encode to model space: g_z = W @ [0, σ]
        return self.norm_whiten_matrix @ jnp.array([0.0, sigma])

    # ── diffrax interface ──────────────────────────────────────────────────────

    def drift(self, t: float, y: NDArray, args) -> NDArray:
        return self._calculate_drift(y, args.evaluate(t))

    def diffusion(self, t: float, y: NDArray, args) -> NDArray:
        """Vector field for ControlTerm: returns (state_size, brownian_size) matrix."""
        diff_vec = self._calculate_diffusion(y, args.evaluate(t))
        return diff_vec[:, None]  # (2,) → (2, 1)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def get_diagnostics(
        self, t: float, y: NDArray, args: NDArray
    ) -> Dict[str, NDArray]:
        """
        Returns drift and instantaneous β(t), γ(t), σ(y,c).
        """
        drift_val = self._calculate_drift(y, args)
        diffusion_val = self._calculate_diffusion(y, args)
        
        beta_t, gamma_t = self._get_beta_gamma(y, args)
        sigma_val = self._scalar_sigma(y, args)

        return {
            "drift": drift_val,
            "sigma": diffusion_val, 
            "beta_t": beta_t,
            "gamma_t": gamma_t,
            "real_sigma": sigma_val,
            "gate": jnp.zeros(()),
            "base_sigma": jnp.zeros_like(drift_val),
        }

    # ── Forward pass (diffrax solve — SDE or ODE depending on key) ─────────────

    def __call__(
        self,
        ts: NDArray,
        y0: NDArray,
        controls: NDArray,
        key: NDArray = None,
    ) -> NDArray:
        t0, t1 = ts[0], ts[-1]
        dt0 = 0.5
        control_function = StepInterpolation(ts=ts, ys=controls)

        if key is not None:
            # ── SDE mode: Euler-Maruyama with Brownian motion ─────────────
            brownian = diffrax.VirtualBrownianTree(
                t0=t0, t1=t1, tol=dt0 / 2,
                shape=(self.brownian_size,), key=key,
            )
            term = diffrax.MultiTerm(
                diffrax.ODETerm(self.drift),
                diffrax.ControlTerm(self.diffusion, brownian),
            )
            solver = diffrax.Euler()
        else:
            # ── Deterministic ODE fallback ─────────────────────────────────
            term = diffrax.ODETerm(self.drift)
            solver = diffrax.Tsit5()

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            args=control_function,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys
