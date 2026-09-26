import jax
import jax.numpy as jnp
import equinox as eqx
import signax
import logging
from typing import Dict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_signature_size(dim: int, depth: int, augment_time: bool = True, lead_lag: bool = True) -> int:
    """
    Calculates the dimension of a signature of a given depth for a path of a given dimension.

    Args:
        dim: The dimension of the path.
        depth: The depth of the signature.
        augment_time: Whether time augmentation is applied, which adds a dimension to the path.
        lead_lag: Whether Lead-Lag transform is applied, which doubles the dimension.

    Returns:
        The total dimension of the flattened signature.
    """
    # If time is augmented, the effective dimension of the path increases by 1.
    effective_dim = dim + 1 if augment_time else dim

    # Lead-Lag doubles the channel dimension
    if lead_lag:
        effective_dim = effective_dim * 2

    # The size of the signature is the sum of the sizes of each level's tensor.
    # Level i has size effective_dim^i.
    size = 0
    for i in range(1, depth + 1):
        size += effective_dim**i
    return size


class JAXSignatureExtractor(eqx.Module):
    """
    JAX-based Observation Layer.
    Uses 'signax' for path signatures. Fully compiled via XLA.
    Supports optional Lead-Lag transform for capturing quadratic variation.
    """

    depth: int = eqx.field(static=True)
    augment_time: bool = eqx.field(static=True)
    lead_lag: bool = eqx.field(static=True)

    def __init__(self, depth: int = 3, augment_time: bool = True, lead_lag: bool = True):
        self.depth = depth
        self.augment_time = augment_time
        self.lead_lag = lead_lag

    def __call__(self, window: jnp.ndarray) -> jnp.ndarray:
        """
        Compute signature for a single window of shape (Length, Channels).
        For batched input (Batch, Length, Channels), use jax.vmap explicitly
        at the call site to avoid Python-level shape branching inside JIT.
        """
        return self._compute_single(window)

    @eqx.filter_jit
    def compute_rolling(self, path: jnp.ndarray, window_size: int) -> jnp.ndarray:
        """
        Computes rolling signatures over a long path efficiently.
        Instead of re-computing the signature for each window from scratch,
        this uses jax.vmap over sliding windows. While not a full Chen's identity
        update, it allows XLA to optimize the batched computation significantly
        better than a Python for-loop.
        
        Args:
            path: Shape (T, C)
            window_size: The length of the sliding window L.
        Returns:
            Signatures of shape (T - window_size + 1, Signature_Dim)
        """
        T, C = path.shape
        num_windows = T - window_size + 1
        
        # Extract sliding windows using jax.lax.dynamic_slice in a scan or vmap
        # A more efficient way in JAX is to use vmap with dynamic_slice
        def get_window(start_idx):
            return jax.lax.dynamic_slice(path, (start_idx, 0), (window_size, C))
            
        windows = jax.vmap(get_window)(jnp.arange(num_windows))
        
        # Compute signatures for all windows in parallel
        return jax.vmap(self._compute_single)(windows)

    @staticmethod
    def _lead_lag_transform(path: jnp.ndarray) -> jnp.ndarray:
        """
        Lead-Lag transform for a discrete path.

        For X = (x_0, x_1, ..., x_T), constructs the interleaved path:
          (x_0, x_0), (x_1, x_0), (x_1, x_1), (x_2, x_1), ..., (x_T, x_T)

        This doubles the channel dimension and produces a path of length 2T+1,
        enabling the signature to capture quadratic variation.

        Args:
            path: Shape (T+1, C)
        Returns:
            Lead-Lag path of shape (2*T+1, 2*C)
        """
        # T+1 points => T increments
        # Lead path: x_0, x_1, x_1, x_2, x_2, ..., x_T
        # Lag  path: x_0, x_0, x_1, x_1, x_2, ..., x_T
        # Interleaved time points: 2*T + 1

        T_plus_1, C = path.shape

        # Build lead: at odd index 2k+1 we advance to x_{k+1}, at even 2k we stay at x_k
        # Build lag:  at odd index 2k+1 we stay at x_k,       at even 2k we are at x_k
        # Total length = 2*T + 1 where T = T_plus_1 - 1

        # Indices for the 2T+1 output points
        n_out = 2 * (T_plus_1 - 1) + 1

        # For even indices i=2k: lead[i] = x_k, lag[i] = x_k
        # For odd  indices i=2k+1: lead[i] = x_{k+1}, lag[i] = x_k
        idx = jnp.arange(n_out)
        lead_idx = (idx + 1) // 2   # 0,1,1,2,2,3,3,...,T
        lag_idx  = idx // 2          # 0,0,1,1,2,2,3,...,T

        lead = path[lead_idx]        # (2T+1, C)
        lag  = path[lag_idx]          # (2T+1, C)

        return jnp.concatenate([lead, lag], axis=-1)  # (2T+1, 2C)

    def _compute_single(self, path: jnp.ndarray) -> jnp.ndarray:
        # path shape: (Length, Channels)

        # 1. Time Augmentation
        if self.augment_time:
            T, C = path.shape
            # Create time channel [0, 1]
            t = jnp.linspace(0.0, 1.0, T)
            # Concatenate: (T, C) -> (T, C+1)
            path = jnp.concatenate([t[:, None], path], axis=-1)

        # 2. Lead-Lag Transform (doubles channel dimension, increases path length)
        if self.lead_lag:
            path = self._lead_lag_transform(path)

        # 3. Compute Signature
        # signax.signature returns a list of terms (level 1, level 2, ...)
        # We flatten them into a single feature vector
        signature_terms = signax.signature(path, self.depth)

        # Flatten all levels into one 1D array
        return jnp.concatenate([term.ravel() for term in signature_terms])

    def get_output_dim(self, input_channels: int) -> int:
        """Helper to calculate output dimension size using the standalone function."""
        return get_signature_size(input_channels, self.depth, self.augment_time, self.lead_lag)


# ==============================================================================
# Compatibility Wrapper (Maintains API for your existing code)
# ==============================================================================
class DriftTrajectoryAnalyzer:
    def __init__(self, feature_dim: int = 3):
        # Initialize JAX extractor
        self.extractor = JAXSignatureExtractor(depth=2, augment_time=True)
        self.output_dim = self.extractor.get_output_dim(feature_dim)
        logger.info(f"JAX Signature Extractor Ready. Output Dim: {self.output_dim}")

    def analyze_drift(self, reference_window, current_window) -> Dict[str, float]:
        # Convert inputs to JAX arrays if they aren't already
        ref = jnp.array(reference_window)
        curr = jnp.array(current_window)

        # Robustness check
        if len(ref) < 5 or len(curr) < 5:
            return {"drift_magnitude": 0.0, "drift_vector": np.zeros(self.output_dim)}

        # Compute
        sig_ref = self.extractor(ref)
        sig_curr = self.extractor(curr)

        # Distance
        diff = sig_curr - sig_ref
        magnitude = jnp.linalg.norm(diff).item()

        return {
            "drift_magnitude": magnitude,
            "drift_vector": np.array(diff),  # Convert back to numpy for compatibility
            "velocity": 0.0,
        }


if __name__ == "__main__":

    print("Testing JAX/Signax Extractor on ...")

    # Fake data
    key = jax.random.PRNGKey(0)
    fake_window = jax.random.normal(key, (30, 3))

    extractor = JAXSignatureExtractor(depth=3, augment_time=True)

    # Warmup / Compile
    sig = extractor(fake_window)

    print(f"Input Shape: {fake_window.shape}")
    print(f"Signature Shape: {sig.shape}")
    print(f"Calculated Dim: {extractor.get_output_dim(3)}")
    print(
        f"Standalone function check: {get_signature_size(dim=3, depth=3, augment_time=True, lead_lag=True)}"
    )
    assert sig.shape[0] == get_signature_size(dim=3, depth=3, augment_time=True, lead_lag=True)
    print("Success! JAX is running and dimensions match.")
