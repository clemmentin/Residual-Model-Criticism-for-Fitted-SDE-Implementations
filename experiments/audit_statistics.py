"""Numerical primitives shared by the audit and reporting scripts.

Callers own their pilot split and tail choice. Rank inputs must be finite.
"""
import math

import numpy as np


def upper_rank(observed: float, reference: np.ndarray) -> float:
    """Plus-one upper rank; reject invalid scores instead of dropping them."""
    reference = np.asarray(reference, dtype=float)
    if not np.isfinite(observed):
        raise ValueError("Non-finite observed rank score.")
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("Rank reference must be a nonempty one-dimensional array.")
    if not np.isfinite(reference).all():
        raise ValueError("Non-finite rank reference.")
    return float((1 + np.sum(reference >= observed)) / (reference.size + 1))


def standardize(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - center) / max(float(scale), 1e-12)


def departure(values: np.ndarray, center: float, scale: float, tail: str) -> np.ndarray:
    standardized = standardize(values, center, scale)
    if tail == "centered":
        return np.abs(standardized)
    if tail == "upper":
        return np.maximum(standardized, 0.0)
    raise ValueError(f"Unknown tail: {tail}")


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054,
                    *, clip: bool = True) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    rate = successes / n
    denominator = 1.0 + z * z / n
    center = (rate + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * n)) / n) / denominator
    low, high = center - half, center + half
    return (max(0.0, low), min(1.0, high)) if clip else (low, high)
