"""Interval, trajectory and binomial calculations used by four_step.py.

Copied from the corresponding research calculations without changing the
numerical recursions. Only the functions needed by this experiment are kept.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent


def outward(lo, hi):
    return np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)


def product(al, ah, bl, bh):
    parts = (al*bl, al*bh, ah*bl, ah*bh)
    return outward(np.minimum.reduce(np.broadcast_arrays(*parts)),
                   np.maximum.reduce(np.broadcast_arrays(*parts)))


def squared(lo, hi):
    lower = np.where((lo <= 0) & (hi >= 0), 0., np.minimum(lo*lo, hi*hi))
    low, high = outward(lower, np.maximum(lo*lo, hi*hi))
    return np.maximum(0., low), high


def sech_range(lo, hi):
    nearest = np.where((lo <= 0) & (hi >= 0), 0., np.minimum(abs(lo), abs(hi)))
    farthest = np.maximum(abs(lo), abs(hi))
    # A small extra guard for transcendental evaluation, separate from the
    # real-arithmetic interval proof; see the numerical-precision discussion.
    low, high = 1/np.cosh(farthest)**2, 1/np.cosh(nearest)**2
    guard = 16*np.finfo(float).eps
    return np.maximum(0., low-guard), np.minimum(1., high+guard)


class Interval:
    """Value and first-derivative intervals for the single unknown kappa."""
    __array_priority__ = 1000

    def __init__(self, lo, hi=None, dl=0., dh=None):
        self.lo = np.asarray(lo)
        self.hi = np.asarray(lo if hi is None else hi)
        self.dl = np.asarray(dl)
        self.dh = np.asarray(dl if dh is None else dh)

    @staticmethod
    def as_interval(value):
        return value if isinstance(value, Interval) else Interval(value)

    def __add__(self, other):
        other = self.as_interval(other)
        lo, hi = outward(self.lo+other.lo, self.hi+other.hi)
        dl, dh = outward(self.dl+other.dl, self.dh+other.dh)
        return Interval(lo, hi, dl, dh)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo, -self.dh, -self.dl)

    def __sub__(self, other):
        return self + (-self.as_interval(other))

    def __rsub__(self, other):
        return self.as_interval(other) + (-self)

    def __mul__(self, other):
        other = self.as_interval(other)
        lo, hi = product(self.lo, self.hi, other.lo, other.hi)
        al, ah = product(self.dl, self.dh, other.lo, other.hi)
        bl, bh = product(self.lo, self.hi, other.dl, other.dh)
        dl, dh = outward(al+bl, ah+bh)
        return Interval(lo, hi, dl, dh)

    __rmul__ = __mul__

    def inverse(self):
        if np.any(self.lo <= 0):
            raise ValueError("The interval denominator is not bounded away from zero.")
        lo, hi = outward(1/self.hi, 1/self.lo)
        sl, sh = squared(lo, hi)
        dl, dh = product(-self.dh, -self.dl, sl, sh)
        return Interval(lo, hi, dl, dh)

    def __truediv__(self, other):
        other = self.as_interval(other)
        return self*other.inverse()

    def square(self):
        lo, hi = squared(self.lo, self.hi)
        dl, dh = product(2*self.lo, 2*self.hi, self.dl, self.dh)
        return Interval(lo, hi, dl, dh)

    def tanh(self):
        guard = 16*np.finfo(float).eps
        lo = np.maximum(-1., np.tanh(self.lo)-guard)
        hi = np.minimum(1., np.tanh(self.hi)+guard)
        fl, fh = sech_range(self.lo, self.hi)
        dl, dh = product(fl, fh, self.dl, self.dh)
        return Interval(lo, hi, dl, dh)

    def sech2(self):
        lo, hi = sech_range(self.lo, self.hi)
        tl, th = self.tanh().lo, self.tanh().hi
        fl, fh = product(-2*th, -2*tl, lo, hi)
        dl, dh = product(fl, fh, self.dl, self.dh)
        return Interval(lo, hi, dl, dh)


def tanh(x):
    return x.tanh() if isinstance(x, Interval) else np.tanh(x)


def square(x):
    return x.square() if isinstance(x, Interval) else x*x


def sech2(x):
    return x.sech2() if isinstance(x, Interval) else 1/np.cosh(x)**2


def paths(noise, kappa, *, b=.5, c=.5, sigma=1.0):
    """noise has shape (observations, substeps, paths); initial state is zero."""
    horizon, substeps, count = noise.shape
    h = 1.0 / substeps
    state = np.zeros((count, 2))
    jac = np.zeros_like(state)
    history, sensitivity = [state.copy()], [jac.copy()]
    for shocks in noise:
        for z in shocks:
            x, y = state.T
            jx, jy = jac.T
            tx, ty = np.tanh(x), np.tanh(y)
            state = np.column_stack(((1-h)*x+h*c*ty+sigma*np.sqrt(h)*z,
                                     (1-b*h)*y+h*kappa*tx))
            jac = np.column_stack(((1-h)*jx+h*c*(1-ty*ty)*jy,
                                   (1-b*h)*jy+h*tx+h*kappa*(1-tx*tx)*jx))
        history.append(state.copy())
        sensitivity.append(jac.copy())
    return np.stack(history, axis=1), np.stack(sensitivity, axis=1)


def predicted(start, start_jac, fitted, *, substeps, b=.5, c=.5, sigma=1.0):
    """Fitted zero-noise prediction, tangent covariance, and their derivatives.

    The fit stays fixed. Derivatives act only through the starting state.
    """
    h, count = 1.0/substeps, len(start)
    mean, direction = start.copy(), start_jac.copy()
    cov, dcov = np.zeros((count, 2, 2)), np.zeros((count, 2, 2))
    for _ in range(substeps):
        tx, ty = np.tanh(mean).T
        dx, dy = 1-tx*tx, 1-ty*ty
        f, df = np.zeros_like(cov), np.zeros_like(cov)
        f[:, 0, 0], f[:, 1, 1] = 1-h, 1-b*h
        f[:, 0, 1], f[:, 1, 0] = h*c*dy, h*fitted*dx
        df[:, 0, 1] = -2*h*c*ty*dy*direction[:, 1]
        df[:, 1, 0] = -2*h*fitted*tx*dx*direction[:, 0]
        ft, dft = f.swapaxes(1, 2), df.swapaxes(1, 2)
        dcov = df@cov@ft + f@dcov@ft + f@cov@dft
        cov = f@cov@ft
        cov[:, 0, 0] += h*sigma*sigma
        direction = np.einsum("nij,nj->ni", f, direction)
        mean = np.column_stack(((1-h)*mean[:, 0]+h*c*ty,
                                (1-b*h)*mean[:, 1]+h*fitted*tx))
    return mean, direction, cov, dcov


def cp_bounds(successes: int, n: int, tail: float) -> tuple[float, float]:
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, n-successes+1))
    high = 1.0 if successes == n else float(beta.ppf(1-tail, successes+1, n-successes))
    return low, high


def paired_difference(indicator, baseline, tail):
    count = len(baseline)
    n10, n01 = int(np.sum(indicator & ~baseline)), int(np.sum(~indicator & baseline))
    l10, u10 = cp_bounds(n10, count, tail)
    l01, u01 = cp_bounds(n01, count, tail)
    return l10-u01, u10-l01, n10, n01


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
