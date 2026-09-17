"""1-D reference and comparison fields for the wavefront strategy.

  - `exact_1d` -- the true arrival time, `|int_s^x n|`, by fine trapezoid
    quadrature (float64).
  - `pu0_fit` -- a normalized SRM with SCALAR weights (the degree-0 case of
    the wavefront model) on the same windows, weights least-squares fitted
    to the exact field.
  - `ridge_fit` -- the "ridge-shaped mother" idea taken literally: an
    UNnormalized sum `sum_j V_j |d| w_j(d)` on the same centres and radii,
    weights least-squares fitted to the exact field.

Both baselines are ORACLE fits (they see the exact field; the wavefront
model never does), so they are generous upper bounds on what those
structures can do with this many splats.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ...domains import DTYPE
from .field import window


def exact_1d(domain, source_n: float, Xn, n_fine: int = 200001):
    """True 1-D arrival time at normalized points `Xn (n, 1)`, `(n,)`."""
    grid = np.linspace(-0.5, 0.5, n_fine)
    pts = jnp.asarray(grid[:, None], DTYPE)
    speed = np.asarray(domain.speed(pts), dtype=np.float64)
    ginv = np.asarray(domain.metric_inv(pts), dtype=np.float64)[:, 0, 0]
    n = 1.0 / (speed * np.sqrt(ginv))
    F = np.concatenate([[0.0], np.cumsum(0.5 * (n[1:] + n[:-1]) * np.diff(grid))])
    x = np.asarray(Xn, dtype=np.float64)[:, 0]
    return np.abs(np.interp(x, grid, F) - np.interp(float(source_n), grid, F))


def _windows(field, params, Xn):
    """Signed offsets and window values of every splat (source first), `(n, k+1)`."""
    B = np.concatenate([np.asarray(field.source)[None, 0], np.asarray(params["B"])[:, 0]])
    sigma = np.concatenate([[1.0], np.sign(np.asarray(params["u"])[:, 0])])
    R = np.exp(np.concatenate([np.asarray(params["src_log_R"])[None], np.asarray(params["log_R"])]))
    d = np.asarray(Xn)[:, 0:1] - B[None, :]
    R_side = np.where(sigma[None, :] * d > 0, R[None, :, 1], R[None, :, 0])
    return d, np.asarray(window(jnp.asarray((d / R_side) ** 2)), dtype=np.float64)


def pu0_fit(field, params, Xn, T_true):
    """Degree-0 normalized SRM, oracle-fitted. Returns predictions at `Xn`."""
    _, w = _windows(field, params, Xn)
    pi = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    V, *_ = np.linalg.lstsq(pi, T_true, rcond=None)
    return pi @ V


def pu0_uniform_fit(Xn, T_true, k: int, overlap: float = 0.75):
    """Degree-0 normalized SRM with `k` EVENLY spaced windows over the box,
    oracle-fitted -- how many scalar splats the same accuracy costs."""
    x = np.asarray(Xn, dtype=np.float64)[:, 0]
    B = np.linspace(-0.5, 0.5, k)
    R = 2.0 * overlap * (B[1] - B[0])
    w = np.asarray(window(jnp.asarray(((x[:, None] - B[None, :]) / R) ** 2)), dtype=np.float64)
    pi = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    V, *_ = np.linalg.lstsq(pi, T_true, rcond=None)
    return pi @ V


def ridge_fit(field, params, Xn, T_true):
    """Unnormalized sum of ridge-shaped atoms `|d| w(d)`, oracle-fitted."""
    d, w = _windows(field, params, Xn)
    atoms = np.abs(d) * w
    V, *_ = np.linalg.lstsq(atoms, T_true, rcond=None)
    return atoms @ V


__all__ = ["exact_1d", "pu0_fit", "pu0_uniform_fit", "ridge_fit"]
