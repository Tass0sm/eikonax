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


def blocked(P, Q, rects, tol: float = 1e-9):
    """Does each segment `P[i] -> Q[i]` cross a `(y0, y1, x0, x1)` rectangle's
    interior? `(m, 2)`, `(m, 2)` in, `(m,)` bool out (Liang-Barsky)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    d = Q - P
    out = np.zeros(len(P), dtype=bool)
    for y0, y1, x0, x1 in rects:
        t0, t1 = np.zeros(len(P)), np.ones(len(P))
        inside = np.ones(len(P), dtype=bool)
        for pk, qk in ((-d[:, 0], P[:, 0] - y0 - tol), (d[:, 0], y1 - tol - P[:, 0]),
                       (-d[:, 1], P[:, 1] - x0 - tol), (d[:, 1], x1 - tol - P[:, 1])):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(pk != 0, qk / np.where(pk != 0, pk, 1.0), 0.0)
            inside &= (pk != 0) | (qk >= 0)
            t0 = np.where(pk < 0, np.maximum(t0, t), t0)
            t1 = np.where(pk > 0, np.minimum(t1, t), t1)
        out |= inside & (t0 < t1)
    return out


def exact_rects(rects, source, pts):
    """Exact unit-speed arrival time around `(y0, y1, x0, x1)` rectangles:
    Dijkstra over the source and the rectangles' corners, then a straight
    shot to each point. `inf` where a point is unreachable (inside a wall).

    The reference for the obstacle scenes -- `fsm`'s own wide-stencil error
    (~1.4e-2 RMS on these grids) is larger than the field's.
    """
    source = np.asarray(source, float)
    nodes = np.array([source] + [(y, x) for y0, y1, x0, x1 in rects
                                 for y in (y0, y1) for x in (x0, x1)])
    n = len(nodes)
    seg = np.repeat(nodes, n, axis=0), np.tile(nodes, (n, 1))
    free = (~blocked(*seg, rects)).reshape(n, n)
    step = np.where(free, np.linalg.norm(nodes[:, None] - nodes[None], axis=-1), np.inf)

    dist = np.full(n, np.inf)
    dist[0] = 0.0
    todo = np.ones(n, dtype=bool)
    for _ in range(n):
        i = int(np.argmin(np.where(todo, dist, np.inf)))
        if not todo[i] or not np.isfinite(dist[i]):
            break
        todo[i] = False
        dist = np.minimum(dist, dist[i] + step[i])

    pts = np.asarray(pts, float)
    m = len(pts)
    visible = ~blocked(np.repeat(nodes, m, axis=0), np.tile(pts, (n, 1)), rects).reshape(n, m)
    reach = np.where(visible, dist[:, None] + np.linalg.norm(pts[None] - nodes[:, None], axis=-1), np.inf)
    return reach.min(axis=0)


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


__all__ = ["blocked", "exact_1d", "exact_rects", "pu0_fit", "pu0_uniform_fit", "ridge_fit"]
