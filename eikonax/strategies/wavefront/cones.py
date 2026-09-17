"""The N-dimensional wavefront field: a partition of unity over CONE splats.

Same shape as the 1-D model (`field.py`) -- a normalized blend of windows
whose weights are local travel-time models -- but the local model is the
wavefront itself rather than its quadratic expansion:

    L_j(x) = c_j + n_j (|x - S_j| - rho_j) + correction(x - B_j)

    S_j = B_j - rho_j p_j        the splat's VIRTUAL SOURCE
    rho_j = |B_j - S_j|          how far the front has travelled from it
    n_j = 1/speed(B_j)           slowness at the centre
    correction(d) = (g_j.d)(e.d) - 1/2 (e.g_j)(e.d)^2,  g_j = grad n(B_j),
                                 e = (x - S_j)/|x - S_j| the local ray

With uniform speed the correction vanishes and `L_j` is EXACT wherever the
front from `S_j` reaches, at any distance -- a quadratic expansion would
only be good to `|d| << rho_j`. That is what lets windows be large in open
space: `grow.py` sizes them by obstacle clearance, not by curvature.

The correction is the same second-order speed-gradient term the 1-D model
carries, written radially (`e` in place of a fixed direction) so that it
also covers a point source, where the 1-D form `1/2 (g.d)|d|` is recovered.
It reproduces the exact Hessian of a travel-time field, `H p = n grad n`,
at the centre.

`rho_j = 0` is a point source (the field's own source splat); large
`rho_j` is a plane wave, and the code keeps `|x - S_j| - rho_j` in the
stable form `(|d|^2 + 2 rho_j p_j.d) / (|x - S_j| + rho_j)`, which stays
exact as `rho_j -> inf`.

Everything here is in PHYSICAL coordinates and float64 (the 1-D model works
in normalized ones; only that one feeds a neural-style trainer). The
per-splat arrays are `B (k, d)`, `c (k,)`, `p (k, d)` unit, `rho (k,)`,
`n (k,)`, `g (k, d)`, `R (k,)`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

#: Below this the ray direction at the apex is arbitrary, not undefined.
APEX_EPS = 1e-9


def local_values(xp, x, params):
    """`L_j(x)` for every splat at ONE point `x (d,)`, `(k,)`.

    Written against `xp` (`numpy` or `jax.numpy`) so the growth loop and the
    trained field evaluate exactly the same model.
    """
    d = x[None, :] - params["B"]                               # (k, d)
    dd = xp.sum(d * d, axis=-1)
    pd = xp.sum(params["p"] * d, axis=-1)
    rho = params["rho"]
    # |x - S|, S = B - rho p  ->  |d + rho p|^2 = |d|^2 + 2 rho p.d + rho^2
    dist = xp.sqrt(xp.maximum(dd + 2.0 * rho * pd + rho ** 2, 0.0) + APEX_EPS ** 2)
    along = (dd + 2.0 * rho * pd) / (dist + rho)               # |x - S| - rho, stably
    e = (d + rho[:, None] * params["p"]) / dist[:, None]       # local ray direction
    ed = xp.sum(e * d, axis=-1)
    gd = xp.sum(params["g"] * d, axis=-1)
    eg = xp.sum(e * params["g"], axis=-1)
    return params["c"] + params["n"] * along + gd * ed - 0.5 * eg * ed ** 2


def values_of(X, params, j):
    """`L_j` at points `X (m, dim)` for ONE splat `j`, `(m,)` (numpy)."""
    d = np.asarray(X) - params["B"][j]
    rho, p, n, g, c = params["rho"][j], params["p"][j], params["n"][j], params["g"][j], params["c"][j]
    dd = np.einsum("mi,mi->m", d, d)
    pd = d @ p
    dist = np.sqrt(np.maximum(dd + 2.0 * rho * pd + rho ** 2, 0.0) + APEX_EPS ** 2)
    along = (dd + 2.0 * rho * pd) / (dist + rho)
    e = (d + rho * p) / dist[:, None]
    ed = np.einsum("mi,mi->m", e, d)
    return c + n * along + (d @ g) * ed - 0.5 * (e @ g) * ed ** 2


def local_gradient(x, params, j):
    """`grad L_j(x)` for one splat `j`, `(d,)` (numpy; the growth loop only)
    -- the direction the front travels, used when a neighbour inherits
    `j`'s wave."""
    d = np.asarray(x) - params["B"][j]
    rho, p, n, g = params["rho"][j], params["p"][j], params["n"][j], params["g"][j]
    dist = np.sqrt(max(d @ d + 2.0 * rho * (p @ d) + rho ** 2, 0.0) + APEX_EPS ** 2)
    e = (d + rho * p) / dist
    ed = e @ d
    return n * e + g * ed + (g @ d) * e - (e @ g) * ed * e


def window(xp, z2):
    """`(1 - z2)^3` inside the unit ball, `0` outside."""
    inside = z2 < 1.0
    return xp.where(inside, (1.0 - xp.where(inside, z2, 0.0)) ** 3, 0.0)


class ConeField:
    """The N-D model bound to a domain and a source (physical coordinates)."""

    def __init__(self, domain, source, cfg):
        self.domain = domain
        self.cfg = cfg
        self.dim = domain.dim
        self.source = np.asarray(source, dtype=np.float64).reshape(self.dim)

    # -- the blend -------------------------------------------------------------

    def _blend(self, params, x):
        L = local_values(jnp, x, params)
        w = window(jnp, jnp.sum((x[None, :] - params["B"]) ** 2, axis=-1) / params["R"] ** 2)
        temp = self.cfg.value_temperature
        if temp is None:
            pi = w / (jnp.sum(w) + 1e-30)
        elif temp <= 0:
            # hard min over the windows that reach x: every L_j is the length
            # of a real path, so the smallest is the best estimate, and an
            # average of two disagreeing fronts is worse than either
            covered = w > 0
            pi = jnp.zeros_like(w).at[jnp.argmin(jnp.where(covered, L, jnp.inf))].set(1.0)
            pi = jnp.where(jnp.any(covered), pi, 0.0)
        else:
            pos = w > 0
            logits = jnp.where(pos, jnp.log(jnp.where(pos, w, 1.0)) - L / temp, -jnp.inf)
            pi = jnp.where(jnp.any(pos), jax.nn.softmax(logits), 0.0)
        return jnp.sum(pi * L), jnp.sum(w)

    def value(self, params, x):
        """`T` at one physical point (scalar). Uncovered points give 0 with
        `coverage = 0`; `Model.time` reports them as `nan`."""
        return self._blend(params, x)[0]

    def coverage(self, params, X):
        """`sum_j w_j` at physical points `(n, dim)`, `(n,)` -- 0 where no
        window reaches, which is where the blend means nothing."""
        return jax.vmap(lambda x: self._blend(params, x)[1], in_axes=0)(X)

    def evaluate(self, params, X):
        """`T` at physical points `(n, dim)`, `(n,)`."""
        return jax.vmap(self.value, in_axes=(None, 0))(params, X)

    def grad(self, params, X):
        """`dT/dx` at physical points, `(n, dim)`."""
        return jax.vmap(jax.grad(self.value, argnums=1), in_axes=(None, 0))(params, X)

    # -- what `Model` needs ----------------------------------------------------

    def to_internal(self, X):
        """`Model` hands physical points; this model already uses them."""
        return jnp.asarray(X, dtype=jnp.float64)

    def gradient_scale(self):
        return 1.0

    def num_splats(self, params) -> int:
        return int(params["B"].shape[0])
