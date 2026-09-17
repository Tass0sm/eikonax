"""Grow the initial splats outward from the source, ridge to ridge (1-D).

In 1-D the arrival time away from the source satisfies `T' = sigma n(x)`
(`sigma = +-1` the direction of travel, `n` the slowness in normalized
coordinates), so the local model of a splat at `B` travelling in direction
`sigma` is exactly

    c + n(B) d + 1/2 sigma n'(B) d^2 + O(n'' d^3).

Each side of the source is walked independently. The next centre is one
step `r` further on, with `r` the largest step (from `r_max`, shrinking by
`step_shrink`) whose third-order error `max |n''| r^3 / 6` over the step is
below `tol`. The new splat's emission time continues the previous one's,
using both splats' local models (slope and its derivative at each end --
a cubic Hermite rule, `O(r^5)` per step):

    c_{j+1} = c_j + r (n_j + n_{j+1}) / 2 + sigma r^2 (n'_j - n'_{j+1}) / 12

Inheriting `c_{j+1} = L_j(B_{j+1})` from the previous model alone drifts
by `O(r^3)` per step, and that drift -- not the blending -- dominated the
error. The walk stops once a centre lies past the domain boundary, so the
windows cover the whole box.

Checking `|n''|` over the whole step (not just at `B`) is what stops a step
starting at an inflection point of `n` from jumping over a slow region --
a per-splat cost, not a per-query one, and the 1-D stand-in for bounding
the speed over a splat's region in higher dimensions.

Window radii are one-sided (see `field.WavefrontField`): `overlap` times
the spacing to the previous centre behind, and to the next centre ahead.
With `1/2 < overlap < 1`, consecutive windows always overlap by the same
fraction and no window reaches past a neighbour's centre -- in particular
not back over the source, where a relay's one-sided model is wrong. A
single symmetric radius cannot do both where the spacing changes: sized to
the longer gap, a splat whose next step is long (the speed has flattened
out ahead) reached back over its short-step neighbours into curvature its
quadratic model does not capture; sized to the shorter gap, the windows
barely overlapped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ...domains import DTYPE


def _derivatives(field):
    """Jitted `x -> (n, n', n'')` of the 1-D normalized slowness."""
    one = jnp.ones(1, DTYPE)

    def n(t):
        return field.covector(jnp.reshape(t, (1,)), one)[0]

    dn = jax.grad(n)
    ddn = jax.grad(dn)
    return jax.jit(jax.vmap(lambda t: jnp.stack([n(t), dn(t), ddn(t)])))


def _step(derivs, B, sigma, cfg):
    """Largest step from `B` in direction `sigma` meeting the error budget."""
    ts = np.linspace(0.0, 1.0, cfg.step_samples)
    r = cfg.r_max
    while r > cfg.r_min:
        pts = jnp.asarray(B + sigma * r * ts, dtype=DTYPE)
        ddn = float(np.max(np.abs(np.asarray(derivs(pts))[:, 2])))
        if ddn * r ** 3 / 6.0 <= cfg.tol:
            return r
        r *= cfg.step_shrink
    return cfg.r_min


def chain_1d(field):
    """Initial `params` for `field` (a 1-D `WavefrontField`)."""
    cfg = field.cfg
    if field.dim != 1:
        raise NotImplementedError("wavefront's chain initializer is 1-D only for now")
    derivs = _derivatives(field)
    s = float(field.source[0])

    Bs, cs, us, Hs, Rs = [], [], [], [], []
    first_steps = {}
    for sigma in (1.0, -1.0):
        B, c = s, 0.0
        centres = [s]
        n, dn, _ = np.asarray(derivs(jnp.asarray([B], DTYPE)))[0]
        while sigma * B < 0.5:
            r = _step(derivs, B, sigma, cfg)
            B = B + sigma * r
            n_next, dn_next, _ = np.asarray(derivs(jnp.asarray([B], DTYPE)))[0]
            # Hermite: both ends' local models, O(r^5) per step
            c = c + 0.5 * r * (n + n_next) + sigma * r ** 2 * (dn - dn_next) / 12.0
            n, dn = n_next, dn_next
            centres.append(B)
            Bs.append(B), cs.append(c), us.append(sigma), Hs.append(sigma * dn)
        gaps = np.abs(np.diff(centres))
        first_steps[sigma] = gaps[0] if len(gaps) else cfg.r_max
        # (behind, ahead) radii for this side's relays (centres[1:])
        for j in range(1, len(centres)):
            behind = gaps[j - 1]
            ahead = gaps[j] if j < len(gaps) else behind
            Rs.append((cfg.overlap * behind, cfg.overlap * ahead))

    k = len(Bs)
    params = {
        "B": jnp.asarray(np.reshape(Bs, (k, 1)), DTYPE),
        "log_R": jnp.asarray(np.log(np.reshape(Rs, (k, 2))), DTYPE),
        "c": jnp.asarray(cs, DTYPE).reshape(k),
        "u": jnp.asarray(np.reshape(us, (k, 1)), DTYPE),
        "H": jnp.asarray(np.reshape(Hs, (k, 1, 1)), DTYPE),
        "src_log_R": jnp.asarray(np.log(cfg.overlap * np.array([first_steps[-1.0], first_steps[1.0]])), DTYPE),
        "src_g": jax.grad(field.slowness_at)(field.source).astype(DTYPE),
    }
    return params
