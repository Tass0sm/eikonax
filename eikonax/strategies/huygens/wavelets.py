"""The Huygens field: a min over ray wavelets, built on `srms`'s generic `SplatModel`.

Huygens' principle for the PHASE of a wave is min-plus: the front is the
envelope of wavelets emitted from earlier fronts, `T(x) = min_y [T(y) +
d(y, x)]`. So the field is an SRM whose combine is a (soft-)min and whose
splat weights act additively, as emission times:

    T(x) = min_j [ c_j + ray_j(x) ]

  - `ray_j(x)` (`ray_mother`) is the travel time of the straight segment
    `B_j -> x`, sampled through the domain's speed and metric, plus
    `occlusion_cost` if the segment crosses an obstacle. With uniform speed
    it is exactly a cone, so the source wavelet ALONE is the solution.
  - `c_j` is not a free parameter: it is the field without `j`, at `B_j`
    (`values`). That is what makes `B_j` a Huygens source rather than an
    arbitrary bump, and what rules out spurious local minima at centres.

Every term `c_j + ray_j(x)` is the length of a real path (source -> ... ->
B_j -> x, straight between wavelets), so `T` is an UPPER bound on the true
travel time, up to ray sampling (and soft-min bias if `eps > 0`). An
obstacle can only RAISE the true `T` while a min can only lower it, so a
wavelet does not bend the wave by being added: the occlusion term removes
each wavelet from where it cannot see, and wavelets placed at obstacle
features (corners, gaps) re-emit into the shadow -- Keller's diffraction
sources. For polygonal obstacles in uniform speed this is exact with one
wavelet per relevant corner.

`A_j` is kept for the `SplatModel` interface as a linear correction applied
to the displacement before it is measured. It is the identity and is NOT
trained: the domain's metric already gives each wavelet its shape, and
minimizing `mean T` over a free `A` would simply shrink every length and
break the upper bound. Learning it needs an objective that pins slopes
(an eikonal residual) -- the hook for wavelets a straight ray cannot
describe. That is the open case: with a position-dependent metric (e.g.
`se2_domain`'s heading-dependent one) geodesics curve, a straight ray is
only a feasible path, and piecewise-straight wavelets approximate the field
poorly (measured on a free 21x21x8 SE(2) grid: 58 wavelets, RMS 0.20 vs
0.12 for `fsm` on the same grid, both against a 2x finer `fsm`).

Coordinates are the domain's NORMALIZED ones (`eikonax.domains`); costs are
physical travel times, because the normalized metric carries the spans.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from srms.lib.composable_splat import SplatModel

from ...domains import DTYPE

#: Bellman-Ford's "+inf": above any reachable value, finite so nothing NaNs.
UNREACHED = 1e6


def periodic_displacement(periodic) -> Callable:
    """`Log_B(x)` on the normalized box: wrapped on periodic axes (period 1)."""
    mask = jnp.asarray(periodic)

    def displacement(B, x):
        d = x - B
        return jnp.where(mask, jnp.mod(d + 0.5, 1.0) - 0.5, d)

    return displacement


def ray_mother(domain, cfg) -> Callable:
    """Straight-ray travel time from `B` over displacement `v`; `aux = B`.

    Midpoint rule over `cfg.ray_samples` points: `mean_t |A v|_{G(p_t)} /
    max(speed(p_t), min_speed)`, the length smoothed at `v = 0` by
    `cone_delta`. Adds `cfg.occlusion_cost` if any sample has `speed <=
    obstacle_speed`. The sample spacing must be finer than the thinnest
    obstacle, or a ray can step over it.
    """
    ts = (jnp.arange(cfg.ray_samples, dtype=DTYPE) + 0.5) / cfg.ray_samples
    delta = cfg.cone_delta

    def mother(v, A, B):
        pts = domain.wrap(B[None, :] + ts[:, None] * v[None, :])
        speed = domain.speed(pts)
        G = jnp.linalg.inv(domain.metric_inv(pts))
        u = A @ v
        length = jnp.sqrt(jnp.einsum("i,pij,j->p", u, G, u) + delta ** 2) - delta
        cost = jnp.mean(length / jnp.clip(speed, cfg.min_speed, None))
        blocked = jnp.any(speed <= cfg.obstacle_speed)
        return cost + jnp.where(blocked, cfg.occlusion_cost, 0.0)

    return mother


def min_combine(eps: float) -> Callable:
    """`min_j (V_j + atom_j)` for `eps == 0`, else the log-semiring soft-min
    `-eps * logsumexp(-(V_j + atom_j) / eps)`. `(1,)` out."""

    def combine(V, atoms):
        costs = V[:, 0] + atoms
        if eps == 0.0:
            return jnp.min(costs)[None]
        return -eps * jax.scipy.special.logsumexp(-costs / eps)[None]

    return combine


class HuygensField:
    """The model, bound to a domain and a source (normalized coordinates).

    Parameters `(V, A, B)` are the LEARNABLE wavelets, `V: (k, 1)`,
    `A: (k, dim, dim)`, `B: (k, dim)`; `k = 0` is valid and is the
    source-only field. The source wavelet (`c = 0`) is prepended on every
    evaluation, so `T(source) = 0` exactly. The `V` held in `params` is a
    cache of `values(params)`: `evaluate` uses it as-is, `time` recomputes it.
    """

    def __init__(self, domain, source_n, cfg):
        self.domain = domain
        self.cfg = cfg
        self.dim = domain.dim
        self.source = jnp.asarray(source_n, dtype=DTYPE).reshape(1, self.dim)
        self.model = SplatModel(
            mother=ray_mother(domain, cfg),
            combine=min_combine(cfg.eps),
            displacement=periodic_displacement(domain.periodic),
        )

    def empty(self):
        d = self.dim
        return jnp.zeros((0, 1), DTYPE), jnp.zeros((0, d, d), DTYPE), jnp.zeros((0, d), DTYPE)

    def wavelets_at(self, B, V=None):
        """Fresh wavelets at centres `B` (identity `A`); `V` zeros unless given."""
        B = jnp.asarray(B, DTYPE)
        n = B.shape[0]
        V = jnp.zeros((n, 1), DTYPE) if V is None else jnp.asarray(V, DTYPE).reshape(n, 1)
        A = jnp.broadcast_to(jnp.eye(self.dim, dtype=DTYPE), (n, self.dim, self.dim))
        return V, A, B

    @staticmethod
    def append(params, new):
        return tuple(jnp.concatenate([p, q]) for p, q in zip(params, new))

    def _full(self, params):
        return self.append(self.wavelets_at(self.source), params)

    def rays(self, params, Xn):
        """`ray_j(x)` (no emission time) for every wavelet, source first, `(n, k+1)`."""
        full = self._full(params)
        return self.model.atom_matrix(full, Xn, full[2])

    def evaluate(self, params, Xn):
        """`T` at normalized points using the stored `V`, `(n,)`."""
        full = self._full(params)
        return self.model(full, Xn, full[2])[:, 0]

    def time(self, params, Xn):
        """`T` at normalized points with `V` recomputed, `(n,)` --
        differentiable in `A` and `B`, including through the emission times."""
        return self.evaluate(self.with_values(params), Xn)

    def grad(self, params, Xn):
        """`dT/dXn`, `(n, dim)`, with the stored `V`."""
        full = self._full(params)
        return self.model.grad(full, Xn, full[2])

    def with_values(self, params):
        return self.values(params), params[1], params[2]

    def values(self, params):
        """`c_j = min_{m != j}(c_m + ray_m(B_j))`, `(k, 1)`: exact and differentiable.

        Hard Bellman-Ford from +inf, with the hop costs held fixed, finds the
        shortest-path tree; then `c_j = c_pred(j) + ray_pred(j)(B_j)` is
        unrolled `k` times along that tree (its depth is at most `k`). The
        result equals the Bellman-Ford value but carries gradients through
        every hop back to the source, so moving a wavelet sees its effect on
        every wavelet downstream of it.
        """
        k = params[2].shape[0]
        idx = jnp.arange(k)
        hop = self.rays(params, params[2]).at[idx, idx + 1].set(UNREACHED)
        hop_sg = jax.lax.stop_gradient(hop)

        def with_source(c):
            return jnp.concatenate([jnp.zeros(1, DTYPE), c])

        def cond(state):
            _, changed, it = state
            return changed & (it <= k)

        def body(state):
            c, _, it = state
            new = jnp.minimum(c, jnp.min(with_source(c)[None, :] + hop_sg, axis=1))
            return new, jnp.any(new < c - 1e-6), it + 1

        init = (jnp.full(k, UNREACHED, DTYPE), jnp.asarray(k > 0), 0)
        c_bf, _, _ = jax.lax.while_loop(cond, body, init)
        pred = jnp.argmin(with_source(c_bf)[None, :] + hop_sg, axis=1)  # 0 = the source
        edge = hop[idx, pred]

        def unroll(c, _):
            return edge + with_source(c)[pred], None

        c, _ = jax.lax.scan(unroll, jax.lax.stop_gradient(c_bf), None, length=k)
        return c[:, None]
