"""Continuous domains the neural solvers in `eikonax.ntfields` train against
-- the counterpart of the `(grid_shape, resolutions, periodic, metric,
speed_fn)` bundle `fsm.Solver` is built from, minus the grid: a neural
travel-time field is trained on random collocation points, so all it needs
is the same geometry as a *continuous* object.

A `Domain` supplies, over an axis-aligned box with per-axis periodicity:

  - `speed(X)` -- the same `speed_fn` idea as `fsm` (obstacles are just
    `speed ~ 0`), and
  - `metric_inv(X)` -- the INVERSE Riemannian metric `G(x)^-1`. `fsm` wants
    `G` because it measures the length of a finite displacement; the
    eikonal PDE `speed(x) * sqrt(grad T^T G(x)^-1 grad T) = 1` measures the
    length of a *covector*, so the neural side wants the inverse.

**Everything the network sees is normalized** to `[-0.5, 0.5]^n` per axis
(what the `ntrl-demo` reference implementation this package's backend is
ported from assumes throughout: Fourier-feature bandwidth, the TD step
length, the output scale are all tuned for that box). A periodic axis
therefore has period exactly 1 in normalized coordinates, which is what
lets `backends.metric_net` be exactly periodic by construction. Travel
times still come out in physical units, because `speed`/`metric_inv` are
rescaled with the coordinates: `Ghat^-1 = diag(1/L) G^-1 diag(1/L)` for
per-axis span `L`, which leaves `grad T^T G^-1 grad T` invariant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np

#: The neural side runs in single precision even though `fsm` turns on
#: jax x64 process-wide at import (arrival times there accumulate over
#: hundreds of sweeps; here they don't).
DTYPE = jnp.float32


def dual_norm(covectors, metric_inv):
    """`|p|_x = sqrt(p^T G(x)^-1 p)`: the norm a covector such as `grad T`
    is measured in. `(n, dim)` and `(n, dim, dim)` in, `(n,)` out."""
    sq = jnp.einsum("ni,nij,nj->n", covectors, metric_inv, covectors)
    return jnp.sqrt(jnp.clip(sq, 1e-12, None))


@runtime_checkable
class Domain(Protocol):
    """What `eikonax.ntfields`'s strategies and backends need from a
    problem. `X`/`Xn` arguments are always `(n_points, dim)` arrays of
    NORMALIZED coordinates unless the name says otherwise."""

    dim: int
    periodic: tuple[bool, ...]

    def to_normalized(self, X: np.ndarray) -> np.ndarray: ...
    def from_normalized(self, Xn: jnp.ndarray) -> jnp.ndarray: ...
    def wrap(self, Xn: jnp.ndarray) -> jnp.ndarray: ...
    def sample(self, rng: np.random.Generator, n: int) -> jnp.ndarray: ...
    def speed(self, Xn: jnp.ndarray) -> jnp.ndarray: ...
    def metric_inv(self, Xn: jnp.ndarray) -> jnp.ndarray: ...
    def grid(self, grid_shape: tuple[int, ...]) -> jnp.ndarray: ...


class BoxDomain:
    """A box `[lower, upper]` with per-axis periodicity, an arbitrary
    JAX-jittable speed field, and an arbitrary per-point inverse metric.

    Args:
        lower, upper: length-`dim` physical bounds. For a periodic axis
            these are the two ends of ONE period (e.g. `0` and `2*pi` for
            an S^1 heading axis).
        periodic: length-`dim` bools, whether each axis wraps.
        speed_fn: `coords -> speed`, physical coordinates `(..., dim)` in,
            `(...)` out -- the same convention `fsm.Solver` uses.
        metric_inv_fn: `coords -> G^-1`, `(..., dim)` in, `(..., dim, dim)`
            out. Defaults to the identity (isotropic).
    """

    def __init__(
            self,
            lower,
            upper,
            periodic,
            speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
            metric_inv_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
            grid_shape: tuple[int, ...] | None = None,
    ):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.dim = len(self.lower)
        if self.upper.shape != (self.dim,):
            raise ValueError(f"upper must have length {self.dim}, got shape {self.upper.shape}")
        if np.any(self.upper <= self.lower):
            raise ValueError("upper must be strictly greater than lower on every axis")
        self.periodic = tuple(bool(p) for p in periodic)
        if len(self.periodic) != self.dim:
            raise ValueError(f"periodic must have length {self.dim}, got {len(self.periodic)}")
        self.span = self.upper - self.lower

        self.grid_shape = None if grid_shape is None else tuple(int(s) for s in grid_shape)
        if self.grid_shape is not None and len(self.grid_shape) != self.dim:
            raise ValueError(f"grid_shape must have length {self.dim}, got {self.grid_shape}")

        self._speed_fn = speed_fn
        self._metric_inv_fn = metric_inv_fn
        self._lower_j = jnp.asarray(self.lower, dtype=DTYPE)
        self._span_j = jnp.asarray(self.span, dtype=DTYPE)
        self._periodic_j = jnp.asarray(self.periodic)

    def to_normalized(self, X):
        """Physical -> `[-0.5, 0.5]^dim`."""
        return (jnp.asarray(X, dtype=DTYPE) - self._lower_j) / self._span_j - 0.5

    def from_normalized(self, Xn):
        """`[-0.5, 0.5]^dim` -> physical."""
        return self._lower_j + (Xn + 0.5) * self._span_j

    def wrap(self, Xn):
        """Canonicalize normalized points: periodic axes wrap into
        `[-0.5, 0.5)`, non-periodic axes clamp to the box."""
        wrapped = jnp.mod(Xn + 0.5, 1.0) - 0.5
        clamped = jnp.clip(Xn, -0.5, 0.5)
        return jnp.where(self._periodic_j, wrapped, clamped)

    def sample(self, rng: np.random.Generator, n: int):
        """`n` uniform normalized points."""
        return jnp.asarray(rng.uniform(-0.5, 0.5, size=(n, self.dim)), dtype=DTYPE)

    def speed(self, Xn):
        """Speed at normalized points, `(n,)`."""
        return self._speed_fn(self.from_normalized(Xn)).astype(DTYPE)

    def metric_inv(self, Xn):
        """`Ghat^-1` at normalized points, `(n, dim, dim)` -- the physical
        inverse metric rescaled to normalized coordinates (see module
        docstring)."""
        if self._metric_inv_fn is None:
            base = jnp.broadcast_to(jnp.eye(self.dim, dtype=DTYPE), (Xn.shape[0], self.dim, self.dim))
        else:
            base = self._metric_inv_fn(self.from_normalized(Xn)).astype(DTYPE)
        return base / (self._span_j[None, :, None] * self._span_j[None, None, :])

    def grid(self, grid_shape: tuple[int, ...]):
        """Normalized coordinates of a dense `grid_shape` grid, raveled to
        `(prod(grid_shape), dim)`. Periodic axes are sampled with
        `endpoint=False` (the wrap point is not duplicated), non-periodic
        axes with `endpoint=True` -- matching what `fsm.Solver`'s own
        `(grid_shape, resolutions, origin)` produces for the same box."""
        if len(grid_shape) != self.dim:
            raise ValueError(f"grid_shape must have length {self.dim}, got {grid_shape}")
        axes = [
            np.linspace(-0.5, 0.5, size, endpoint=not self.periodic[i])
            for i, size in enumerate(grid_shape)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        return jnp.asarray(np.stack(mesh, axis=-1).reshape(-1, self.dim), dtype=DTYPE)

    def fsm_solver(self, grid_shape: tuple[int, ...] | None = None, radius: int = 2,
                   obstacle_fill: float | None = None):
        """An `eikonax.fsm.Solver` for the same box -- the grid counterpart
        of this continuous domain, so `eikonax.strategies.fsm` and
        `eikonax.strategies.ntfields` can be handed the SAME domain object.

        `grid_shape` defaults to `self.grid_shape` (set by `se2_domain`).
        Per-axis spacing is `span / n` for a periodic axis and `span /
        (n - 1)` for a bounded one, matching `Domain.grid`'s node layout and
        `se2.build_solver`'s `(resolution, h_theta)`. The per-node metric
        `G` is `inv(metric_inv_fn(node))` (identity when no `metric_inv_fn`
        was given) -- the length-of-a-displacement form `fsm.Solver` wants,
        the inverse of the length-of-a-covector form the PDE side uses.
        """
        from . import fsm

        gs = self.grid_shape if grid_shape is None else tuple(int(s) for s in grid_shape)
        if gs is None:
            raise ValueError("grid_shape not given and the domain has no default grid_shape")
        if len(gs) != self.dim:
            raise ValueError(f"grid_shape must have length {self.dim}, got {gs}")

        resolutions = [
            float(self.span[i] / (gs[i] if self.periodic[i] else gs[i] - 1))
            for i in range(self.dim)
        ]
        nodes = np.asarray(self.from_normalized(self.grid(gs))).reshape(*gs, self.dim)
        if self._metric_inv_fn is None:
            metric = np.broadcast_to(np.eye(self.dim), (*gs, self.dim, self.dim))
        else:
            metric_inv = np.asarray(self._metric_inv_fn(jnp.asarray(nodes, dtype=DTYPE)), dtype=float)
            metric = np.linalg.inv(metric_inv)

        fill = {} if obstacle_fill is None else {"obstacle_fill": obstacle_fill}
        return fsm.Solver(gs, resolutions, self.periodic, metric, self._speed_fn,
                          origin=self.lower, radius=radius, **fill)


def se2_metric_inv_fn(xi_lateral: float = 0.4, xi_turn: float = 0.9) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """`G(theta)^-1` for `se2.default_metric_at_theta`'s "soft preference
    for the current heading" metric, as a jittable function of physical
    `(y, x, theta)` coordinates -- SAME axis order as `se2.build_solver`'s
    `grid_shape = (ny, nx, n_theta)`, see that module for why it's `(dy,
    dx, dtheta)` and not `(dx, dy, dtheta)`.

    `G = A^T D A` with `D = diag(1, 1/xi_lateral^2, 1/xi_turn^2)` and `A`
    the forward/lateral resolution at heading `theta`; `A` is symmetric
    and orthogonal (`A @ A == I`), so `G^-1 = A^T D^-1 A` with no matrix
    inverse to take.
    """
    d_inv = jnp.array([1.0, xi_lateral ** 2, xi_turn ** 2], dtype=DTYPE)

    def metric_inv_fn(coords: jnp.ndarray) -> jnp.ndarray:
        theta = coords[..., 2]
        c, s = jnp.cos(theta), jnp.sin(theta)
        z, o = jnp.zeros_like(c), jnp.ones_like(c)
        a = jnp.stack(
            [jnp.stack([s, c, z], axis=-1),
             jnp.stack([c, -s, z], axis=-1),
             jnp.stack([z, z, o], axis=-1)],
            axis=-2,
        )
        return jnp.einsum("...ki,k,...kj->...ij", a, d_inv, a)

    return metric_inv_fn


def se2_domain(
        speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
        ny: int = 41,
        nx: int = 41,
        resolution: float = 0.1,
        n_theta: int = 16,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        xi_lateral: float = 0.4,
        xi_turn: float = 0.9,
        metric_inv_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> BoxDomain:
    """The `BoxDomain` matching `se2.build_solver(ny, nx, resolution,
    n_theta, ...)` -- `(y, x, theta)` in that order, `theta` periodic over
    `[0, 2*pi)` in `n_theta` bins, `y`/`x` spanning exactly the `(ny, nx)`
    grid `fsm` would lay down at `resolution` from `(origin_x, origin_y)` --
    so a field trained here and a field swept there are directly comparable
    node for node. `grid_shape = (ny, nx, n_theta)` is baked in, so
    `.fsm_solver()` needs no arguments.

    `metric_inv_fn` defaults to `se2_metric_inv_fn(xi_lateral, xi_turn)`;
    pass your own for any other metric (including one with genuine
    translation-rotation coupling).
    """
    lower = (origin_y, origin_x, 0.0)
    upper = (origin_y + (ny - 1) * resolution, origin_x + (nx - 1) * resolution, 2.0 * np.pi)
    if metric_inv_fn is None:
        metric_inv_fn = se2_metric_inv_fn(xi_lateral, xi_turn)
    return BoxDomain(lower, upper, (False, False, True), speed_fn, metric_inv_fn,
                     grid_shape=(ny, nx, n_theta))


#: Named domain constructors the CLI can pick with `--domain`. Each takes a
#: `speed_fn` plus configuration keyword arguments (exposed as flags).
DOMAINS = {"se2": se2_domain}
