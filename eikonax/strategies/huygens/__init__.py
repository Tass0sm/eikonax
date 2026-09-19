"""Huygens-splat strategy: a single-source travel-time field represented as
a min over ray wavelets, grown from the source -- Huygens' principle with the
wavelets as splats of an `srms` splat regression model (`SplatModel`) whose
mother is a straight-ray travel time and whose combine is a (soft-)min. See
`wavelets.py` for the model and `train.py` for how it is grown and refined.

With uniform speed and no obstacles the source wavelet alone is exact and
nothing is added; obstacles are handled by adding wavelets where they
re-emit into shadows.

`solve(domain, *, source, <kwargs>) -> Model`, the same call shape as
`strategies.fsm` (`source` is a grid index into `domain.grid_shape`) so the
two fields are directly comparable node for node.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from ...domains import DTYPE, Domain
from . import train
from .wavelets import HuygensField


@dataclasses.dataclass
class Model:
    """A trained single-source field `T(x)` over PHYSICAL coordinates."""

    domain: Domain
    cfg: object
    wavelets: HuygensField
    params: tuple
    #: The source, physical coordinates.
    source: np.ndarray

    def __post_init__(self):
        self._time_fn = jax.jit(self.wavelets.evaluate)

    @property
    def num_splats(self) -> int:
        return int(self.params[0].shape[0])

    def time(self, X, batch_size: int = 4096) -> np.ndarray:
        """Arrival time at physical points `(n, dim)`, `(n,)`."""
        Xn = self._normalized(X)
        return np.concatenate([np.asarray(self._time_fn(self.params, Xn[i:i + batch_size]))
                               for i in range(0, Xn.shape[0], batch_size)])

    def gradient(self, X) -> np.ndarray:
        """`dT/dx` in physical coordinates, `(n, dim)`."""
        g = self.wavelets.grad(self.params, self._normalized(X))
        return np.asarray(g / jnp.asarray(self.domain.span, dtype=DTYPE))

    def grid_field(self, grid_shape: tuple[int, ...] | None = None) -> np.ndarray:
        """`T` on every node of a dense grid, shaped `grid_shape` (defaults to
        the domain's) -- the same layout `strategies.fsm.solve` returns."""
        gs = self.domain.grid_shape if grid_shape is None else tuple(grid_shape)
        nodes = np.asarray(self.domain.from_normalized(self.domain.grid(gs)))
        return self.time(nodes).reshape(gs)

    def descend(self, X, step: float = 0.02, n_steps: int = 500, backtracks: int = 6) -> np.ndarray:
        """Follow `-G^-1 grad T` (normalized to travel-time `step` per move)
        from physical points `X`. Returns trajectories `(n_steps + 1, n, dim)`.

        Each move takes the largest of `step, step/2, ...` (`backtracks`
        halvings) that lowers `T`, and stays put if none does. Without this,
        a path grazing an obstacle corner -- where it runs along a shadow
        boundary and `grad T` flips across it -- bounced between two points
        on either side forever."""
        field, domain = self.wavelets, self.domain
        scales = step * 0.5 ** jnp.arange(backtracks + 1, dtype=DTYPE)

        @jax.jit
        def advance(p, Xn):
            T = field.evaluate(p, Xn)
            g = field.grad(p, Xn)
            d = jnp.einsum("nij,nj->ni", domain.metric_inv(Xn), g)
            norm = jnp.sqrt(jnp.clip(jnp.einsum("ni,ni->n", d, g), 1e-12, None))
            trials = domain.wrap(Xn[None] - scales[:, None, None] * (d / norm[:, None])[None])  # (S, n, dim)
            T_trial = jax.vmap(lambda Y: field.evaluate(p, Y))(trials)  # (S, n)
            better = T_trial < T[None]
            first = jnp.argmax(better, axis=0)
            moved = trials[first, jnp.arange(Xn.shape[0])]
            return jnp.where(jnp.any(better, axis=0)[:, None], moved, Xn)

        Xn = self._normalized(X)
        traj = [Xn]
        for _ in range(n_steps):
            Xn = advance(self.params, Xn)
            traj.append(Xn)
        return np.asarray(domain.from_normalized(jnp.stack(traj)))

    def _normalized(self, X):
        return self.domain.wrap(self.domain.to_normalized(jnp.asarray(X, dtype=DTYPE)))


def solve(
        domain,
        *,
        source: tuple[int, ...] | None = None,
        # model (wavelets.py)
        eps: float = 0.0,
        ray_samples: int = 160,
        cone_delta: float = 1e-3,
        min_speed: float = 1e-2,
        obstacle_speed: float = 1e-3,
        occlusion_cost: float = 1e3,
        # growth / refinement (train.py)
        max_rounds: int = 30,
        max_splats: int = 64,
        spawn_per: int = 4,
        candidates: int = 512,
        jitter: float = 0.01,
        growth_tol: float = 1e-3,
        refine_steps: int = 100,
        batch_size: int = 1024,
        center_lr: float = 3e-3,
        grad_clip: float = 1.0,
        chunk: int = 64,
        seed: int = 0,
        progress_fn=None,
) -> Model:
    """Grow a Huygens-splat field from `source` on `domain`.

    Args:
        domain: an `eikonax.domains` domain. `source` needs its `grid_shape`.
        source: grid indices of the source (as `strategies.fsm.solve`);
            defaults to the grid centre.
        eps: soft-min temperature in travel-time units; `0` is the hard min
            (and keeps `T` a strict upper bound).
        ray_samples: samples per straight-ray cost -- their spacing must be
            finer than the thinnest obstacle.
        cone_delta: smoothing of the ray length at zero displacement.
        min_speed: speed floor for the slowness `1/speed`.
        obstacle_speed: speed at or below which a point is an obstacle: it
            blocks rays, and gets no collocation points or wavelets.
        occlusion_cost: added to a ray that crosses an obstacle.
        max_rounds, max_splats, spawn_per, candidates, growth_tol: growth --
            per round, up to `spawn_per` greedy additions from `candidates`
            random points, each needing a `mean T` drop of `growth_tol`
            (travel-time units). Stops after a round that adds nothing.
            `jitter` (normalized units) perturbs existing centres into extra
            candidates.
        refine_steps, batch_size, center_lr, grad_clip: per-round Adam
            refinement of the centres minimizing `mean T`.
        chunk: candidates per batched ray evaluation (memory).
        progress_fn(round, metrics): called after every round with
            `mean_T`, `uncovered` (fraction of samples no wavelet sees),
            `eikonal_rms` (diagnostic), `best_gain`, `num_splats`, `pruned`.
    """
    cfg = SimpleNamespace(**{k: v for k, v in locals().items() if k not in ("domain", "progress_fn", "source")})
    grid_shape = getattr(domain, "grid_shape", None)
    if grid_shape is None:
        raise ValueError("huygens.solve needs a domain with a grid_shape to place the grid-index source")
    if source is None:
        source = tuple(s // 2 for s in grid_shape)
    if len(source) != domain.dim:
        raise ValueError(f"source must have {domain.dim} indices, got {source}")
    nodes = np.asarray(domain.grid(grid_shape)).reshape(*grid_shape, domain.dim)
    source_n = nodes[tuple(int(s) for s in source)]

    field, params = train.solve(domain, source_n, cfg, progress_fn=progress_fn)
    source_phys = np.asarray(domain.from_normalized(jnp.asarray(source_n[None], dtype=DTYPE)))[0]
    return Model(domain=domain, cfg=cfg, wavelets=field, params=params, source=source_phys)


def make_config(**overrides) -> SimpleNamespace:
    """`solve`'s defaults as a `cfg` namespace, plus `overrides` -- for
    calling `HuygensField` / `train` internals directly in tests."""
    base = {
        name: p.default
        for name, p in inspect.signature(solve).parameters.items()
        if p.default is not inspect.Parameter.empty and name not in ("progress_fn", "source")
    }
    return SimpleNamespace(**{**base, **overrides})


__all__ = ["HuygensField", "Model", "make_config", "solve"]
