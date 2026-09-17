"""Wavefront strategy: a single-source travel-time field as a partition of
unity over LOCAL travel-time models, chained outward from the source.

Each splat is a compactly supported window (an SRM density) whose weight is
a local quadratic model of the arrival time -- emission time, a slope tied
to the speed at the splat's centre, and a second-order term -- instead of a
scalar. A splat placed at the edge of the previous one's window continues
the wave with the local speed there, so smoothly varying speed needs no ray
marching: slow regions get short steps and steep slopes, uniform regions a
few long ones. See `field.py` for the model, `chain.py` for how the splats
are grown, `train.py` for the gradient refinement of every parameter, and
`baselines.py` for the 1-D comparisons (`python -m eikonax.scripts.wavefront_1d`).

1-D only for now (the model is written for any dimension; the chain
initializer is not). Pure JAX, no `srms` dependency.

`solve(domain, *, source, <kwargs>) -> Model`, the same call shape as
`strategies.fsm` (`source` is a grid index into `domain.grid_shape`).
"""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from ...domains import DTYPE, Domain
from . import baselines, chain, train
from .field import WavefrontField


@dataclasses.dataclass
class Model:
    """A trained single-source field `T(x)` over PHYSICAL coordinates."""

    domain: Domain
    cfg: object
    wavefront: WavefrontField
    params: dict
    #: The source, physical coordinates.
    source: np.ndarray

    def __post_init__(self):
        self._time_fn = jax.jit(self.wavefront.evaluate)
        self._grad_fn = jax.jit(self.wavefront.grad)

    @property
    def num_splats(self) -> int:
        return self.wavefront.num_splats(self.params)

    def time(self, X, batch_size: int = 4096) -> np.ndarray:
        """Arrival time at physical points `(n, dim)`, `(n,)`."""
        Xn = self._normalized(X)
        return np.concatenate([np.asarray(self._time_fn(self.params, Xn[i:i + batch_size]))
                               for i in range(0, Xn.shape[0], batch_size)])

    def gradient(self, X) -> np.ndarray:
        """`dT/dx` in physical coordinates, `(n, dim)`."""
        g = self._grad_fn(self.params, self._normalized(X))
        return np.asarray(g / jnp.asarray(self.domain.span, dtype=DTYPE))

    def grid_field(self, grid_shape: tuple[int, ...] | None = None) -> np.ndarray:
        """`T` on every node of a dense grid, shaped `grid_shape` (defaults to
        the domain's) -- the same layout `strategies.fsm.solve` returns."""
        gs = self.domain.grid_shape if grid_shape is None else tuple(grid_shape)
        nodes = np.asarray(self.domain.from_normalized(self.domain.grid(gs)))
        return self.time(nodes).reshape(gs)

    def _normalized(self, X):
        return self.domain.wrap(self.domain.to_normalized(jnp.asarray(X, dtype=DTYPE)))


def solve(
        domain,
        *,
        source: tuple[int, ...] | None = None,
        # model (field.py)
        value_temperature: float | None = None,
        min_speed: float = 1e-2,
        cone_delta: float = 1e-6,
        # chain (chain.py)
        tol: float = 1e-3,
        r_max: float = 0.25,
        r_min: float = 2e-3,
        step_shrink: float = 0.8,
        step_samples: int = 16,
        overlap: float = 0.75,
        # refinement (train.py)
        train_steps: int = 0,
        batch_size: int = 512,
        lr: float = 1e-3,
        center_lr: float = 1e-5,
        grad_clip: float = 1.0,
        residual_weight: float = 1.0,
        consistency_weight: float = 1.0,
        coverage_weight: float = 1.0,
        source_weight: float = 10.0,
        min_coverage: float = 0.2,
        seed: int = 0,
        progress_fn=None,
) -> Model:
    """Chain a wavefront field out of `source` on `domain`, then refine it.

    Args:
        domain: a 1-D `eikonax.domains` domain with a `grid_shape` (e.g.
            `line_domain`).
        source: grid index of the source (as `strategies.fsm.solve`);
            defaults to the grid centre.
        value_temperature: `> 0` adds a soft-min preference for the earliest
            local arrival to the blend weights; `None` is the pure partition
            of unity.
        min_speed: speed floor for the slowness `1/speed`.
        cone_delta: smoothing of the source cone at its apex (normalized).
        tol: per-step third-order error budget of the chain (travel-time
            units); smaller means more, shorter splats.
        r_max, r_min, step_shrink, step_samples: chain step search, in
            normalized units -- the largest step `r_max * step_shrink^i`
            within `tol`, `|n''|` checked at `step_samples` points along it.
        overlap: window radius as a fraction of the neighbouring spacing.
        train_steps, batch_size, lr, center_lr, grad_clip: Adam refinement
            of every parameter. Off by default: it helps a coarse chain but
            degrades an accurate one (see `train.py`).
        residual_weight, consistency_weight, coverage_weight, source_weight,
            min_coverage: loss terms, see `train.py`.
        progress_fn(step, metrics): called through refinement with the loss
            terms.
    """
    cfg = SimpleNamespace(**{k: v for k, v in locals().items() if k not in ("domain", "progress_fn", "source")})
    grid_shape = getattr(domain, "grid_shape", None)
    if grid_shape is None:
        raise ValueError("wavefront.solve needs a domain with a grid_shape to place the grid-index source")
    if source is None:
        source = tuple(s // 2 for s in grid_shape)
    if len(source) != domain.dim:
        raise ValueError(f"source must have {domain.dim} indices, got {source}")
    nodes = np.asarray(domain.grid(grid_shape)).reshape(*grid_shape, domain.dim)
    source_n = nodes[tuple(int(s) for s in source)]

    field = WavefrontField(domain, source_n, cfg)
    params = chain.chain_1d(field)
    params = train.refine(field, params, np.random.default_rng(seed), progress_fn=progress_fn)
    source_phys = np.asarray(domain.from_normalized(jnp.asarray(source_n[None], dtype=DTYPE)))[0]
    return Model(domain=domain, cfg=cfg, wavefront=field, params=params, source=source_phys)


def make_config(**overrides) -> SimpleNamespace:
    """`solve`'s defaults as a `cfg` namespace, plus `overrides` -- for
    calling `WavefrontField` / `chain` / `train` internals directly."""
    base = {
        name: p.default
        for name, p in inspect.signature(solve).parameters.items()
        if p.default is not inspect.Parameter.empty and name not in ("progress_fn", "source")
    }
    return SimpleNamespace(**{**base, **overrides})


__all__ = ["Model", "WavefrontField", "baselines", "make_config", "solve"]
