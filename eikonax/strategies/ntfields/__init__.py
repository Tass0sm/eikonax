"""Neural travel-time field strategy: a learned, continuous, ALL-PAIRS
alternative to the grid sweep. `fsm` sweeps one grid per source; `ntfields`
fits a single two-point network `T(x0, x1)` to the same eikonal equation by
physics-informed training -- no grid, no per-source solve, differentiable, at
the price of being approximate.

`solve(domain, *, objective="td_ntfields", backend="metric_net", ...)` -- the
keyword arguments are the configuration (defaults are the `ntrl-demo`
reference implementation's; each field's *why* lives in the module it
belongs to, `backends/metric_net.py` and the objective module). `objective`
selects the training objective:

  - `td_ntfields` -- TD-NTFields (Ni, Pan & Qureshi, ICLR 2025): eikonal +
    Bellman + obstacle-normal losses under a causality curriculum,
    generalized to an arbitrary Riemannian metric.

Returns a `Model`: the trained field plus `.time` / `.gradient` / `.speed` /
`.field` helpers over PHYSICAL coordinates.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from ...backends import BACKENDS, time_and_grads
from ...domains import DTYPE, Domain, dual_norm
from . import td_ntfields

#: `objective=` name -> module exposing `solve(domain, cfg, backend, progress_fn=None) -> params`.
OBJECTIVES = {"td_ntfields": td_ntfields}


@dataclasses.dataclass
class Model:
    """A trained two-point travel-time field: `T(x0, x1)` for any pair of
    PHYSICAL coordinates, plus the quantities derived from it.

    Unlike `fsm`'s per-source grid, this field is continuous and all-pairs,
    so there is nothing to re-solve per source -- `field` just evaluates it
    on a grid for comparison.
    """

    domain: Domain
    cfg: object
    params: object
    backend: ModuleType

    def time(self, X0, X1) -> np.ndarray:
        """Arrival time between batches of physical coordinates, `(n,)`."""
        Xn0, Xn1 = self._normalized(X0), self._normalized(X1)
        return np.asarray(self.backend.travel_time(self.params, Xn0, Xn1, self.cfg))

    def gradient(self, X0, X1) -> tuple[np.ndarray, np.ndarray]:
        """`(dT/dx0, dT/dx1)` in physical coordinates, each `(n, dim)`."""
        Xn0, Xn1 = self._normalized(X0), self._normalized(X1)
        _, g0, g1 = time_and_grads(self.backend, self.params, Xn0, Xn1, self.cfg)
        span = jnp.asarray(self.domain.span, dtype=DTYPE)
        return np.asarray(g0 / span), np.asarray(g1 / span)

    def speed(self, X0, X1) -> np.ndarray:
        """The speed the field implies at `x0`, `1 / |dT/dx0|` in the
        domain's dual norm -- compare against `domain.speed`."""
        Xn0, Xn1 = self._normalized(X0), self._normalized(X1)
        _, g0, _ = time_and_grads(self.backend, self.params, Xn0, Xn1, self.cfg)
        return np.asarray(1.0 / dual_norm(g0, self.domain.metric_inv(Xn0)))

    def field(self, source, grid_shape: tuple[int, ...], batch_size: int = 8192) -> np.ndarray:
        """`T(source, node)` at every node of a dense `grid_shape` grid over
        the domain, shaped `grid_shape`. `source` is a physical coordinate."""
        nodes = self.domain.grid(grid_shape)
        src = jnp.broadcast_to(self._normalized(np.asarray(source)[None, :]), nodes.shape)
        out = [
            self.backend.travel_time(self.params, src[i:i + batch_size], nodes[i:i + batch_size], self.cfg)
            for i in range(0, nodes.shape[0], batch_size)
        ]
        return np.asarray(jnp.concatenate(out)).reshape(grid_shape)

    def _normalized(self, X):
        return self.domain.wrap(self.domain.to_normalized(jnp.asarray(X, dtype=DTYPE)))


def solve(
        domain,
        *,
        objective: str = "td_ntfields",
        backend: str = "metric_net",
        # backend (backends/metric_net.py)
        hidden: int = 256,
        n_blocks: int = 2,
        n_freq: int | None = None,
        group: int = 16,
        out_scale: float = 0.2,
        lse_scale: float = 10.0,
        softplus_beta: float = 10.0,
        # objective (strategies/ntfields/td_ntfields.py)
        eikonal_weight: float = 1e-2,
        td_weight: float = 1e-3,
        normal_weight: float = 1e-3,
        causal_lambda: float = 0.5,
        detach_causal: bool = False,
        td_step: float = 0.03,
        pair_radius: float | None = None,
        speed_alpha: float = 1.025,
        speed_smoothstep: bool = True,
        min_speed: float = 1e-2,
        # weak supervision: PRM anchor (roadmap.py), off by default
        roadmap_weight: float = 0.0,
        roadmap_nodes: int = 256,
        roadmap_k: int = 10,
        roadmap_segment_samples: int = 8,
        # budget
        epochs: int = 5000,
        batches_per_epoch: int = 5,
        batch_size: int = 2000,
        lr: float = 5e-4,
        weight_decay: float = 0.5,
        seed: int = 0,
        # rollback / loss rescaling (see the objective module)
        rollback: bool = True,
        rollback_ratio: float = 1.2,
        rollback_queue: int = 5,
        rollback_max_retries: int = 10,
        adaptive_beta: bool = True,
        log_every: int = 10,
        progress_fn=None,
) -> Model:
    """Train `objective` against `backend` on `domain`. `progress_fn(epoch,
    metrics)`, if given, is called every `log_every` epochs with scalar
    training metrics. Returns the trained `Model`."""
    cfg = SimpleNamespace(**{k: v for k, v in locals().items() if k not in ("domain", "progress_fn")})

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}, expected one of {sorted(OBJECTIVES)}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {sorted(BACKENDS)}")
    backend_module = BACKENDS[backend]
    params = OBJECTIVES[objective].solve(domain, cfg, backend_module, progress_fn=progress_fn)
    return Model(domain=domain, cfg=cfg, params=params, backend=backend_module)


def make_config(**overrides) -> SimpleNamespace:
    """A `cfg` namespace with `solve`'s defaults, plus `overrides` -- for
    calling `backends` / `OBJECTIVES` internals (`metric_net.init`,
    `td_ntfields.loss_terms`) directly, e.g. in tests."""
    base = {
        name: p.default
        for name, p in inspect.signature(solve).parameters.items()
        if p.default is not inspect.Parameter.empty and name != "progress_fn"
    }
    return SimpleNamespace(**{**base, **overrides})


__all__ = ["OBJECTIVES", "Model", "make_config", "solve"]
