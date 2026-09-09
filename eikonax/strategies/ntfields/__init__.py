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
import pathlib
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from ... import geometry
from ...backends import BACKENDS, metric_net, time_and_grads
from ...domains import DTYPE, BoxDomain, Domain, dual_norm
from . import td_ntfields

#: `objective=` name -> module exposing `solve(domain, cfg, backend, progress_fn=None) -> params`.
OBJECTIVES = {"td_ntfields": td_ntfields}

#: Stamped into every checkpoint `save` writes and checked on `load_ntfield`.
#: This is the `backends.metric_net` network layout, NOT a training-data
#: version -- bump it only when a param-tree change would make an older
#: checkpoint load into the wrong shape. The downstream consumer
#: (`po_goc_mpc.experiments.objectives.ntfield`) folds it into its disk-cache
#: key so a bump retrains rather than loads stale weights.
ARCHITECTURE_VERSION = "eikonax_metric_net_v1"

#: `cfg` fields that fix the network's SHAPE -- the only ones a checkpoint
#: needs to carry to be reloadable (`metric_net.init` reads the first four,
#: `metric_net.travel_time` the rest). Everything else in `cfg` is a training
#: knob with no bearing on a trained field's evaluation.
_SHAPE_FIELDS = ("hidden", "n_blocks", "n_freq", "group",
                 "out_scale", "lse_scale", "softplus_beta")


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
    #: What the field's coordinates mean (`"workspace_xyz"`, a c-space id,
    #: ...). Set by `train_ntfield`; carried into `save`. Purely a label.
    coordinate_space: str = "unknown"

    def save(self, path) -> pathlib.Path:
        """Write the trained field to `path` (an `.npz`): the `params` tree,
        the network-shape `cfg` fields, the domain's box/periodicity, and the
        `coordinate_space` / `ARCHITECTURE_VERSION` tags. `load_ntfield`
        reads it back into a `TrainedField`."""
        path = pathlib.Path(path)
        leaves = jax.tree_util.tree_flatten_with_path(self.params)[0]
        blob = {f"param::{jax.tree_util.keystr(kp)}": np.asarray(v) for kp, v in leaves}
        n_freq = self.cfg.n_freq
        meta = {
            "coordinate_space": self.coordinate_space,
            "architecture_version": ARCHITECTURE_VERSION,
            "lower": np.asarray(self.domain.lower, dtype=np.float64),
            "upper": np.asarray(self.domain.upper, dtype=np.float64),
            "periodic": np.asarray(self.domain.periodic, dtype=bool),
            "n_freq": np.asarray(-1 if n_freq is None else int(n_freq)),
            **{name: np.asarray(getattr(self.cfg, name))
               for name in _SHAPE_FIELDS if name != "n_freq"},
        }
        np.savez(path, **blob, **{f"meta::{k}": v for k, v in meta.items()})
        return path

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
        roadmap_segment_samples: int = 16,
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


@dataclasses.dataclass
class TrainedField:
    """A field reloaded from disk: enough to EVALUATE `T(x0, x1)`, not to
    resume training. `travel_time` is `jit`/`vmap`/`grad`-safe on PHYSICAL
    coordinates -- the single primitive an `edge_cost_fn(a, b)` wrapper
    needs -- and normalization (`(x - lo)/span - 0.5`, then periodic-wrap /
    box-clamp) happens inside it.
    """

    domain: BoxDomain
    params: object
    cfg: object
    coordinate_space: str
    architecture_version: str

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.domain.lower)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.domain.upper)

    @property
    def dim(self) -> int:
        return self.domain.dim

    @property
    def periodic(self) -> tuple[bool, ...]:
        return self.domain.periodic

    def travel_time(self, X0, X1) -> jnp.ndarray:
        """`T(x0, x1)` for batches of physical coordinates `(n, dim)`, `(n,)`
        out. Stays traceable -- no `np.asarray` on the path."""
        Xn0 = self.domain.wrap(self.domain.to_normalized(jnp.asarray(X0, dtype=DTYPE)))
        Xn1 = self.domain.wrap(self.domain.to_normalized(jnp.asarray(X1, dtype=DTYPE)))
        return metric_net.travel_time(self.params, Xn0, Xn1, self.cfg)

    def time(self, X0, X1) -> np.ndarray:
        """`np.asarray(self.travel_time(...))` -- the eager convenience form."""
        return np.asarray(self.travel_time(X0, X1))


def load_ntfield(path) -> TrainedField:
    """Read a checkpoint written by `Model.save` back into a `TrainedField`.

    The `params` tree is rebuilt by re-`init`-ing the network at the saved
    shape (for a correct skeleton) and swapping every leaf for the stored
    array, matched by its `jax.tree_util` key path -- so it does not depend
    on leaf ORDER, only on the layout `metric_net.Params` declares.
    """
    path = pathlib.Path(path)
    with np.load(path) as data:
        blob = {name: data[name] for name in data.files}
    meta = {name[len("meta::"):]: value
            for name, value in blob.items() if name.startswith("meta::")}

    n_freq = int(meta["n_freq"])
    cfg = make_config(
        n_freq=None if n_freq < 0 else n_freq,
        **{name: (int(meta[name]) if name in ("hidden", "n_blocks", "group")
                  else float(meta[name]))
           for name in _SHAPE_FIELDS if name != "n_freq"},
    )
    periodic = tuple(bool(flag) for flag in meta["periodic"])
    domain = BoxDomain(
        np.asarray(meta["lower"]), np.asarray(meta["upper"]), periodic,
        lambda X: jnp.ones(jnp.asarray(X).shape[:-1], dtype=DTYPE))

    skeleton = metric_net.init(jax.random.PRNGKey(0), domain, cfg)
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(skeleton)
    params = jax.tree_util.tree_unflatten(
        treedef,
        [jnp.asarray(blob[f"param::{jax.tree_util.keystr(kp)}"])
         for kp, _ in leaves_with_path],
    )
    return TrainedField(
        domain=domain, params=params, cfg=cfg,
        coordinate_space=str(meta["coordinate_space"]),
        architecture_version=str(meta["architecture_version"]),
    )


def train_ntfield(
        *,
        coordinate_space: str,
        normalization_box,
        obstacle_boxes,
        margin: float,
        fk=None,
        sphere_radii=(0.0,),
        speed_floor: float = 1e-2,
        speed_smoothstep_geometry: bool = False,
        frame_offset=None,
        out=None,
        **solve_kwargs,
) -> Model:
    """Train an NTField for one obstacle scene and (optionally) save it.

    Builds the geometry speed field (`geometry.spheres_speed_fn`) and a
    `BoxDomain` over `normalization_box = (lower, upper)`, then runs
    `solve(domain, **solve_kwargs)`.

    Args:
        coordinate_space: label stored on the checkpoint. `"workspace_xyz"`
            is the only space that works without an explicit `fk`.
        normalization_box: `(lower, upper)`, each length-`dim`, the frame the
            field's coordinates live in.
        obstacle_boxes: `(centers, half_extents)`, each `(K, 3)`, SAME frame
            as the FK sphere centres (see `frame_offset`).
        margin: clearance (m) at which the speed reaches free space.
        fk: `coords -> (..., n_spheres, 3)` sphere centres. Defaults to
            `geometry.identity_fk` (workspace only).
        sphere_radii: `(n_spheres,)`, matching `fk`'s output order.
        speed_floor: floor speed at/inside an obstacle
            (`geometry.spheres_speed_fn`'s `min_speed`).
        speed_smoothstep_geometry: `C1`-shape the geometry ramp (distinct
            from `solve`'s own `speed_smoothstep`, which remaps whatever the
            domain returns).
        frame_offset: translation subtracted from FK sphere centres before
            the obstacle-distance query.
        out: if given, `Model.save(out)` after training.
        **solve_kwargs: forwarded to `solve` (epochs, seed, weights, ...).
    """
    lower, upper = normalization_box
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    dim = len(lower)

    if fk is None:
        if coordinate_space != "workspace_xyz":
            raise ValueError(
                f"coordinate_space {coordinate_space!r} needs an explicit fk= "
                "(only 'workspace_xyz' defaults to geometry.identity_fk)")
        fk = geometry.identity_fk

    centers, half_extents = obstacle_boxes
    speed_fn = geometry.spheres_speed_fn(
        fk, sphere_radii, centers, half_extents,
        margin=margin, min_speed=speed_floor,
        smoothstep=speed_smoothstep_geometry, frame_offset=frame_offset)

    domain = BoxDomain(lower, upper, (False,) * dim, speed_fn)
    model = solve(domain, **solve_kwargs)
    model.coordinate_space = coordinate_space
    if out is not None:
        model.save(out)
    return model


__all__ = [
    "OBJECTIVES", "ARCHITECTURE_VERSION", "Model", "TrainedField",
    "load_ntfield", "make_config", "solve", "train_ntfield",
]
