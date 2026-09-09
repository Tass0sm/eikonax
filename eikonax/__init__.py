"""JAX numerical solvers for eikonal / Hamilton-Jacobi-Bellman equations.

Two layers:

  - `fsm` / `se2` -- the grid Fast Sweeping numerics and their SE(2)
    plumbing (`fsm.Solver`, `se2.build_solver`, `se2.mask_speed_fn`). Used
    directly by downstream consumers.
  - `strategies` -- a uniform `solve(domain, *, <config kwargs>)` entry per
    method (`strategies.fsm`, `strategies.ntfields`), taking their
    environmental constraints as a shared `domains` object and driving the
    `eikonax.scripts.solve` CLI (see `eikonax.config`).
"""

from . import backends, config, domains, fsm, geometry, scenarios, se2, strategies
from .domains import DOMAINS, BoxDomain, se2_domain
from .scenarios import SCENARIOS
from .strategies import STRATEGIES
from .strategies.ntfields import (
    ARCHITECTURE_VERSION,
    TrainedField,
    load_ntfield,
    train_ntfield,
)

__all__ = [
    "ARCHITECTURE_VERSION",
    "DOMAINS",
    "SCENARIOS",
    "STRATEGIES",
    "BoxDomain",
    "TrainedField",
    "backends",
    "config",
    "domains",
    "fsm",
    "geometry",
    "load_ntfield",
    "scenarios",
    "se2",
    "se2_domain",
    "strategies",
    "train_ntfield",
]
