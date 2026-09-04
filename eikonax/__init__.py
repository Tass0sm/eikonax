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

from . import backends, config, domains, fsm, scenarios, se2, strategies
from .domains import DOMAINS, BoxDomain, se2_domain
from .scenarios import SCENARIOS
from .strategies import STRATEGIES

__all__ = [
    "DOMAINS",
    "SCENARIOS",
    "STRATEGIES",
    "BoxDomain",
    "backends",
    "config",
    "domains",
    "fsm",
    "scenarios",
    "se2",
    "se2_domain",
    "strategies",
]
