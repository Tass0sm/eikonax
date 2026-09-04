"""Solve strategies. Each is a module exposing

    solve(domain, *, <configuration keyword arguments>, progress_fn=None)

whose keyword arguments *are* its configuration -- `eikonax.scripts.solve`
builds its CLI by introspecting the signature (see `eikonax.config`). Both
strategies take their environmental constraints the same way: a `domain`
object (`eikonax.domains`) carrying the geometry, periodicity, speed field
and metric.

  - `fsm` -- the grid Fast Sweeping sweep (`eikonax.fsm`), exact but one
    solve per source.
  - `ntfields` -- physics-informed training of a continuous all-pairs
    neural travel-time field; its `objective=` kwarg selects the
    implementation (`td_ntfields`).
"""

from . import fsm, ntfields

STRATEGIES = {"fsm": fsm, "ntfields": ntfields}

__all__ = ["STRATEGIES", "fsm", "ntfields"]
