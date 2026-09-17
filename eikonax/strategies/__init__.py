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
  - `huygens` -- a single-source field as a min over ray wavelets grown
    from the source (an `srms` splat regression model). Only registered
    when `srms` is installed (the `splats` extra).
  - `wavefront` -- a single-source field as a partition of unity over
    local travel-time models chained out from the source (1-D for now).
"""

from . import fsm, huygens, ntfields, wavefront

STRATEGIES = {"fsm": fsm, "huygens": huygens, "ntfields": ntfields, "wavefront": wavefront}

__all__ = ["STRATEGIES", "fsm", "huygens", "ntfields", "wavefront"]
