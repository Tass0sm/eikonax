"""Grid Fast Sweeping strategy: sweep `domain`'s grid to an exact arrival-time
field. One solve per source (`eikonax.fsm.Solver` compiles the sweep once and
reuses it), so an all-pairs field is `n_sources` sweeps -- `all_pairs=True`
runs that loop.
"""

from __future__ import annotations

import itertools

import numpy as np

from .. import fsm


def solve(
        domain,
        *,
        radius: int = 2,
        source: tuple[int, ...] | None = None,
        all_pairs: bool = False,
        n_iters: int = 300,
        tol: float = 1e-5,
        obstacle_fill: float = fsm.OBSTACLE_FILL,
        progress_fn=None,
):
    """Sweep `domain`'s grid.

    Args:
        domain: an `eikonax.domains` domain with a `grid_shape` (e.g. from
            `se2_domain`). `domain.fsm_solver` supplies the grid, metric and
            speed field.
        radius: candidate-move radius -- `>= 2` lets one move translate AND
            rotate at once (see `eikonax.fsm`).
        source: grid indices of the single source state. Defaults to the
            grid centre. Ignored when `all_pairs=True`.
        all_pairs: solve from every non-obstacle source and return a
            `(*grid_shape, *grid_shape)` field, `field[src][dst]` being the
            arrival time from `src` to `dst`.
        n_iters, tol: value-iteration budget per source.
        obstacle_fill: finite sentinel for unreachable / obstacle nodes.
        progress_fn(step, metrics): called through the all-pairs loop.

    Returns:
        A numpy array: `grid_shape` for a single source, or
        `(*grid_shape, *grid_shape)` for `all_pairs`.
    """
    solver = domain.fsm_solver(radius=radius, obstacle_fill=obstacle_fill)
    grid_shape = solver.grid_shape

    if not all_pairs:
        if source is None:
            source = tuple(s // 2 for s in grid_shape)
        return solver.solve(tuple(int(s) for s in source), n_iters=n_iters, tol=tol)

    free = np.asarray(solver.speed_field) > 0.0
    sources = [idx for idx in itertools.product(*(range(s) for s in grid_shape)) if free[idx]]
    field = np.full((*grid_shape, *grid_shape), obstacle_fill, dtype=np.float64)
    for i, src in enumerate(sources):
        field[src] = solver.solve(src, n_iters=n_iters, tol=tol)
        if progress_fn is not None and (i % max(1, len(sources) // 100) == 0 or i == len(sources) - 1):
            progress_fn(i + 1, {"n_sources": len(sources), "source": src})
    return field
