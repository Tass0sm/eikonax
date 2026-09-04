"""Named obstacle layouts for the CLI.

A `speed_fn(coords) -> speed` is a Python callable, so it cannot be passed
on a command line. `--scenario <name>` picks one of these factories instead;
each returns an analytic, JAX-jittable speed field over PHYSICAL coordinates
(same convention `fsm.Solver` / `Domain.speed` use: `coords[..., 0]` is y,
`[..., 1]` is x, and any further axis -- a heading -- is ignored, since
occupancy does not depend on it). Obstacles are `speed = 0`, free space is
`speed = 1`; a factory's keyword arguments become `--flags` via
`eikonax.config.build_subcommand_parser`.

Python callers who want a richer field (slow terrain, a soft margin, a real
occupancy grid via `se2.mask_speed_fn`) pass their own `speed_fn` straight
to the domain constructor and skip this module.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp


def free() -> Callable:
    """Unobstructed space -- speed 1 everywhere."""
    def speed_fn(coords):
        return jnp.ones(coords.shape[:-1])
    return speed_fn


def wall(wall_x: float = 2.0, wall_y_max: float = 2.8, thickness: float = 0.05) -> Callable:
    """A single wall along `x = wall_x`, blocking `y < wall_y_max` -- a
    barrier with a gap past its far end, forcing a detour around it."""
    def speed_fn(coords):
        y, x = coords[..., 0], coords[..., 1]
        in_wall = (jnp.abs(x - wall_x) < thickness / 2) & (y < wall_y_max)
        return jnp.where(in_wall, 0.0, 1.0)
    return speed_fn


def gap(wall_x: float = 2.0, gap_center: float = 2.0, gap_width: float = 0.6,
        thickness: float = 0.05) -> Callable:
    """A full wall along `x = wall_x` with a `gap_width` opening centred on
    `y = gap_center` -- a doorway the path has to thread."""
    def speed_fn(coords):
        y, x = coords[..., 0], coords[..., 1]
        in_slab = jnp.abs(x - wall_x) < thickness / 2
        in_gap = jnp.abs(y - gap_center) < gap_width / 2
        return jnp.where(in_slab & ~in_gap, 0.0, 1.0)
    return speed_fn


#: `--scenario` name -> speed-field factory.
SCENARIOS: dict[str, Callable[..., Callable]] = {
    "free": free,
    "wall": wall,
    "gap": gap,
}
