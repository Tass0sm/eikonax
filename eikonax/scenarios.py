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

Each `speed_fn` also carries two attributes describing the same obstacles
in a form a distance query understands (`Domain.clearance_fn` /
`Domain.obstacle_rects` hand them to solvers that need them, e.g.
`strategies.wavefront` in 2-D):

  - `speed_fn.clearance(coords) -> distance to the nearest obstacle`
    (physical units, `OPEN` where there are none) -- the analytic stand-in
    for a collision checker's distance query, which is what a robot
    configuration space offers instead of an analytic obstacle;
  - `speed_fn.rects` -- the obstacles as `(y0, y1, x0, x1)` rectangles,
    for exact references (a visibility graph).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from .geometry import box_sdf

#: Clearance reported where a scene has no obstacles at all -- "far enough
#: that nothing limits a window".
OPEN = 1e3


def _rect_clearance(rects) -> Callable:
    """Distance to the union of `(y0, y1, x0, x1)` rectangles, `OPEN` if none.

    Reads the first two axes only, so it applies unchanged to an SE(2)
    domain's `(y, x, theta)` coordinates.
    """
    if not rects:
        return lambda coords: jnp.full(coords.shape[:-1], OPEN)
    centers = jnp.asarray([[(y0 + y1) / 2, (x0 + x1) / 2] for y0, y1, x0, x1 in rects])
    half = jnp.asarray([[(y1 - y0) / 2, (x1 - x0) / 2] for y0, y1, x0, x1 in rects])

    def clearance(coords):
        return box_sdf(jnp.asarray(coords)[..., :2], centers, half)
    return clearance


def _with_obstacles(speed_fn: Callable, rects) -> Callable:
    """Attach `rects` and the matching `clearance` to a speed field."""
    speed_fn.rects = list(rects)
    speed_fn.clearance = _rect_clearance(rects)
    return speed_fn


#: A rectangle's "runs off the domain" end.
FAR = 1e3


def free() -> Callable:
    """Unobstructed space -- speed 1 everywhere."""
    def speed_fn(coords):
        return jnp.ones(coords.shape[:-1])
    return _with_obstacles(speed_fn, [])


def wall(wall_x: float = 2.0, wall_y_max: float = 2.8, thickness: float = 0.05) -> Callable:
    """A single wall along `x = wall_x`, blocking `y < wall_y_max` -- a
    barrier with a gap past its far end, forcing a detour around it."""
    def speed_fn(coords):
        y, x = coords[..., 0], coords[..., 1]
        in_wall = (jnp.abs(x - wall_x) < thickness / 2) & (y < wall_y_max)
        return jnp.where(in_wall, 0.0, 1.0)
    return _with_obstacles(speed_fn, [(-FAR, wall_y_max, wall_x - thickness / 2, wall_x + thickness / 2)])


def gap(wall_x: float = 2.0, gap_center: float = 2.0, gap_width: float = 0.6,
        thickness: float = 0.05) -> Callable:
    """A full wall along `x = wall_x` with a `gap_width` opening centred on
    `y = gap_center` -- a doorway the path has to thread."""
    def speed_fn(coords):
        y, x = coords[..., 0], coords[..., 1]
        in_slab = jnp.abs(x - wall_x) < thickness / 2
        in_gap = jnp.abs(y - gap_center) < gap_width / 2
        return jnp.where(in_slab & ~in_gap, 0.0, 1.0)
    x0, x1 = wall_x - thickness / 2, wall_x + thickness / 2
    return _with_obstacles(speed_fn, [(-FAR, gap_center - gap_width / 2, x0, x1),
                                      (gap_center + gap_width / 2, FAR, x0, x1)])


def slow(center_y: float = 2.0, center_x: float = 2.0, radius: float = 0.5,
         depth: float = 0.7) -> Callable:
    """A smooth slow patch: `speed = 1 - depth * exp(-r^2 / (2 radius^2))`,
    `r` the distance to `(center_y, center_x)` over the first two axes (a
    1-D domain's single axis plays the role of `y`). No obstacle -- the
    speed bottoms out at `1 - depth`."""
    centre = jnp.asarray([center_y, center_x])

    def speed_fn(coords):
        k = min(coords.shape[-1], 2)
        r2 = jnp.sum((coords[..., :k] - centre[:k]) ** 2, axis=-1)
        return 1.0 - depth * jnp.exp(-r2 / (2.0 * radius ** 2))
    return _with_obstacles(speed_fn, [])


#: `--scenario` name -> speed-field factory.
SCENARIOS: dict[str, Callable[..., Callable]] = {
    "free": free,
    "wall": wall,
    "gap": gap,
    "slow": slow,
}
