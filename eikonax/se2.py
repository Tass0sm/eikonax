"""SE(2) (R^2 x S^1) application of the general `eikonax.fsm` solver: builds
the grid/periodicity/metric-field plumbing `fsm.Solver` needs for a
"unicycle-ish" `(x, y, theta)` state space, plus a convenience metric-matrix
builder for the standard "soft preference for moving/turning toward the
current heading" model. ANY metric matrix can be substituted (`build_solver`'s
`metric_at_theta` argument) -- including one with genuine
translation-rotation coupling that this default, purely-diagonal
`xi_lateral`/`xi_turn` matrix cannot express. See `fsm.py`'s own module
docstring for why that coupling matters: with only "translate-only" and
"rotate-only" candidate moves (this package's own earlier design), no
metric choice could ever produce a shortest path that actually turns into
its direction of travel -- `fsm.py`'s combined-offset candidates fix that
structurally; the metric matrix built here just decides how STRONGLY that
preference is expressed once it's actually representable.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from . import fsm


def default_metric_at_theta(thetas: np.ndarray, xi_lateral: float, xi_turn: float) -> np.ndarray:
    """The "soft preference" metric matrix `G(theta)` at every theta bin:
    `F(theta, dy, dx, dtheta)^2 = u1^2 + u2^2/xi_lateral^2 +
    dtheta^2/xi_turn^2`, `(u1, u2)` = `(dy, dx)` resolved into the
    forward/lateral frame at heading `theta` -- moving along the current
    heading is unit cost, moving laterally costs `1/xi_lateral`, turning
    costs `1/xi_turn`, all independently tunable (see this module's own
    docstring for why they're decoupled, not one shared ratio). Returns
    shape `(n_theta, 3, 3)`; `xi_lateral`/`xi_turn` both in `(0, 1]`.

    NOTE the `(dy, dx, dtheta)` input order, not `(dx, dy, dtheta)`: this
    has to match `build_solver`'s `grid_shape = (ny, nx, n_theta)` axis
    order (axis 0 = y/row, matching `_mask_and_coords`'s own `mask[sy,
    sx]` convention that the rest of this project's occupancy-grid code
    already uses), since that's the order `fsm.Solver` indexes physical
    displacement components in. Getting this backwards is a real bug this
    module's own dev hit: at `theta=0` exactly, `sin(0)=0` collapses the
    rotation to the identity matrix EITHER WAY, so a `(dx, dy, dtheta)`-
    ordered version of this function looks perfectly correct for a
    `theta=0`-only test and only misattributes forward/lateral for every
    OTHER heading -- caught by testing at a nonzero heading too.
    """
    n_theta = len(thetas)
    d = np.diag([1.0, 1.0 / xi_lateral ** 2, 1.0 / xi_turn ** 2])
    metric = np.empty((n_theta, 3, 3))
    for k, theta in enumerate(thetas):
        c, s = np.cos(theta), np.sin(theta)
        # u1 (forward) = dx*cos + dy*sin; u2 (lateral) = dy*cos - dx*sin;
        # u3 = dtheta -- expressed here as a matrix acting on (dy, dx,
        # dtheta), per the NOTE above.
        a = np.array([[s, c, 0.0], [c, -s, 0.0], [0.0, 0.0, 1.0]])
        metric[k] = a.T @ d @ a
    return metric


def mask_speed_fn(
        mask: np.ndarray, resolution: float, origin_xy=(0.0, 0.0),
        obstacle_speed: float = 0.0, free_speed: float = 1.0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Bridges a binary `(ny, nx)` occupancy mask (True = obstacle, e.g.
    from `po_goc_mpc.experiments.objectives.fmm._mask_and_coords`) into the
    JAX-jittable `speed_fn(coords) -> speed` `fsm.Solver` wants (see that
    module's own docstring for why a general speed field replaced a mask
    parameter there) -- a convenience for the common case, not the only way
    to build one: pass your own `speed_fn` directly to `build_solver`
    instead for anything richer (slow terrain, a soft margin around
    obstacles, ...).

    `coords`' theta component (axis -1, index 2) is ignored -- occupancy
    doesn't depend on heading.

    NOTE `coords[..., 0]` is Y (row) and `coords[..., 1]` is X (col), not
    the other way around -- matches `build_solver`'s `grid_shape = (ny,
    nx, n_theta)` axis order (axis 0 = row = y, per `_mask_and_coords`'s
    own `mask[sy, sx]` convention). Getting this backwards is the same
    class of bug `default_metric_at_theta`'s own docstring describes.
    """
    mask_j = jnp.asarray(mask)
    ny, nx = mask.shape
    x0, y0 = origin_xy

    def speed_fn(coords: jnp.ndarray) -> jnp.ndarray:
        yi = jnp.clip(jnp.round((coords[..., 0] - y0) / resolution).astype(jnp.int32), 0, ny - 1)
        xi = jnp.clip(jnp.round((coords[..., 1] - x0) / resolution).astype(jnp.int32), 0, nx - 1)
        occupied = mask_j[yi, xi]
        return jnp.where(occupied, obstacle_speed, free_speed)

    return speed_fn


def build_solver(
        ny: int,
        nx: int,
        resolution: float,
        n_theta: int,
        speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
        origin_xy=(0.0, 0.0),
        metric_at_theta: np.ndarray | None = None,
        xi_lateral: float = 0.4,
        xi_turn: float = 0.9,
        radius: int = 2,
        obstacle_fill: float = fsm.OBSTACLE_FILL,
) -> tuple[fsm.Solver, np.ndarray]:
    """Returns `(solver, thetas)`: an `fsm.Solver` over the `(ny, nx,
    n_theta)` SE(2) grid (x, y non-periodic at `resolution`; theta
    periodic, `n_theta` evenly-spaced bins), using `metric_at_theta`
    (defaulting to `default_metric_at_theta(thetas, xi_lateral, xi_turn)`
    if not given -- pass your own `(n_theta, 3, 3)` array for a fully
    customized metric) broadcast across every `(x, y)` position (the
    metric is position-INDEPENDENT here -- only `speed_fn` varies by
    position; build a position-dependent metric yourself and call
    `fsm.Solver` directly if you need that).
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    if metric_at_theta is None:
        metric_at_theta = default_metric_at_theta(thetas, xi_lateral, xi_turn)
    elif metric_at_theta.shape != (n_theta, 3, 3):
        raise ValueError(f"metric_at_theta must have shape ({n_theta}, 3, 3), got {metric_at_theta.shape}")

    metric = np.empty((ny, nx, n_theta, 3, 3))
    metric[:, :, :] = metric_at_theta[None, None, :, :, :]

    h_theta = 2.0 * np.pi / n_theta
    resolutions = (resolution, resolution, h_theta)
    periodic = (False, False, True)
    origin = (origin_xy[0], origin_xy[1], 0.0)

    solver = fsm.Solver(
        (ny, nx, n_theta), resolutions, periodic, metric, speed_fn,
        origin=origin, radius=radius, obstacle_fill=obstacle_fill,
    )
    return solver, thetas


def solve(
        ny: int,
        nx: int,
        resolution: float,
        n_theta: int,
        speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
        source: tuple[int, int, int],
        origin_xy=(0.0, 0.0),
        metric_at_theta: np.ndarray | None = None,
        xi_lateral: float = 0.4,
        xi_turn: float = 0.9,
        radius: int = 2,
        n_iters: int = 300,
        tol: float = 1e-5,
        obstacle_fill: float = fsm.OBSTACLE_FILL,
) -> np.ndarray:
    """Convenience wrapper for a single solve -- builds a fresh solver via
    `build_solver` (see its docstring for the args) and calls
    `.solve(source, n_iters, tol)` on it. Prefer `build_solver` +
    `Solver.solve` directly when solving many sources against the same
    grid/metric/speed field -- e.g. an all-pairs field -- to avoid
    recompiling the sweep once per source.

    Returns `field`, shape `(ny, nx, n_theta)`: `field[i, j, k]` is the
    arrival time from state `source` (grid indices `(sy, sx, sk)`) to
    state `(x[j], y[i], theta[k])`, `theta[k] = 2*pi*k/n_theta`.
    """
    solver, _thetas = build_solver(
        ny, nx, resolution, n_theta, speed_fn, origin_xy, metric_at_theta,
        xi_lateral, xi_turn, radius, obstacle_fill,
    )
    return solver.solve(source, n_iters, tol)
