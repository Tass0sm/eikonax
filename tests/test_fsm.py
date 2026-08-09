import jax.numpy as jnp
import numpy as np

from eikonax.fsm import Solver


def _identity_metric(grid_shape, n):
    metric = np.empty((*grid_shape, n, n))
    metric[..., :, :] = np.eye(n)
    return metric


def _free_speed_fn(coords):
    return jnp.ones(coords.shape[:-1])


def test_isotropic_matches_euclidean_distance_in_free_space():
    ny, nx = 21, 21
    resolution = 0.1
    metric = _identity_metric((ny, nx), 2)

    solver = Solver((ny, nx), (resolution, resolution), (False, False), metric, _free_speed_fn, radius=3)
    field = solver.solve((10, 10))

    for (r, c) in [(10, 15), (15, 10), (5, 5), (15, 15)]:
        expected = resolution * np.hypot(r - 10, c - 10)
        got = float(field[r, c])
        assert abs(got - expected) / expected < 0.1, f"({r},{c}): got {got}, expected ~{expected}"


def test_obstacle_speed_fn_forces_a_detour():
    """speed_fn returning 0 along a wall (instead of a mask) should force
    the same kind of detour a binary obstacle mask would.

    `radius=3` and a 1-cell-thick wall are deliberate: this is also the
    regression test for a real tunneling bug this package's own dev
    caught -- a wide-radius offset can jump clean over a sub-radius-thick
    obstacle without ever landing on it, if only the move's two ENDPOINTS
    are checked for speed. Fixed by checking every lattice point the move
    passes through (see fsm._segment_lattice_points); this test failed
    before that fix (arrival time came out exactly equal to the
    unobstructed straight-line distance -- the wall was invisible to a
    3-cell jump)."""
    ny, nx = 21, 21
    resolution = 0.1
    metric = _identity_metric((ny, nx), 2)

    def speed_fn(coords):
        x = coords[..., 1]  # column axis
        y = coords[..., 0]  # row axis
        in_wall = (jnp.abs(x - 1.0) < resolution / 2) & (y < 1.4)
        return jnp.where(in_wall, 0.0, 1.0)

    solver = Solver((ny, nx), (resolution, resolution), (False, False), metric, speed_fn, radius=3)
    field = solver.solve((10, 5))
    got = float(field[10, 15])
    straight_line = resolution * 10
    assert got > straight_line * 1.2, f"expected a real detour, got {got} vs straight {straight_line}"


def test_anisotropic_metric_favors_aligned_direction():
    """A metric that penalizes axis-1 motion relative to axis-0 should
    make reaching an axis-0-aligned target cheaper than an equidistant
    axis-1-aligned one."""
    ny, nx = 21, 21
    resolution = 0.1
    metric = np.zeros((ny, nx, 2, 2))
    metric[..., 0, 0] = 1.0
    metric[..., 1, 1] = 1.0 / 0.3 ** 2  # axis-1 motion costs 1/0.3

    solver = Solver((ny, nx), (resolution, resolution), (False, False), metric, _free_speed_fn, radius=3)
    field = solver.solve((10, 10))
    cheap_axis = float(field[15, 10])   # axis-0 (row) displacement
    expensive_axis = float(field[10, 15])  # axis-1 (col) displacement
    assert cheap_axis < expensive_axis


def test_speed_fn_scales_arrival_time():
    """Halving the speed everywhere should roughly double the arrival
    time -- a sanity check that the speed field is actually used, not
    just accepted and ignored."""
    ny, nx = 21, 21
    resolution = 0.1
    metric = _identity_metric((ny, nx), 2)

    def half_speed_fn(coords):
        return 0.5 * jnp.ones(coords.shape[:-1])

    fast = Solver((ny, nx), (resolution, resolution), (False, False), metric, _free_speed_fn, radius=3)
    slow = Solver((ny, nx), (resolution, resolution), (False, False), metric, half_speed_fn, radius=3)
    fast_cost = float(fast.solve((10, 10))[15, 15])
    slow_cost = float(slow.solve((10, 10))[15, 15])
    assert abs(slow_cost / fast_cost - 2.0) < 0.05


def test_source_inside_obstacle_raises():
    ny, nx = 11, 11
    resolution = 0.1
    metric = _identity_metric((ny, nx), 2)

    def speed_fn(coords):
        r = coords[..., 0]
        return jnp.where(r < 0.15, 0.0, 1.0)

    solver = Solver((ny, nx), (resolution, resolution), (False, False), metric, speed_fn, radius=3)
    try:
        solver.solve((0, 0))
        assert False, "expected a ValueError for a zero-speed source"
    except ValueError:
        pass
