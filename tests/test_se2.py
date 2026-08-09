import jax.numpy as jnp
import numpy as np

from eikonax.se2 import build_solver, mask_speed_fn, solve


def _free_speed_fn(coords):
    return jnp.ones(coords.shape[:-1])


def test_isotropic_matches_euclidean_distance_in_free_space():
    """xi_lateral=xi_turn=1.0 (isotropic): translation speed is
    direction-independent, so the BEST-case arrival time at a target
    position -- minimized over final heading, since reaching a SPECIFIC
    final heading still costs real rotate_cost even at xi_turn=1 --
    should approximate straight-line Euclidean distance / speed, up to
    grid-discretization error."""
    ny, nx, n_theta = 21, 21, 8
    resolution = 0.1
    sy, sx = 10, 10

    field = solve(
        ny, nx, resolution, n_theta, _free_speed_fn, source=(sy, sx, 0),
        xi_lateral=1.0, xi_turn=1.0, radius=3,
    )

    for (ty, tx) in [(10, 15), (15, 10), (5, 5), (15, 15)]:
        expected = resolution * np.hypot(ty - sy, tx - sx)
        got = float(np.min(field[ty, tx, :]))
        assert abs(got - expected) / expected < 0.15, (
            f"({ty},{tx}): got {got}, expected ~{expected} (euclidean)")


def test_anisotropy_favors_the_facing_direction():
    """xi_lateral<1: a target straight ahead of the source's own heading
    should be reached faster than an equidistant target off to the side.
    Tested at a NONZERO start heading (not theta=0) -- at theta=0
    specifically, sin(0)=0 collapses the forward/lateral rotation to the
    identity matrix regardless of whether the (dy, dx) axis order is
    right, so a theta=0-only test can't actually catch that this project's
    own dev got that axis order backwards once (see
    default_metric_at_theta's own docstring)."""
    ny, nx, n_theta = 21, 21, 8
    resolution = 0.1
    sy, sx = 10, 10
    sk = 2  # heading = theta[2] = pi/2, i.e. facing +y (increasing row)

    field = solve(
        ny, nx, resolution, n_theta, _free_speed_fn, source=(sy, sx, sk),
        xi_lateral=0.4, xi_turn=0.4, radius=3,
    )

    ahead = field[sy + 5, sx, sk]    # +y -- straight ahead of a +y heading
    lateral = field[sy, sx + 5, sk]  # +x -- directly sideways of a +y heading
    assert ahead < lateral, f"ahead={ahead} should be < lateral={lateral} for xi_lateral<1"


def test_obstacle_forces_a_detour():
    ny, nx, n_theta = 21, 21, 8
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:15, 10] = True  # a wall spanning most of the grid, leaving a gap below
    resolution = 0.1
    sy, sx = 10, 5
    ty, tx = 10, 15

    speed_fn = mask_speed_fn(mask, resolution)
    field = solve(
        ny, nx, resolution, n_theta, speed_fn, source=(sy, sx, 0),
        xi_lateral=1.0, xi_turn=1.0, radius=3,
    )
    got = float(field[ty, tx, 0])
    straight_line = resolution * np.hypot(ty - sy, tx - sx)
    assert got > straight_line * 1.2, f"expected a real detour, got {got} vs straight {straight_line}"


def test_combined_moves_make_turning_worth_it_even_with_fixed_start_and_goal_heading():
    """The whole point of the fsm.py refactor: with only "translate-only"
    xor "rotate-only" candidate moves (this package's earlier design),
    raising xi_turn NEVER made turning worth it for a same-start-and-goal
    -heading query, no matter how cheap turning got -- a diagonal zig-zag
    that never touches theta at all was always at least as good, since
    genuine turning always paid a real, non-amortizable "there and back"
    cost on top of it. `fsm.py`'s combined offsets (a single candidate
    move can translate AND rotate at once) remove that floor: confirmed
    directly here by comparing a decoupled, cheap-turn metric against a
    tied (old-style) one on the SAME fixed-start=goal-heading query --
    the decoupled case should come out meaningfully cheaper."""
    ny, nx, n_theta = 41, 21, 8
    resolution = 0.1
    sy, sx, sk = 10, 10, 0
    ty, tx = sy + 25, sx  # pure lateral (relative to the start heading) target

    tied = solve(
        ny, nx, resolution, n_theta, _free_speed_fn, source=(sy, sx, sk),
        xi_lateral=0.3, xi_turn=0.3, radius=2,
    )
    decoupled = solve(
        ny, nx, resolution, n_theta, _free_speed_fn, source=(sy, sx, sk),
        xi_lateral=0.3, xi_turn=0.95, radius=2,
    )

    cost_tied = float(tied[ty, tx, sk])
    cost_decoupled = float(decoupled[ty, tx, sk])
    assert cost_decoupled < 0.75 * cost_tied, (
        f"decoupled (cheap-turn) cost {cost_decoupled} should be meaningfully < tied cost {cost_tied}")


def test_custom_metric_matrix_is_honored():
    """build_solver's metric_at_theta lets a caller substitute a fully
    custom metric -- confirmed here with a trivial custom isotropic
    metric (scaled identity, no heading dependence at all) matching a
    hand-picked scalar speed, independent of default_metric_at_theta."""
    ny, nx, n_theta = 21, 21, 8
    resolution = 0.1
    scale = 2.0
    custom = np.tile(scale * np.eye(3)[None, :, :], (n_theta, 1, 1))

    solver, _thetas = build_solver(
        ny, nx, resolution, n_theta, _free_speed_fn, metric_at_theta=custom, radius=3,
    )
    field = solver.solve((10, 10, 0))
    got = float(np.min(field[15, 15, :]))
    expected = np.sqrt(scale) * resolution * np.hypot(5, 5)
    assert abs(got - expected) / expected < 0.15
