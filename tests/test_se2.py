import numpy as np

from eikonax.se2 import solve


def test_isotropic_matches_euclidean_distance_in_free_space():
    """xi=1.0 (isotropic): translation speed is direction-independent, so
    the BEST-case arrival time at a target position -- minimized over final
    heading, since reaching a SPECIFIC final heading still costs real
    rotate_cost even at xi=1 -- should approximate straight-line Euclidean
    distance / speed, up to grid-discretization error from the
    candidate-direction sampling."""
    ny, nx, n_theta = 21, 21, 8
    mask = np.zeros((ny, nx), dtype=bool)
    resolution = 0.1
    sy, sx = 10, 10

    field = solve(mask, resolution, n_theta, xi=1.0, source=(sy, sx, 0))

    for (ty, tx) in [(10, 15), (15, 10), (5, 5), (15, 15)]:
        expected = resolution * np.hypot(ty - sy, tx - sx)
        got = float(np.min(field[ty, tx, :]))
        assert abs(got - expected) / expected < 0.15, (
            f"({ty},{tx}): got {got}, expected ~{expected} (euclidean)")


def test_anisotropy_favors_the_facing_direction():
    """xi<1: a target straight ahead of the source's own heading should be
    reached faster than an equidistant target off to the side."""
    ny, nx, n_theta = 21, 21, 8
    mask = np.zeros((ny, nx), dtype=bool)
    resolution = 0.1
    sy, sx = 10, 10
    sk = 0  # heading = theta[0] = 0 rad, i.e. facing +x (increasing column)

    field = solve(mask, resolution, n_theta, xi=0.4, source=(sy, sx, sk))

    ahead = field[sy, sx + 5, sk]        # same row, +x -- straight ahead
    lateral = field[sy + 5, sx, sk]      # same column, +y -- directly sideways
    assert ahead < lateral, f"ahead={ahead} should be < lateral={lateral} for xi<1"


def test_obstacle_forces_a_detour():
    """A wall directly between source and target should make the arrival
    time strictly greater than the unobstructed straight-line distance."""
    ny, nx, n_theta = 21, 21, 8
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:15, 10] = True  # a wall spanning most of the grid, leaving a gap below
    resolution = 0.1
    sy, sx = 10, 5
    ty, tx = 10, 15

    field = solve(mask, resolution, n_theta, xi=1.0, source=(sy, sx, 0))
    got = float(field[ty, tx, 0])
    straight_line = resolution * np.hypot(ty - sy, tx - sx)
    assert got > straight_line * 1.2, f"expected a real detour, got {got} vs straight {straight_line}"
