import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eikonax.geometry import box_sdf, identity_fk, spheres_speed_fn

UNIT = (np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 1.0, 1.0]]))  # centre, half


def test_box_sdf_inside_is_negative_depth_below_nearest_face():
    c, h = UNIT
    assert float(box_sdf(np.array([0.0, 0.0, 0.0]), c, h)) == pytest.approx(-1.0, abs=1e-5)
    # 0.2 from the +x face, deeper on the other axes -> nearest face wins.
    assert float(box_sdf(np.array([0.8, 0.0, 0.0]), c, h)) == pytest.approx(-0.2, abs=1e-5)


def test_box_sdf_outside_is_euclidean_gap_to_surface():
    c, h = UNIT
    assert float(box_sdf(np.array([3.0, 0.0, 0.0]), c, h)) == pytest.approx(2.0)
    assert float(box_sdf(np.array([2.0, 2.0, 0.0]), c, h)) == pytest.approx(np.sqrt(2.0), abs=1e-5)


def test_box_sdf_union_takes_the_nearer_box():
    centers = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    half = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert float(box_sdf(np.array([3.5, 0.0, 0.0]), centers, half)) == pytest.approx(0.5, abs=1e-5)


def test_box_sdf_gradient_is_finite_inside_and_on_the_surface():
    c, h = UNIT
    g = jax.grad(lambda p: box_sdf(p, c, h))
    for point in (np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])):
        assert np.all(np.isfinite(np.asarray(g(point)))), point


def test_box_sdf_batches():
    c, h = UNIT
    pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert np.allclose(np.asarray(box_sdf(pts, c, h)), [-1.0, 2.0], atol=1e-5)


def test_spheres_speed_floors_inside_and_saturates_far_away():
    c, h = UNIT
    speed_fn = spheres_speed_fn(identity_fk, (0.0,), c, h, margin=0.5, min_speed=0.1)
    assert float(speed_fn(jnp.zeros(3))) == pytest.approx(0.1)
    assert float(speed_fn(jnp.array([10.0, 0.0, 0.0]))) == pytest.approx(1.0)


def test_spheres_speed_ramps_monotonically_across_the_margin():
    c, h = UNIT
    speed_fn = spheres_speed_fn(identity_fk, (0.0,), c, h, margin=0.4, min_speed=0.1)
    xs = 1.0 + np.linspace(0.0, 0.4, 9)  # clearance 0 -> margin along +x
    speeds = np.array([float(speed_fn(jnp.array([x, 0.0, 0.0]))) for x in xs])
    assert np.all(np.diff(speeds) >= -1e-6)
    assert speeds[0] == pytest.approx(0.1) and speeds[-1] == pytest.approx(1.0)


def test_spheres_speed_accounts_for_the_sphere_radius():
    c, h = UNIT
    r = 0.3
    speed_fn = spheres_speed_fn(identity_fk, (r,), c, h, margin=0.5, min_speed=0.1)
    # A point whose CENTRE clears the box by 0.2 but whose 0.3-radius sphere
    # still overlaps it -> floored.
    assert float(speed_fn(jnp.array([1.2, 0.0, 0.0]))) == pytest.approx(0.1)


def test_spheres_speed_is_jittable_and_grad_finite():
    c, h = UNIT
    speed_fn = spheres_speed_fn(identity_fk, (0.0,), c, h, margin=0.5)
    jitted = jax.jit(speed_fn)
    grad = jax.grad(lambda p: speed_fn(p))
    for point in (np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.25, 0.1, 0.0])):
        assert np.isfinite(float(jitted(point)))
        assert np.all(np.isfinite(np.asarray(grad(point)))), point


def test_spheres_speed_frame_offset_shifts_the_scene():
    c, h = UNIT
    offset = np.array([10.0, 0.0, 0.0])
    speed_fn = spheres_speed_fn(identity_fk, (0.0,), c, h, margin=0.5, frame_offset=offset)
    # The obstacle now sits around x = 10 in the caller's frame.
    assert float(speed_fn(jnp.array([10.0, 0.0, 0.0]))) == pytest.approx(0.1)
    assert float(speed_fn(jnp.array([0.0, 0.0, 0.0]))) == pytest.approx(1.0)


def test_identity_fk_shape():
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = np.asarray(identity_fk(coords))
    assert out.shape == (2, 1, 3)
    assert np.allclose(out[:, 0, :], coords)
