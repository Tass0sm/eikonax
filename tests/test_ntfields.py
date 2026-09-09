import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eikonax import scenarios, se2
from eikonax.backends import metric_net
from eikonax.domains import BoxDomain, dual_norm, se2_domain, se2_metric_inv_fn
from eikonax.strategies import ntfields
from eikonax.strategies.ntfields import roadmap, td_ntfields

#: keeps the network and batches tiny -- these are smoke tests, not
#: convergence claims.
_SMALL = dict(hidden=32, n_blocks=1, group=8, batch_size=64, batches_per_epoch=1)


def _small_cfg(**kw):
    """A `cfg` namespace for calling backend / objective internals directly."""
    return ntfields.make_config(**{**_SMALL, **kw})


def _train(domain, progress_fn=None, **kw):
    return ntfields.solve(domain, progress_fn=progress_fn, **{**_SMALL, **kw})


def _free_speed_fn(coords):
    return jnp.ones(coords.shape[:-1])


def _free_box(dim=2, periodic=None):
    periodic = (False,) * dim if periodic is None else periodic
    return BoxDomain([0.0] * dim, [2.0] * dim, periodic, _free_speed_fn)


# --------------------------------------------------------------------- domains


def test_normalization_roundtrips():
    domain = _free_box()
    X = np.array([[0.0, 0.0], [2.0, 1.0], [0.5, 1.7]])
    assert np.allclose(np.asarray(domain.from_normalized(domain.to_normalized(X))), X, atol=1e-5)
    assert np.allclose(np.asarray(domain.to_normalized(X)).min(), -0.5)


def test_wrap_wraps_periodic_and_clamps_the_rest():
    domain = _free_box(periodic=(False, True))
    Xn = jnp.array([[0.9, 0.7], [-0.9, -0.7]])
    wrapped = np.asarray(domain.wrap(Xn))
    assert np.allclose(wrapped[:, 0], [0.5, -0.5])
    assert np.allclose(wrapped[:, 1], [-0.3, 0.3])


def test_se2_metric_inv_matches_se2_modules_metric():
    """`se2_metric_inv_fn` must be the inverse of the very matrix
    `se2.default_metric_at_theta` builds, at the same (dy, dx, dtheta) axis
    order -- otherwise the neural field and the swept field are solving
    two different problems."""
    thetas = np.array([0.0, 0.7, 2.9, 5.1])
    expected = np.linalg.inv(se2.default_metric_at_theta(thetas, xi_lateral=0.4, xi_turn=0.9))
    coords = np.stack([np.zeros_like(thetas), np.zeros_like(thetas), thetas], axis=-1)
    got = np.asarray(se2_metric_inv_fn(0.4, 0.9)(jnp.asarray(coords)))
    assert np.allclose(got, expected, atol=1e-5)


def test_metric_inv_is_rescaled_to_normalized_coordinates():
    """`|grad T|_x` must not depend on the coordinate normalization."""
    domain = se2_domain(_free_speed_fn, ny=21, nx=21, resolution=0.1)
    Xn = jnp.array([[0.1, -0.2, 0.3]])
    physical_inv = np.asarray(se2_metric_inv_fn(0.4, 0.9)(domain.from_normalized(Xn)))[0]
    grad_physical = np.array([1.0, -2.0, 0.5])
    grad_normalized = grad_physical * domain.span
    got = float(dual_norm(jnp.asarray(grad_normalized)[None, :], domain.metric_inv(Xn))[0])
    expected = float(np.sqrt(grad_physical @ physical_inv @ grad_physical))
    assert got == pytest.approx(expected, rel=1e-4)


def test_grid_matches_the_fsm_grid_it_mirrors():
    domain = se2_domain(_free_speed_fn, ny=5, nx=7, resolution=0.1)
    nodes = np.asarray(domain.from_normalized(domain.grid((5, 7, 4)))).reshape(5, 7, 4, 3)
    assert np.allclose(nodes[:, 0, 0, 0], 0.1 * np.arange(5))
    assert np.allclose(nodes[0, :, 0, 1], 0.1 * np.arange(7))
    assert np.allclose(nodes[0, 0, :, 2], np.linspace(0, 2 * np.pi, 4, endpoint=False))


def test_se2_domain_bakes_in_its_grid_shape():
    domain = se2_domain(_free_speed_fn, ny=9, nx=11, n_theta=6)
    assert domain.grid_shape == (9, 11, 6)
    solver = domain.fsm_solver(radius=2)
    assert solver.grid_shape == (9, 11, 6)


# --------------------------------------------------------------------- backend


def test_travel_time_is_a_symmetric_pseudometric():
    """The whole point of the quasimetric head: `T(x,x)=0` and
    `T(x0,x1)=T(x1,x0)` hold at initialization, with no training and no
    boundary condition ever fit."""
    domain = _free_box()
    cfg = _small_cfg()
    params = metric_net.init(jax.random.PRNGKey(0), domain, cfg)
    rng = np.random.default_rng(0)
    X0, X1 = domain.sample(rng, 32), domain.sample(rng, 32)

    forward = np.asarray(metric_net.travel_time(params, X0, X1, cfg))
    backward = np.asarray(metric_net.travel_time(params, X1, X0, cfg))
    diagonal = np.asarray(metric_net.travel_time(params, X0, X0, cfg))

    assert np.allclose(forward, backward, atol=1e-5)
    assert np.all(diagonal < 1e-2) and np.all(diagonal >= 0.0)
    assert np.all(forward >= 0.0) and np.all(np.isfinite(forward))


def test_travel_time_is_exactly_periodic_on_a_periodic_axis():
    """Integer Fourier rows on periodic axes make the network periodic by
    construction, not by training -- a half-turn of theta plus a full turn
    must be indistinguishable from the half-turn alone."""
    domain = se2_domain(_free_speed_fn, ny=21, nx=21, resolution=0.1)
    cfg = _small_cfg()
    params = metric_net.init(jax.random.PRNGKey(0), domain, cfg)

    X0 = jnp.array([[0.1, 0.2, 0.3]])
    X1 = jnp.array([[-0.2, 0.1, 0.25]])
    shifted = X1 + jnp.array([[0.0, 0.0, 1.0]])  # one full period in normalized units
    assert float(metric_net.travel_time(params, X0, X1, cfg)[0]) == pytest.approx(
        float(metric_net.travel_time(params, X0, shifted, cfg)[0]), rel=1e-4)


def test_the_fourier_matrix_stays_frozen():
    domain = _free_box()
    cfg = _small_cfg()
    params = metric_net.init(jax.random.PRNGKey(0), domain, cfg)
    mask = metric_net.trainable_mask(params)
    assert mask.fourier is False
    assert all(jax.tree_util.tree_leaves(mask.blocks))
    assert metric_net.num_params(params) > 0


def test_hidden_must_be_twice_the_fourier_width():
    domain = _free_box()
    with pytest.raises(ValueError):
        metric_net.init(jax.random.PRNGKey(0), domain, _small_cfg(n_freq=8))


# -------------------------------------------------------------------- strategy


def test_loss_terms_are_finite_and_the_td_mask_bites():
    domain = _free_box()
    cfg = _small_cfg(td_step=10.0)  # every pair is within one step -> L_TD fully masked
    params = metric_net.init(jax.random.PRNGKey(0), domain, cfg)
    rng = np.random.default_rng(0)
    X0, X1 = domain.sample(rng, 64), domain.sample(rng, 64)

    eikonal, td, normal, causal = td_ntfields.loss_terms(metric_net, params, X0, X1, domain, cfg)
    for term in (eikonal, td, normal, causal):
        assert np.all(np.isfinite(np.asarray(term)))
    assert np.allclose(np.asarray(td), 0.0)

    unmasked = td_ntfields.loss_terms(metric_net, params, X0, X1, domain, _small_cfg(td_step=0.03))[1]
    assert np.any(np.asarray(unmasked) > 0.0)


def test_pair_sampler_is_local_and_stays_in_the_domain():
    """Pairs must be short-range (that is what the causality curriculum
    bites on) and land inside the box on non-periodic axes."""
    domain = _free_box(periodic=(False, True))
    rng = np.random.default_rng(0)
    X0, X1 = td_ntfields.sample_pairs(domain, rng, 2000)

    assert X0.shape == (2000, 2) and X1.shape == (2000, 2)
    assert np.all(np.abs(np.asarray(X1)[:, 0]) <= 0.5)
    assert np.all(np.abs(np.asarray(X1)[:, 1]) <= 0.5)

    separation = np.linalg.norm(np.asarray(X0 - X1), axis=1)
    independent = np.linalg.norm(
        rng.uniform(-0.5, 0.5, (2000, 2)) - rng.uniform(-0.5, 0.5, (2000, 2)), axis=1)
    assert separation.mean() < independent.mean()
    assert np.mean(separation < 0.1) > 3 * np.mean(independent < 0.1)


def test_speed_remap_is_the_identity_in_free_space():
    domain = _free_box()
    cfg = _small_cfg()
    speed = np.asarray(td_ntfields.speed_star(domain, domain.sample(np.random.default_rng(0), 16), cfg))
    assert np.allclose(speed, 1.0, atol=1e-5)


def test_unknown_objective_and_backend_raise():
    domain = _free_box()
    with pytest.raises(ValueError):
        ntfields.solve(domain, objective="nope", epochs=1, **_SMALL)
    with pytest.raises(ValueError):
        ntfields.solve(domain, backend="nope", epochs=1, **_SMALL)


def test_short_training_run_reduces_the_eikonal_residual():
    """A free-space box: `T` should approach the straight-line distance, so
    the eikonal term must actually come down. Deliberately tiny -- this is
    a smoke test of the whole train loop, not a convergence claim."""
    domain = _free_box()
    history = []
    model = _train(domain, progress_fn=lambda epoch, m: history.append(m["eikonal"]),
                   epochs=40, batches_per_epoch=2, lr=2e-3, log_every=1, seed=0)

    assert len(history) == 40
    assert np.mean(history[-5:]) < np.mean(history[:5])
    assert np.all(np.isfinite(model.field((1.0, 1.0), (5, 5))))


# -------------------------------------------------------- weak supervision (PRM)


def test_roadmap_routes_around_an_obstacle():
    """A wall between two points must make the roadmap distance exceed the
    straight-line one; free space must not."""
    wall = se2_domain(scenarios.wall(wall_x=1.0, wall_y_max=1.8, thickness=0.3),
                      ny=21, nx=21, resolution=0.1, n_theta=8, xi_lateral=1.0, xi_turn=1.0)
    free = se2_domain(_free_speed_fn, ny=21, nx=21, resolution=0.1, n_theta=8,
                      xi_lateral=1.0, xi_turn=1.0)
    rm_wall = roadmap.build_roadmap(wall, n_nodes=250, k=12, segment_samples=24, seed=0)
    rm_free = roadmap.build_roadmap(free, n_nodes=250, k=12, segment_samples=24, seed=0)

    a = jnp.array([[-0.3, -0.35, 0.0]])  # left of the wall
    b = jnp.array([[-0.3, 0.35, 0.0]])   # right of the wall, straddling it
    d_wall = float(roadmap.roadmap_distance(rm_wall, wall, a, b)[0])
    d_free = float(roadmap.roadmap_distance(rm_free, free, a, b)[0])
    assert np.isfinite(d_wall) and np.isfinite(d_free)
    assert d_wall > 1.2 * d_free


def test_roadmap_distance_batches_and_tracks_fsm_in_free_space():
    free = se2_domain(_free_speed_fn, ny=21, nx=21, resolution=0.1, n_theta=8,
                      xi_lateral=1.0, xi_turn=1.0)
    rm = roadmap.build_roadmap(free, n_nodes=300, k=12, seed=0)
    rng = np.random.default_rng(0)
    X0, X1 = free.sample(rng, 128), free.sample(rng, 128)
    d = np.asarray(roadmap.roadmap_distance(rm, free, X0, X1))
    assert d.shape == (128,) and np.all(np.isfinite(d)) and np.all(d >= 0.0)

    # the roadmap distance is a (piecewise) near-upper bound on the true
    # geodesic, so it should sit around or above the straight-line metric
    # distance -- not collapse to something far smaller.
    span = np.asarray(free.span)
    straight = np.linalg.norm((np.asarray(X0) - np.asarray(X1)) * span, axis=1)
    assert np.mean(d) >= 0.8 * np.mean(straight)


def test_roadmap_can_report_a_disconnected_pair_as_inf():
    """A wall spanning the whole width splits the domain -- some node pairs
    are genuinely unreachable through the graph."""
    full_wall = se2_domain(scenarios.wall(wall_x=1.0, wall_y_max=100.0, thickness=0.5),
                           ny=21, nx=21, resolution=0.1, n_theta=4)
    rm = roadmap.build_roadmap(full_wall, n_nodes=200, k=10, segment_samples=16, seed=0)
    assert not np.all(np.isfinite(np.asarray(rm.node_dist)))


def test_training_with_roadmap_weight_runs_and_reports_the_term():
    domain = se2_domain(scenarios.wall(), ny=15, nx=15, resolution=0.1, n_theta=4)
    history = []
    ntfields.solve(
        domain, progress_fn=lambda e, m: history.append(m),
        **_SMALL, epochs=6, log_every=1,
        roadmap_weight=1e-2, roadmap_nodes=80, roadmap_k=8,
    )
    assert all("roadmap" in m and np.isfinite(m["roadmap"]) for m in history)
    assert history[0]["roadmap"] > 0.0


def test_roadmap_weight_zero_builds_no_roadmap(monkeypatch):
    monkeypatch.setattr(td_ntfields, "build_roadmap",
                        lambda *a, **k: pytest.fail("build_roadmap called with roadmap_weight=0"))
    history = []
    _train(_free_box(), progress_fn=lambda e, m: history.append(m), epochs=2, log_every=1)
    assert all(m["roadmap"] == 0.0 for m in history)


def test_model_helpers_round_trip_physical_coordinates():
    domain = se2_domain(_free_speed_fn, ny=11, nx=11, resolution=0.1)
    model = _train(domain, epochs=1)

    X0 = np.array([[0.5, 0.5, 0.0], [0.2, 0.8, 3.0]])
    X1 = np.array([[0.6, 0.4, 1.0], [0.2, 0.8, 3.0]])
    times = model.time(X0, X1)
    assert times.shape == (2,) and np.all(np.isfinite(times))
    assert times[1] < times[0]  # the identical pair is the cheaper one

    g0, g1 = model.gradient(X0, X1)
    assert g0.shape == (2, 3) and g1.shape == (2, 3)
    assert np.all(np.isfinite(model.speed(X0, X1)))

    field = model.field((0.5, 0.5, 0.0), (11, 11, 4))
    assert field.shape == (11, 11, 4)
    assert field[5, 5, 0] == pytest.approx(float(np.min(field)), abs=1e-3)  # source is the minimum


def test_train_ntfield_save_load_round_trips(tmp_path):
    from eikonax import load_ntfield, train_ntfield
    from eikonax.strategies.ntfields import ARCHITECTURE_VERSION

    out = tmp_path / "field.npz"
    obstacles = (np.array([[0.0, 0.0, 0.0]]), np.array([[0.2, 0.2, 0.2]]))
    model = train_ntfield(
        coordinate_space="workspace_xyz",
        normalization_box=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        obstacle_boxes=obstacles, margin=0.3, out=out,
        epochs=15, lr=2e-3, seed=0, **_SMALL,
    )
    assert out.exists()

    field = load_ntfield(out)
    assert field.coordinate_space == "workspace_xyz"
    assert field.architecture_version == ARCHITECTURE_VERSION
    assert np.allclose(field.lower, [-1.0, -1.0, -1.0])

    X0 = np.array([[-0.5, -0.5, -0.5], [0.4, 0.1, 0.2]])
    X1 = np.array([[0.5, 0.5, 0.5], [-0.3, 0.0, 0.1]])
    assert np.allclose(np.asarray(model.time(X0, X1)),
                       np.asarray(field.time(X0, X1)), atol=1e-5)

    # The eval primitive stays traceable.
    jitted = jax.jit(field.travel_time)
    assert np.allclose(np.asarray(jitted(X0, X1)), np.asarray(field.time(X0, X1)), atol=1e-5)

    # T(x, x) is the smooth-max floor, not fit.
    floor = 0.2 * (_SMALL["hidden"] // _SMALL["group"]) * np.sqrt(1e-6)
    assert float(np.asarray(field.time(X0[:1], X0[:1]))[0]) == pytest.approx(floor, abs=1e-4)


def test_train_ntfield_non_workspace_space_needs_explicit_fk():
    from eikonax import train_ntfield

    with pytest.raises(ValueError, match="needs an explicit fk"):
        train_ntfield(
            coordinate_space="cspace_ur5", normalization_box=((0.0,), (1.0,)),
            obstacle_boxes=(np.zeros((1, 3)), np.ones((1, 3))), margin=0.1,
        )
