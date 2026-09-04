import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eikonax import se2
from eikonax.backends import metric_net
from eikonax.domains import BoxDomain, dual_norm, se2_domain, se2_metric_inv_fn
from eikonax.strategies import ntfields
from eikonax.strategies.ntfields import td_ntfields

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
