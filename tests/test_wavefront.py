"""The wavefront strategy: chained local-model splats on a 1-D line."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eikonax.domains import BoxDomain, line_domain
from eikonax.scenarios import SCENARIOS
from eikonax.scripts import solve as solve_cli
from eikonax.strategies import wavefront
from eikonax.strategies.wavefront import baselines, train


def _slow_domain():
    return line_domain(SCENARIOS["slow"](center_y=2.0, radius=0.3, depth=0.7))


def _max_error(model, n_eval=2000):
    X = model.domain.grid((n_eval,))
    T_true = baselines.exact_1d(model.domain, float(model.wavefront.source[0]), X)
    return float(np.max(np.abs(model.wavefront.evaluate(model.params, X) - T_true))), X, T_true


def test_uniform_speed_is_exact_with_a_handful_of_splats():
    """Constant speed: every local model is exact, the chain takes maximal
    steps, and the blend reproduces |x - s| to float32 precision."""
    model = wavefront.solve(line_domain(SCENARIOS["free"]()))
    err, _, _ = _max_error(model)
    assert model.num_splats <= 5
    assert err < 1e-5


def test_slow_patch_converges_with_tol_and_beats_scalar_splats():
    errors, counts = [], []
    for tol in (1e-2, 1e-3, 1e-4):
        model = wavefront.solve(_slow_domain(), tol=tol)
        err, X, T_true = _max_error(model)
        errors.append(err)
        counts.append(model.num_splats)
        # scalar weights on the SAME windows, even oracle-fitted, are far off
        pu0 = baselines.pu0_fit(model.wavefront, model.params, X, T_true)
        assert np.max(np.abs(pu0 - T_true)) > 100 * err
    assert counts[0] < counts[1] < counts[2]
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] < 5e-4


def test_ridge_mother_sum_breaks_at_the_handoff():
    """The literal ridge-shaped-mother sum cannot represent the field even
    with oracle weights: the atoms vanish at their own centres."""
    model = wavefront.solve(_slow_domain(), tol=1e-3)
    _, X, T_true = _max_error(model)
    ridge = baselines.ridge_fit(model.wavefront, model.params, X, T_true)
    assert np.max(np.abs(ridge - T_true)) > 0.1


def test_source_value_gradient_and_coverage():
    domain = _slow_domain()
    model = wavefront.solve(domain, source=(40,), tol=1e-3)
    field, params = model.wavefront, model.params
    assert float(field.value(params, field.source)) == 0.0
    assert float(field.source_leak(params)) == 0.0

    X = domain.grid((1000,))
    _, _, _, cover = jax.vmap(field.stats, in_axes=(None, 0))(params, X)
    assert float(jnp.min(cover)) > 0.3

    # |dT/dx| = 1/speed, pointing away from the source (the blend's slope
    # is less accurate than its value: ~0.3% where the speed curves at tol 1e-3)
    x = np.linspace(0.2, 4.8, 50)[:, None]
    x = x[np.abs(x[:, 0] - model.source[0]) > 0.05]
    g = model.gradient(x)[:, 0]
    speed = np.asarray(domain._speed_fn(jnp.asarray(x)))
    np.testing.assert_allclose(np.abs(g), 1.0 / speed, rtol=5e-3)
    np.testing.assert_array_equal(np.sign(g), np.sign(x[:, 0] - model.source[0]))


def test_grid_field_matches_the_domain_grid():
    domain = _slow_domain()
    model = wavefront.solve(domain)
    field = model.grid_field()
    assert field.shape == domain.grid_shape
    assert field[100] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.isfinite(field))


def test_refinement_trains_every_parameter_and_lowers_its_loss():
    domain = _slow_domain()
    chained = wavefront.solve(domain, tol=1e-2)
    logged = []
    trained = wavefront.solve(domain, tol=1e-2, train_steps=200, batch_size=128,
                              progress_fn=lambda i, m: logged.append(m))
    assert logged and set(logged[-1]) >= {"loss", "residual", "consistency", "coverage", "source"}

    X = domain.grid((1000,))
    before = train.total(chained.cfg, train.losses(chained.wavefront, chained.params, X))
    after = train.total(trained.cfg, train.losses(trained.wavefront, trained.params, X))
    assert float(after) < float(before)
    for name in ("B", "log_R", "c", "H", "src_log_R", "src_g"):
        assert not np.allclose(chained.params[name], trained.params[name]), name
    assert np.all(np.isfinite(trained.grid_field()))


def test_value_temperature_blend_is_finite_and_close():
    model = wavefront.solve(_slow_domain(), tol=1e-3, value_temperature=0.05)
    err, _, _ = _max_error(model)
    assert np.isfinite(err) and err < 1e-2


def test_chain_is_1d_only():
    domain = BoxDomain((0.0, 0.0), (1.0, 1.0), (False, False),
                       lambda c: jnp.ones(c.shape[:-1]), grid_shape=(5, 5))
    with pytest.raises(NotImplementedError):
        wavefront.solve(domain)


def test_cli_wavefront_writes_a_line_field(tmp_path):
    out = tmp_path / "w.npz"
    rc = solve_cli.main([
        "--strategy", "wavefront", "--domain", "line", "--scenario", "slow",
        "--n", "101", "--resolution", "0.05", "--center-y", "2.0",
        "--tol", "1e-3", "--out", str(out),
    ])
    assert rc == 0
    field = np.load(out)["field"]
    assert field.shape == (101,) and np.all(np.isfinite(field))
