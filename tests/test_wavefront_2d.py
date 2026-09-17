"""The wavefront strategy in 2-D: cone splats grown around obstacles.

The reference is the exact travel time around the scenarios' rectangles (a
visibility graph), as in `test_huygens.py` -- `fsm`'s own wide-stencil
error on these grids is larger than the field's.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from eikonax.domains import plane_domain
from eikonax.scenarios import SCENARIOS
from eikonax.scripts import solve as solve_cli
from eikonax.strategies import wavefront
from eikonax.strategies.wavefront import baselines

SOURCE_IDX = (20, 5)
SOURCE = np.array([2.0, 0.5])
WALL_CORNER = np.array([2.8, 2.025])


def _domain(scenario):
    return plane_domain(SCENARIOS[scenario](), ny=41, nx=41, resolution=0.1)


def _score(model, scenario, n_eval=81):
    domain = model.domain
    nodes = np.asarray(domain.from_normalized(domain.grid((n_eval, n_eval))))
    exact = baselines.exact_rects(SCENARIOS[scenario]().rects, model.source, nodes)
    free = np.isfinite(exact) & (np.asarray(domain.speed_fn(jnp.asarray(nodes))) > 0)
    return model.time(nodes)[free] - exact[free], nodes[free]


def _sources(model):
    """The splats' virtual sources, rounded and de-duplicated."""
    p = {k: np.asarray(v) for k, v in model.params.items()}
    return np.unique(np.round(p["B"] - p["rho"][:, None] * p["p"], 2), axis=0)


def test_uniform_speed_is_one_cone_however_many_splats():
    """No obstacle: every splat inherits the source's own wave, so the blend
    is that single cone -- exact, whatever the layout."""
    model = wavefront.solve(_domain("free"), source=SOURCE_IDX)
    err, _ = _score(model, "free")
    assert model.num_splats < 50
    assert np.abs(err).max() < 1e-6
    assert len(_sources(model)) == 1
    np.testing.assert_allclose(_sources(model)[0], SOURCE, atol=1e-6)


@pytest.mark.parametrize("scenario", ["wall", "gap"])
def test_obstacle_scenes_are_accurate_and_fully_covered(scenario):
    model = wavefront.solve(_domain(scenario), source=SOURCE_IDX)
    err, _ = _score(model, scenario)
    assert np.all(np.isfinite(err))          # every free point is inside some window
    assert np.sqrt(np.mean(err ** 2)) < 0.02
    assert np.abs(err).max() < 0.06


def test_a_diffraction_source_appears_at_the_wall_corner():
    """The growth finds the corner by itself: a splat whose virtual source
    is blocked starts a fresh circular front, and it lands on the corner."""
    model = wavefront.solve(_domain("wall"), source=SOURCE_IDX)
    sources = _sources(model)
    assert len(sources) >= 2
    assert np.min(np.linalg.norm(sources - WALL_CORNER, axis=1)) < 0.05
    assert np.min(np.linalg.norm(sources - SOURCE, axis=1)) < 1e-6


def test_the_field_goes_around_the_wall_not_through_it():
    model = wavefront.solve(_domain("wall"), source=SOURCE_IDX)
    behind = np.array([[1.0, 3.0]])          # deep in the shadow
    around = float(np.linalg.norm(WALL_CORNER - SOURCE) + np.linalg.norm(behind[0] - WALL_CORNER))
    straight = float(np.linalg.norm(behind[0] - SOURCE))
    assert straight + 0.5 < around           # the detour is much longer -- a real test
    assert abs(float(model.time(behind)[0]) - around) < 0.05


def test_two_dimensions_needs_a_clearance_function():
    bare = plane_domain(lambda coords: jnp.ones(coords.shape[:-1]), ny=9, nx=9)
    assert bare.clearance_fn is None
    with pytest.raises(ValueError, match="clearance"):
        wavefront.solve(bare, source=(4, 4))


def test_refinement_is_not_available_in_2d():
    with pytest.raises(NotImplementedError):
        wavefront.solve(_domain("free"), source=SOURCE_IDX, train_steps=10)


def test_gradient_is_a_unit_covector_away_from_the_source():
    model = wavefront.solve(_domain("free"), source=SOURCE_IDX)
    X = np.array([[1.0, 1.5], [3.0, 2.5], [2.5, 3.5]])
    g = model.gradient(X)
    np.testing.assert_allclose(np.linalg.norm(g, axis=1), 1.0, rtol=1e-6)
    directions = (X - SOURCE) / np.linalg.norm(X - SOURCE, axis=1, keepdims=True)
    np.testing.assert_allclose(g, directions, atol=1e-6)


def test_cli_wavefront_writes_a_plane_field(tmp_path):
    out = tmp_path / "w2.npz"
    rc = solve_cli.main([
        "--strategy", "wavefront", "--domain", "plane", "--scenario", "wall",
        "--ny", "21", "--nx", "21", "--resolution", "0.2",
        "--source", "10", "2", "--min-window", "0.05", "--out", str(out),
    ])
    assert rc == 0
    field = np.load(out)["field"]
    assert field.shape == (21, 21)
    assert np.all(np.isfinite(field[np.asarray(field) < 1e3]))
