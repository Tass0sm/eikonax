"""The uniform strategy layer: `strategies.<name>.solve(domain, ...)` and the
`eikonax.scripts.solve` CLI built on top of it."""

import jax.numpy as jnp
import numpy as np
import pytest

from eikonax import se2
from eikonax.domains import se2_domain
from eikonax.scenarios import SCENARIOS
from eikonax.scripts import solve as solve_cli
from eikonax.strategies import STRATEGIES, fsm, ntfields


def _free_speed_fn(coords):
    return jnp.ones(coords.shape[:-1])


def test_registry():
    assert set(STRATEGIES) == {"fsm", "ntfields"}
    assert STRATEGIES["fsm"] is fsm and STRATEGIES["ntfields"] is ntfields


def test_fsm_solve_matches_se2_solve_node_for_node():
    """`strategies.fsm.solve` drives the same numerics `se2.build_solver`
    does -- a domain-built solver and the SE(2) one must agree."""
    domain = se2_domain(_free_speed_fn, ny=21, nx=21, resolution=0.1, n_theta=8,
                        xi_lateral=1.0, xi_turn=1.0)
    got = fsm.solve(domain, source=(10, 10, 0), radius=3)
    ref = se2.solve(21, 21, 0.1, 8, _free_speed_fn, source=(10, 10, 0),
                    xi_lateral=1.0, xi_turn=1.0, radius=3)
    assert got.shape == (21, 21, 8)
    assert np.allclose(got, ref, atol=1e-5)

    # ... and free-space arrival time is Euclidean distance
    for (ty, tx) in [(10, 15), (15, 10), (5, 5)]:
        expected = 0.1 * np.hypot(ty - 10, tx - 10)
        assert abs(float(np.min(got[ty, tx, :])) - expected) / expected < 0.15


def test_fsm_solve_default_source_is_the_grid_centre():
    domain = se2_domain(_free_speed_fn, ny=11, nx=11, n_theta=4)
    field = fsm.solve(domain, radius=2)
    assert field.shape == (11, 11, 4)
    assert field[5, 5, 2] == pytest.approx(float(np.min(field)), abs=1e-6)


def test_fsm_all_pairs_shape_and_progress():
    domain = se2_domain(_free_speed_fn, ny=5, nx=5, n_theta=4)
    steps = []
    field = fsm.solve(domain, radius=1, all_pairs=True,
                      progress_fn=lambda i, m: steps.append(i))
    assert field.shape == (5, 5, 4, 5, 5, 4)
    assert np.all(np.isfinite(field))
    assert steps and steps[-1] == 5 * 5 * 4  # every free source visited


def test_scenario_wall_forces_a_detour_through_the_strategy():
    speed_fn = SCENARIOS["wall"](wall_x=1.0, wall_y_max=1.4, thickness=0.05)
    domain = se2_domain(speed_fn, ny=21, nx=21, resolution=0.1, n_theta=8,
                        xi_lateral=1.0, xi_turn=1.0)
    field = fsm.solve(domain, source=(10, 5, 0), radius=3)
    straight = 0.1 * 10
    assert float(field[10, 15, 0]) > 1.2 * straight


def test_ntfields_solve_returns_a_model_and_reduces_the_residual():
    domain = se2_domain(_free_speed_fn, ny=11, nx=11, n_theta=4)
    history = []
    model = ntfields.solve(
        domain, progress_fn=lambda e, m: history.append(m["eikonal"]),
        hidden=32, n_blocks=1, group=8, batch_size=64, batches_per_epoch=2,
        epochs=40, lr=2e-3, log_every=1,
    )
    assert isinstance(model, ntfields.Model)
    assert np.mean(history[-5:]) < np.mean(history[:5])


# ------------------------------------------------------------------------- CLI


def test_cli_fsm_writes_a_finite_field(tmp_path):
    out = tmp_path / "f.npz"
    rc = solve_cli.main([
        "--strategy", "fsm", "--domain", "se2", "--scenario", "wall",
        "--ny", "15", "--nx", "15", "--n-theta", "4", "--radius", "3",
        "--source", "3", "3", "0", "--out", str(out),
    ])
    assert rc == 0
    field = np.load(out)["field"]
    assert field.shape == (15, 15, 4) and np.all(np.isfinite(field))


def test_cli_ntfields_writes_a_finite_field(tmp_path):
    out = tmp_path / "n.npz"
    rc = solve_cli.main([
        "--strategy", "ntfields", "--domain", "se2", "--scenario", "free",
        "--ny", "9", "--nx", "9", "--n-theta", "4",
        "--epochs", "2", "--batches-per-epoch", "1", "--batch-size", "64",
        "--hidden", "32", "--n-blocks", "1", "--group", "8", "--out", str(out),
    ])
    assert rc == 0
    field = np.load(out)["field"]
    assert field.shape == (9, 9, 4) and np.all(np.isfinite(field))


def test_cli_help_lists_strategy_specific_flags(capsys):
    rc = solve_cli.main(["--strategy", "ntfields", "--help"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "--epochs" in text and "--no-rollback" in text
