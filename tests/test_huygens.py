"""`strategies.huygens`: a min over ray wavelets grown from the source.

The reference for obstacle scenes is the EXACT travel time for the
scenarios' rectangular walls (a visibility graph over their corners), not
`fsm`, whose wide-stencil error (~0.014 RMS on these grids, not shrinking
with resolution) is larger than the field's own.
"""

import heapq

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eikonax.domains import plane_domain
from eikonax.scenarios import SCENARIOS
from eikonax.scripts import solve as solve_cli
from eikonax.strategies import STRATEGIES, huygens

SOURCE_IDX = (20, 5)
SOURCE = np.array([2.0, 0.5])
HALF = 0.025  # scenarios' default wall half-thickness
WALLS = {
    "free": [],
    "wall": [(-10.0, 2.8, 2.0 - HALF, 2.0 + HALF)],
    "gap": [(-10.0, 1.7, 2.0 - HALF, 2.0 + HALF), (2.3, 10.0, 2.0 - HALF, 2.0 + HALF)],
}
FAST = dict(candidates=256, batch_size=512, refine_steps=60)


def _blocked(p, q, rects, tol=1e-6):
    """Does segment p->q cross the open interior of a `(y0, y1, x0, x1)` rectangle (Liang-Barsky)?"""
    d = q - p
    for y0, y1, x0, x1 in rects:
        t0, t1, inside = 0.0, 1.0, True
        for pk, qk in ((-d[0], p[0] - y0 - tol), (d[0], y1 - tol - p[0]),
                       (-d[1], p[1] - x0 - tol), (d[1], x1 - tol - p[1])):
            if pk == 0:
                inside &= qk >= 0
            elif pk < 0:
                t0 = max(t0, qk / pk)
            else:
                t1 = min(t1, qk / pk)
        if inside and t0 < t1:
            return True
    return False


def exact_travel_time(rects, pts):
    """Shortest unit-speed path from `SOURCE` around rectangles: Dijkstra over corners."""
    nodes = [SOURCE] + [np.array(c) for y0, y1, x0, x1 in rects
                        for c in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))]
    dist = [np.inf] * len(nodes)
    dist[0] = 0.0
    heap = [(0.0, 0)]
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for j, node in enumerate(nodes):
            if j != i and not _blocked(nodes[i], node, rects):
                nd = d + float(np.linalg.norm(nodes[i] - node))
                if nd < dist[j]:
                    dist[j] = nd
                    heapq.heappush(heap, (nd, j))
    return np.array([min((dist[i] + np.linalg.norm(p - n) for i, n in enumerate(nodes)
                          if not _blocked(n, p, rects)), default=np.inf) for p in pts])


def _domain(scenario):
    return plane_domain(SCENARIOS[scenario](), ny=41, nx=41, resolution=0.1)


def _score(model, scenario):
    domain = model.domain
    nodes = np.asarray(domain.from_normalized(domain.grid(domain.grid_shape)))
    exact = exact_travel_time(WALLS[scenario], nodes)
    free = (np.asarray(domain.speed(domain.grid(domain.grid_shape))) > 0) & np.isfinite(exact)
    return model.grid_field().ravel()[free] - exact[free]


def test_registered():
    assert STRATEGIES["huygens"] is huygens


def test_uniform_speed_is_the_source_wavelet_alone():
    model = huygens.solve(_domain("free"), source=SOURCE_IDX, **FAST)
    assert model.num_splats == 0
    err = _score(model, "free")
    assert np.abs(err).max() < 5e-3  # only ray-length smoothing (cone_delta) remains


@pytest.mark.parametrize("scenario, max_splats", [("wall", 4), ("gap", 6)])
def test_obstacles_are_handled_by_a_few_diffraction_wavelets(scenario, max_splats):
    model = huygens.solve(_domain(scenario), source=SOURCE_IDX, **FAST)
    assert 1 <= model.num_splats <= max_splats
    err = _score(model, scenario)
    assert np.sqrt(np.mean(err ** 2)) < 0.02
    # an upper bound, up to the length smoothing accumulated over a couple of hops
    assert err.min() > -5e-3


def test_wall_wavelets_sit_at_the_wall_end():
    model = huygens.solve(_domain("wall"), source=SOURCE_IDX, **FAST)
    centres = np.asarray(model.domain.from_normalized(model.params[2]))
    assert np.all(np.linalg.norm(centres - np.array([2.8, 2.0]), axis=1) < 0.15)


def test_values_are_bellman_consistent_and_differentiable():
    domain = _domain("wall")
    cfg = huygens.make_config()
    nodes = np.asarray(domain.grid(domain.grid_shape)).reshape(41, 41, 2)
    field = huygens.HuygensField(domain, nodes[SOURCE_IDX], cfg)
    B = domain.to_normalized(np.array([[2.9, 2.0], [2.0, 3.0], [3.5, 3.5]]))
    params = field.wavelets_at(B)

    V = np.asarray(field.values(params))[:, 0]
    # c_j = min_{m != j} (c_m + ray_m(B_j)), checked against the definition directly
    rays = np.asarray(field.rays(params, B))
    for j in range(3):
        others = [0] + [m + 1 for m in range(3) if m != j]
        c_all = np.concatenate([[0.0], V])
        assert V[j] == pytest.approx(min(c_all[m] + rays[j, m] for m in others), abs=1e-4)
    assert V[0] == pytest.approx(np.hypot(0.9, 1.5), abs=2e-3)  # wall end: straight from source
    assert 2.9 < V[1] < 3.2  # behind the wall: around its end (3.09), not through it (2.5)

    # the wavelet behind the wall hangs off the one at the wall end: moving the latter moves it
    grad = jax.grad(lambda b: field.values((params[0], params[1], params[2].at[0].set(b)))[1, 0])(params[2][0])
    assert np.all(np.isfinite(np.asarray(grad))) and float(jnp.linalg.norm(grad)) > 0.1


def test_descent_reaches_the_source_around_the_wall():
    model = huygens.solve(_domain("wall"), source=SOURCE_IDX, **FAST)
    starts = np.array([[1.0, 3.5], [0.5, 2.5], [3.5, 3.5]])
    traj = model.descend(starts, step=0.02, n_steps=400)
    assert np.all(np.linalg.norm(traj[-1] - SOURCE, axis=1) < 0.1)


def test_cli_huygens_writes_field_and_wavelets(tmp_path):
    out = tmp_path / "h.npz"
    rc = solve_cli.main([
        "--strategy", "huygens", "--domain", "plane", "--scenario", "wall",
        "--ny", "21", "--nx", "21", "--resolution", "0.2", "--source", "10", "2",
        "--candidates", "128", "--batch-size", "256", "--refine-steps", "20",
        "--out", str(out),
    ])
    assert rc == 0
    data = np.load(out)
    assert data["field"].shape == (21, 21) and np.all(np.isfinite(data["field"]))
    assert data["wavelet_centres"].shape[0] == data["wavelet_times"].shape[0] >= 1
