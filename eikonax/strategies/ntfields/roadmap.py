"""A probabilistic roadmap (PRM) over a `domain`, and an approximate geodesic
`d_PRM(x0, x1)` read off it -- the weak-supervision prior for the TD-NTFields
objective (`td_ntfields.py`, gated by `cfg.roadmap_weight`).

`srms`'s `weak_supervision.py` gives a single-source field a coarse,
obstacle-aware cost-to-come to refine (an RRT* tree, `T = base * exp(g)`).
The field here is ALL-PAIRS and the backend is a fixed quasimetric network,
so neither the tree nor the factorization carries over. A PRM does: sample
free nodes once, connect near neighbours with collision-checked,
slowness-weighted edges, run ALL-PAIRS shortest paths on the graph, and
bridge any query pair to the graph through its `k` nearest nodes. The result
is a cheap, obstacle-aware near-upper bound on the true geodesic between
*any* two points -- used as a plain MSE anchor on `T`, outside the causal
curriculum (its whole value is anchoring the far / around-obstacle pairs the
curriculum suppresses).

Everything is in the domain's NORMALIZED coordinates (`[-0.5, 0.5]^dim`,
periodic axes wrap with period 1 -- see `eikonax.domains`). Edge/hop cost is
`metric_length(disp) / speed`, matching `fsm.Solver`'s `costs =
metric_length / effective_speed` and the physical units `T` is measured in:
`metric_length` uses `G_hat = inv(domain.metric_inv(midpoint))` (the
displacement-length metric, the inverse of the covector-length metric the
PDE side uses), and `speed` is the raw `domain.speed` (obstacles -> 0), not
`td_ntfields.speed_star`'s training remap.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ...domains import DTYPE


class Roadmap(NamedTuple):
    """Built once by `build_roadmap`, then closed over (NOT passed as a jit
    argument -- `k` must stay a Python int) by `roadmap_distance`."""

    nodes: jnp.ndarray        # (M, dim) normalized free-space nodes
    node_dist: jnp.ndarray    # (M, M) all-pairs graph shortest-path cost; inf where disconnected
    node_speed: jnp.ndarray   # (M,) domain.speed at each node
    k: int                    # neighbours per node, and per query endpoint
    min_speed: float          # speed floor / obstacle threshold


def _wrap_disp(disp, periodic):
    """Fold a displacement onto `[-0.5, 0.5]` on periodic axes (np or jnp)."""
    xp = jnp if isinstance(disp, jnp.ndarray) else np
    wrapped = xp.mod(disp + 0.5, 1.0) - 0.5
    return xp.where(periodic, wrapped, disp)


def build_roadmap(
        domain,
        *,
        n_nodes: int = 256,
        k: int = 10,
        segment_samples: int = 8,
        min_speed: float = 1e-3,
        seed: int = 0,
) -> Roadmap:
    """Sample `n_nodes` free-space nodes, connect each to its `k` nearest by
    a collision-checked slowness-weighted edge, and Floyd-Warshall the graph
    to all-pairs node distances.

    `segment_samples` points are checked along every candidate edge (an edge
    touching `speed <= min_speed` anywhere is dropped -- the straight-line
    collision check). A wall thinner than the spacing between those samples
    can be tunnelled (the same wide-move hazard `fsm._segment_lattice_points`
    guards against) -- raise `segment_samples` for thin obstacles. This is a
    coarse prior, not an exact solver: the anchor it feeds is weighted low
    and stop-gradiented.

    Cost is `O(n_nodes**3)` for the all-pairs step (~1e7 at the default);
    keep `n_nodes` <~ 512.
    """
    rng = np.random.default_rng(seed)
    dim = domain.dim
    periodic = np.asarray(domain.periodic)

    raw = np.asarray(domain.sample(rng, 8 * n_nodes), dtype=float)
    free = raw[np.asarray(domain.speed(jnp.asarray(raw, dtype=DTYPE))) > min_speed]
    if len(free) < n_nodes:  # pathologically cluttered domain -- pad to keep M fixed
        free = np.concatenate([free, free[rng.integers(0, max(len(free), 1), n_nodes)]])
    nodes = free[:n_nodes]
    M = len(nodes)
    k = min(k, M - 1)

    # kNN in wrapped normalized coordinates.
    pair_disp = _wrap_disp(nodes[:, None, :] - nodes[None, :, :], periodic)  # (M, M, dim)
    d_norm = np.sqrt(np.sum(pair_disp ** 2, axis=-1))
    np.fill_diagonal(d_norm, np.inf)
    knn = np.argpartition(d_norm, k, axis=1)[:, :k]  # (M, k)

    src = np.repeat(np.arange(M), k)
    dst = knn.reshape(-1)
    disp = _wrap_disp(nodes[dst] - nodes[src], periodic)  # (E, dim)

    ts = np.linspace(0.0, 1.0, segment_samples)
    seg = nodes[src][:, None, :] + ts[None, :, None] * disp[:, None, :]  # (E, S, dim)
    seg = np.asarray(domain.wrap(jnp.asarray(seg.reshape(-1, dim), dtype=DTYPE)))
    seg_speed = np.asarray(domain.speed(jnp.asarray(seg, dtype=DTYPE))).reshape(len(src), segment_samples)
    min_hop_speed = seg_speed.min(axis=1)  # (E,)

    mid = np.asarray(domain.wrap(jnp.asarray(nodes[src] + 0.5 * disp, dtype=DTYPE)))
    g_hat = np.linalg.inv(np.asarray(domain.metric_inv(jnp.asarray(mid, dtype=DTYPE))))  # (E, dim, dim)
    length = np.sqrt(np.clip(np.einsum("ei,eij,ej->e", disp, g_hat, disp), 0.0, None))
    weight = np.where(min_hop_speed > min_speed, length / np.maximum(min_hop_speed, min_speed), np.inf)

    W = np.full((M, M), np.inf)
    np.minimum.at(W, (src, dst), weight)
    np.minimum.at(W, (dst, src), weight)  # undirected
    np.fill_diagonal(W, 0.0)

    D = W.copy()
    for kk in range(M):
        D = np.minimum(D, D[:, kk:kk + 1] + D[kk:kk + 1, :])

    node_speed = np.asarray(domain.speed(jnp.asarray(nodes, dtype=DTYPE)))
    return Roadmap(
        nodes=jnp.asarray(nodes, dtype=DTYPE),
        node_dist=jnp.asarray(D, dtype=DTYPE),
        node_speed=jnp.asarray(node_speed, dtype=DTYPE),
        k=int(k),
        min_speed=float(min_speed),
    )


def roadmap_distance(roadmap: Roadmap, domain, X0, X1) -> jnp.ndarray:
    """Approximate geodesic between batches of normalized points `(B, dim)`,
    returned `(B,)`: bridge each endpoint to its `k` nearest roadmap nodes
    (cost `metric_length / speed`), then
    `min over (a, c) of conn0[a] + node_dist[a, c] + conn1[c]`. `inf`
    propagates where the graph cannot connect the pair -- the caller masks
    those.

    `roadmap` must be CLOSED OVER, not a jit argument (`roadmap.k` has to be
    a static Python int).
    """
    nodes = roadmap.nodes
    node_dist = roadmap.node_dist
    periodic = jnp.asarray(domain.periodic)
    dim = domain.dim

    def endpoint(X):
        diff = _wrap_disp(X[:, None, :] - nodes[None, :, :], periodic)  # (B, M, dim)
        d_norm = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)
        _, idx = jax.lax.top_k(-d_norm, roadmap.k)  # (B, k) nearest nodes
        sel = jnp.take_along_axis(diff, idx[:, :, None], axis=1)  # (B, k, dim), = X - node
        mid = domain.wrap((X[:, None, :] - 0.5 * sel).reshape(-1, dim))
        g_hat = jnp.linalg.inv(domain.metric_inv(mid)).reshape(idx.shape[0], roadmap.k, dim, dim)
        length = jnp.sqrt(jnp.clip(jnp.einsum("bki,bkij,bkj->bk", sel, g_hat, sel), 1e-12, None))
        speed = jnp.minimum(roadmap.node_speed[idx], domain.speed(X)[:, None])
        conn = length / jnp.maximum(speed, roadmap.min_speed)
        return idx, conn

    idx0, conn0 = endpoint(X0)
    idx1, conn1 = endpoint(X1)
    bridged = conn0[:, :, None] + node_dist[idx0[:, :, None], idx1[:, None, :]] + conn1[:, None, :]
    return jnp.min(bridged.reshape(bridged.shape[0], -1), axis=1)
