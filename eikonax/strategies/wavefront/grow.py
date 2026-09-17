"""Lay out and grow N-D cone splats (`cones.py`) around obstacles.

Two stages, both driven by the obstacle CLEARANCE (distance to the nearest
obstacle) rather than by ray marching through the speed field: in a robot
configuration space a collision checker's distance query is exactly this,
while an occupancy ray test is not.

**1. Layout (`leaves`).** A `2^dim`-tree over the box. A cell is split while
its window radius `R = window_scale * halfdiag * size` is too large for
where it sits:

  - `R > max(clearance(centre), min_window)` -- the window must fit in free
    space, so that every straight segment inside it is collision-free. The
    floor lets windows hug a wall; keep `2 * min_window` below the thinnest
    obstacle, or a window could span one and the field would leak through.
  - `|grad^2 n| R^3 / 6 > tol` (at the centre and the corners) -- the same
    third-order budget the 1-D chain uses, so a slow region gets small
    splats and uniform space gets few big ones.

Cells whose centre is inside an obstacle are dropped.

**2. Growth (`solve`).** Dijkstra over the splats, from the source outward,
carrying a wavefront instead of just a value. Splat `y` is reached from an
accepted neighbour `j` (`|B_y - B_j| < max(R_j, R_y)`, so the hop is a
straight segment inside a free window) in one of two ways:

  - `S_j` VISIBLE from `B_y`: `y` inherits `j`'s wave. Value `L_j(B_y)`,
    direction `grad L_j(B_y)`, virtual source distance `|B_y - S_j|`. With
    uniform speed this is exact -- the whole lit region ends up as one
    cone, however many splats cover it.
  - `S_j` BLOCKED: the wave cannot have reached `y` from `S_j`, so `j`
    itself becomes `y`'s source: `c_j + n_bar |B_y - B_j|`, `rho_y =
    |B_y - B_j|`. This is what puts a fresh circular front just past an
    obstacle's corner -- Keller's diffraction source, found without ever
    searching for a corner.

Visibility is decided by conservative advancement (`visible`): step along
the segment by the clearance at the current point, which cannot jump over
an obstacle, and is how a distance-query collision checker validates a
roadmap edge. It runs once per neighbour pair while growing, never per
query point.

The front's curvature is therefore not transported by a Riccati equation;
it is re-derived from whichever virtual source is currently visible, which
is what makes diffraction come out right.
"""

from __future__ import annotations

import heapq

import jax
import jax.numpy as jnp
import numpy as np

from .cones import ConeField, local_gradient, values_of


def _slowness_fn(domain, cfg):
    """`x -> 1/speed(x)` over physical coordinates (jittable)."""
    def slowness(x):
        speed = domain.speed_fn(x[None, :])[0]
        return 1.0 / jnp.clip(speed, cfg.min_speed, None)
    return slowness


def leaves(domain, cfg, clearance_fn):
    """Splat centres and window radii for the box, `(k, dim)` and `(k,)`."""
    dim = domain.dim
    half_diag = np.sqrt(dim) / 2.0
    lower, upper = domain.lower, domain.upper

    slowness = _slowness_fn(domain, cfg)
    hess = jax.jit(jax.vmap(jax.hessian(slowness)))
    clearance = jax.jit(lambda X: clearance_fn(X))
    speed = jax.jit(lambda X: domain.speed_fn(X))
    corner_offsets = np.stack(np.meshgrid(*([np.array([-0.5, 0.5])] * dim), indexing="ij"), -1)
    corner_offsets = corner_offsets.reshape(-1, dim)

    size = cfg.max_window / (cfg.window_scale * half_diag)
    counts = np.maximum(np.ceil((upper - lower) / size).astype(int), 1)
    axes = [lower[i] + (np.arange(counts[i]) + 0.5) * size for i in range(dim)]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)
    sizes = np.full(len(centres), size)

    out_B, out_R = [], []
    while len(centres):
        R = cfg.window_scale * half_diag * sizes
        clear = np.asarray(clearance(jnp.asarray(centres)), dtype=np.float64)
        probes = (centres[:, None, :] + corner_offsets[None] * sizes[:, None, None]).reshape(-1, dim)
        curvature = np.linalg.norm(np.asarray(hess(jnp.asarray(probes))), axis=(-2, -1))
        curvature = np.concatenate([
            curvature.reshape(len(centres), -1),
            np.linalg.norm(np.asarray(hess(jnp.asarray(centres))), axis=(-2, -1))[:, None],
        ], axis=1).max(axis=1)

        too_big = R > np.maximum(clear, cfg.min_window)
        too_curved = curvature * R ** 3 / 6.0 > cfg.tol
        split = (too_big | too_curved) & (R / 2.0 >= cfg.min_window)
        out_B.append(centres[~split])
        out_R.append(R[~split])

        centres = (centres[split][:, None, :]
                   + corner_offsets[None] * sizes[split][:, None, None] / 2.0).reshape(-1, dim)
        sizes = np.repeat(sizes[split] / 2.0, len(corner_offsets))
        if len(centres):  # drop children that fall outside the box entirely
            keep = np.all((centres > lower - sizes[:, None] / 2) & (centres < upper + sizes[:, None] / 2), axis=1)
            centres, sizes = centres[keep], sizes[keep]

    B = np.concatenate(out_B)
    R = np.concatenate(out_R)
    free = np.asarray(speed(jnp.asarray(B))) > cfg.obstacle_speed
    return B[free], R[free]


def visibility_fn(clearance_fn, cfg):
    """`(A, B) -> bool (m,)`: is every segment `A[i] -> B[i]` obstacle-free?

    Conservative advancement: step by the clearance at the current point,
    which can never step over an obstacle. Returns `False` for a segment
    still unfinished after `vis_steps` steps (a ray creeping along a
    surface), which only ever costs an unnecessary diffraction source.
    """
    @jax.jit
    @jax.vmap
    def visible(a, b):
        length = jnp.linalg.norm(b - a)
        u = (b - a) / jnp.maximum(length, 1e-12)

        def cond(state):
            t, ok, i = state
            return ok & (t < length) & (i < cfg.vis_steps)

        def body(state):
            t, ok, i = state
            clear = clearance_fn((a + t * u)[None, :])[0]
            return t + jnp.maximum(clear, cfg.vis_eps), ok & (clear > cfg.vis_eps), i + 1

        t, ok, _ = jax.lax.while_loop(cond, body, (0.0, True, 0))
        return ok & (t >= length)
    return visible


def _neighbours(B, R):
    """`j` in `i`'s list when `|B_i - B_j| < max(R_i, R_j)` (symmetric)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(B)
    lists = tree.query_ball_point(B, R)   # j within R_i of i
    out = [set(js) for js in lists]
    for i, js in enumerate(lists):
        for j in js:
            out[j].add(i)
    return [np.array(sorted(js - {i}), dtype=int) for i, js in enumerate(out)]


def solve(domain, source, cfg, progress_fn=None):
    """Lay out and grow a `ConeField` from `source` (physical). Returns
    `(field, params)`; unreached splats are dropped."""
    field = ConeField(domain, source, cfg)
    clearance_fn = domain.clearance_fn
    if clearance_fn is None:
        raise ValueError(
            "wavefront needs a clearance function in 2-D and up: "
            "`domain.clearance_fn` is None (eikonax.scenarios attaches one to its speed fields)")
    dim = domain.dim
    B, R = leaves(domain, cfg, clearance_fn)
    # the source is splat 0
    src_clear = float(clearance_fn(jnp.asarray(field.source[None])) [0])
    B = np.concatenate([field.source[None], B])
    R = np.concatenate([[min(max(src_clear, cfg.min_window), cfg.max_window)], R])
    k = len(B)

    slowness = jax.jit(jax.vmap(_slowness_fn(domain, cfg)))
    slowness_grad = jax.jit(jax.vmap(jax.grad(_slowness_fn(domain, cfg))))
    params = {
        "B": B,
        "R": R,
        "c": np.full(k, np.inf),
        "p": np.zeros((k, dim)),
        "rho": np.zeros(k),
        "n": np.asarray(slowness(jnp.asarray(B)), dtype=np.float64),
        "g": np.asarray(slowness_grad(jnp.asarray(B)), dtype=np.float64),
    }
    params["p"][:, 0] = 1.0
    nbrs = _neighbours(B, R)
    visible = visibility_fn(clearance_fn, cfg)
    max_degree = max(len(js) for js in nbrs)

    def can_see(origin, targets):
        """Visibility of `origin -> targets`, padded to one compiled shape."""
        pad = np.repeat(targets[:1], max_degree - len(targets), axis=0)
        batch = np.concatenate([targets, pad]) if len(pad) else targets
        seen = np.asarray(visible(jnp.tile(jnp.asarray(origin), (max_degree, 1)), jnp.asarray(batch)))
        return seen[:len(targets)]

    params["c"][0] = 0.0
    accepted = np.zeros(k, dtype=bool)
    heap = [(0.0, 0)]
    n_diffractions = 0
    while heap:
        c_i, i = heapq.heappop(heap)
        if accepted[i]:
            continue
        accepted[i] = True
        js = nbrs[i][~accepted[nbrs[i]]]
        if not len(js):
            continue

        # can each neighbour see this splat's virtual source?
        S_i = params["B"][i] - params["rho"][i] * params["p"][i]
        seen = can_see(S_i, params["B"][js])
        L_i = values_of(params["B"][js], params, i)
        d = params["B"][js] - params["B"][i]
        hop = np.linalg.norm(d, axis=1)
        via_i = params["c"][i] + 0.5 * (params["n"][i] + params["n"][js]) * hop
        cand = np.where(seen, np.maximum(L_i, params["c"][i]), via_i)

        better = cand < params["c"][js]
        for j, value, inherits in zip(js[better], cand[better], seen[better]):
            params["c"][j] = value
            if inherits:
                g = local_gradient(params["B"][j], params, i)
                params["p"][j] = g / max(np.linalg.norm(g), 1e-12)
                params["rho"][j] = np.linalg.norm(params["B"][j] - S_i)
            else:
                step = params["B"][j] - params["B"][i]
                params["p"][j] = step / max(np.linalg.norm(step), 1e-12)
                params["rho"][j] = np.linalg.norm(step)
            heapq.heappush(heap, (float(value), int(j)))
        n_diffractions += int(np.sum(better & ~seen))

    reached = np.isfinite(params["c"]) & accepted
    params = {key: jnp.asarray(value[reached]) for key, value in params.items()}
    if progress_fn is not None:
        progress_fn(0, {"splats": int(reached.sum()), "laid_out": k,
                        "diffraction_sources": n_diffractions})
    return field, params
