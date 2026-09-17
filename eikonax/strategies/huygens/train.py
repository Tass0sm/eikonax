"""Grow a Huygens field from its source, then refine where the wavelets sit.

The field starts as the source wavelet alone -- exact for uniform speed and
no obstacles, in which case nothing is ever added. Each round:

1. **Grow.** Draw free-space candidates `y` and collocation points `x`. A
   candidate's emission time is the current field at it, `T(y)` (so it is
   born consistent), and adding it would lower the field to
   `min(T(x), T(y) + ray_y(x))`. Its gain is the resulting drop in
   `mean_x T(x)`. Greedily add the best candidate, update `T` (and the
   candidates' own `T(y)`, which the new wavelet may have lowered), and
   repeat, up to `spawn_per` wavelets or until the best gain is below
   `growth_tol`. Points no wavelet can see sit at `~occlusion_cost`, so
   covering them dominates the gain: shadows are filled first.
   Candidates are uniform free points, plus points on the current shadow
   boundary (the covered sample nearest each uncovered one) and jittered
   copies of existing centres: a diffraction source belongs at a shadow
   edge, which uniform sampling rarely hits in a narrow gap.
2. **Refine.** `refine_steps` of Adam on the centres minimizing `mean_x T(x)`, with the emission times
   recomputed differentiably each step (`HuygensField.values`). Since every
   `T` value is a real path length, this tightens an upper bound: a
   diffraction wavelet slides onto the corner it re-emits from. The
   gradient cannot see visibility cliffs (occlusion is a constant penalty),
   so a step is only accepted if `mean T` on its batch does not rise --
   without this, wavelets slid out of view of the shadow they covered,
   and growth kept re-covering it (the wall scene piled 35 wavelets onto
   one corner). A step that would move a centre into an obstacle is not
   taken for that centre.
3. **Prune.** Drop, one at a time, any wavelet whose removal raises
   `mean T` by less than `growth_tol` (emission times recomputed without
   it), so refinement that stacks wavelets on the same feature leaves one.

Training stops after a round that adds nothing, or at `max_rounds` /
`max_splats`. There is no eikonal residual in the objective: with the
domain's metric in the ray cost, each wavelet already has the right slope
wherever it is the minimizer, and the residual cannot see a wrong LEVEL
(it is reported as a diagnostic only).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ...domains import DTYPE, dual_norm
from .wavelets import HuygensField


def sample_free(domain, rng: np.random.Generator, n: int, obstacle_speed: float, max_tries: int = 50):
    """`n` uniform normalized points with `speed > obstacle_speed`."""
    out, have = [], 0
    for _ in range(max_tries):
        X = domain.sample(rng, 2 * n)
        X = X[np.asarray(domain.speed(X)) > obstacle_speed]
        out.append(np.asarray(X))
        have += len(X)
        if have >= n:
            break
    X = np.concatenate(out)[:n]
    if len(X) < n:
        raise ValueError(f"could only sample {len(X)}/{n} free points -- is the domain all obstacle?")
    return jnp.asarray(X, dtype=DTYPE)


def eikonal_residual(field: HuygensField, params, Xn):
    """`speed * |grad T|_{G^-1} - 1` at each point (diagnostic)."""
    grad = field.grad(params, Xn)
    speed = jnp.clip(field.domain.speed(Xn), field.cfg.min_speed, None)
    return speed * dual_norm(grad, field.domain.metric_inv(Xn)) - 1.0


def solve(domain, source_n, cfg, progress_fn=None):
    """Grow and refine a `HuygensField` from `source_n` (normalized
    coordinates). Returns `(field, params)`, `params` with `V` up to date."""
    field = HuygensField(domain, source_n, cfg)
    rng = np.random.default_rng(cfg.seed)
    params = field.empty()

    rays_fn = jax.jit(lambda p, X: field.rays(p, X))
    evaluate_fn = jax.jit(field.evaluate)
    values_fn = jax.jit(field.values)
    residual_fn = jax.jit(lambda p, X: eikonal_residual(field, p, X))

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.multi_transform(
            {"V": optax.set_to_zero(), "A": optax.set_to_zero(), "B": optax.adam(cfg.center_lr)},
            ("V", "A", "B"),
        ),
    )

    def loss_fn(p, X):
        return jnp.mean(field.time(p, X))

    @jax.jit
    def step(p, state, X):
        loss, grads = jax.value_and_grad(loss_fn)(p, X)
        updates, new_state = optimizer.update(grads, state, p)
        V, A, B = optax.apply_updates(p, updates)
        B = domain.wrap(B)
        B = jnp.where((domain.speed(B) > cfg.obstacle_speed)[:, None], B, p[2])
        new = (V, A, B)
        accept = loss_fn(new, X) <= loss
        pick = lambda a, b: jax.tree_util.tree_map(lambda x, y: jnp.where(accept, x, y), a, b)
        return pick(new, p), pick(new_state, state), loss

    mean_time_fn = jax.jit(lambda p, X: jnp.mean(field.time(p, X)))

    for round_ in range(cfg.max_rounds):
        X = sample_free(domain, rng, cfg.batch_size, cfg.obstacle_speed)
        added, gain = _grow(field, params, X, rng, rays_fn, evaluate_fn)
        if added is not None:
            params = field.append(params, added)
            params = (values_fn(params), params[1], params[2])

        if params[2].shape[0] > 0 and cfg.center_lr > 0:
            state = optimizer.init(params)
            for _ in range(cfg.refine_steps):
                X_step = sample_free(domain, rng, cfg.batch_size, cfg.obstacle_speed)
                params, state, _ = step(params, state, X_step)
            params = (values_fn(params), params[1], params[2])

        params, pruned = _prune(field, params, X, mean_time_fn)
        params = (values_fn(params), params[1], params[2])

        if progress_fn is not None:
            T = np.asarray(evaluate_fn(params, X))
            res = np.asarray(residual_fn(params, X))
            progress_fn(round_, {
                "mean_T": float(T.mean()),
                "uncovered": float(np.mean(T >= cfg.occlusion_cost)),
                "eikonal_rms": float(np.sqrt(np.mean(res[T < cfg.occlusion_cost] ** 2))),
                "best_gain": gain,
                "num_splats": int(params[2].shape[0]),
                "pruned": pruned,
            })
        if added is None:
            break
    return field, params


def _grow(field, params, X, rng, rays_fn, evaluate_fn):
    """Greedy additions for one round: `(new wavelets or None, best gain seen)`."""
    cfg = field.cfg
    room = cfg.max_splats - int(params[2].shape[0])
    if room <= 0:
        return None, 0.0
    T_x = np.asarray(evaluate_fn(params, X))
    cand = _candidates(field, params, X, T_x, rng)
    T_c = np.asarray(evaluate_fn(params, cand))
    m = cand.shape[0]

    # ray costs candidate -> collocation (n, m) and candidate -> candidate (m, m), chunked by candidate
    as_wavelets = field.wavelets_at(cand)
    ray_x = np.concatenate([np.asarray(rays_fn(_chunk(as_wavelets, i, cfg.chunk), X))[:, 1:]
                            for i in range(0, m, cfg.chunk)], axis=1)
    ray_c = np.concatenate([np.asarray(rays_fn(_chunk(as_wavelets, i, cfg.chunk), cand))[:, 1:]
                            for i in range(0, m, cfg.chunk)], axis=1)

    chosen, best_gain = [], 0.0
    for _ in range(min(cfg.spawn_per, room)):
        new_T = T_c[None, :] + ray_x
        gain = np.maximum(T_x[:, None] - new_T, 0.0).mean(axis=0)
        gain[chosen] = 0.0
        j = int(np.argmax(gain))
        best_gain = max(best_gain, float(gain[j]))
        if gain[j] < cfg.growth_tol:
            break
        chosen.append(j)
        T_x = np.minimum(T_x, new_T[:, j])
        T_c = np.minimum(T_c, T_c[j] + ray_c[:, j])

    if not chosen:
        return None, best_gain
    return field.wavelets_at(cand[np.asarray(chosen)]), best_gain


def _candidates(field, params, X, T_x, rng):
    """Uniform free points + shadow-boundary points + jittered existing centres."""
    cfg, domain = field.cfg, field.domain
    parts = [np.asarray(sample_free(domain, rng, cfg.candidates, cfg.obstacle_speed))]
    X = np.asarray(X)
    uncovered = T_x >= cfg.occlusion_cost
    if uncovered.any() and (~uncovered).any():
        U, C = X[uncovered], X[~uncovered]
        U = U[rng.choice(len(U), min(len(U), cfg.candidates // 2), replace=False)]
        nearest = C[np.argmin(((U[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=1)]
        parts.append(nearest)
    if params[2].shape[0] > 0:
        B = np.asarray(params[2])
        B = B[rng.integers(len(B), size=cfg.candidates // 4)]
        parts.append(B + cfg.jitter * rng.standard_normal(B.shape))
    cand = jnp.asarray(np.concatenate(parts), dtype=DTYPE)
    cand = domain.wrap(cand)
    return cand[np.asarray(domain.speed(cand)) > cfg.obstacle_speed]


def _prune(field, params, X, mean_time_fn):
    """Backward elimination; returns `(params, number removed)`."""
    cfg = field.cfg
    removed = 0
    base = float(mean_time_fn(params, X))
    j = 0
    while j < params[2].shape[0]:
        keep = np.delete(np.arange(params[2].shape[0]), j)
        trial = tuple(p[keep] for p in params)
        value = float(mean_time_fn(trial, X))
        if value - base < cfg.growth_tol:
            params, base, removed = trial, value, removed + 1
        else:
            j += 1
    return params, removed


def _chunk(params, start, size):
    return tuple(p[start:start + size] for p in params)
