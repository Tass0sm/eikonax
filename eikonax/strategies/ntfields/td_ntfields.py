"""TD-NTFields: the physics-informed objective of Ni, Pan & Qureshi (ICLR
2025), ported from the `ntrl-demo` reference implementation and generalized
from "isotropic Euclidean speed field" to "arbitrary per-point Riemannian
metric plus speed field" -- the same generalization `eikonax.fsm` makes on
the grid side, so a `Domain` and an `fsm.Solver` can be handed the same
`(metric, speed_fn)` and compared.

The field being fit is the two-point arrival time `T(x0, x1)` of the
backend (see `backends.metric_net`), and every term is evaluated at BOTH
endpoints, exactly as the reference does. For a domain with speed `S(x)`
and inverse metric `G(x)^-1`, writing `|p|_x = sqrt(p^T G(x)^-1 p)` for the
dual norm of a covector, the eikonal equation is `S(x) * |grad T|_x = 1`,
and the objective is

    L = mean[ (w_E*L_E + w_TD*L_TD + w_N*L_N) * exp(-lambda_C * T) ]

  - `L_E  = (sqrt(S(x) * |grad_x T|_x) - 1)^2`, per endpoint. The
    reference's squared, one-directional eikonal residual -- NOT NTFields'
    symmetric `|1-sqrt(q)| + |1-1/sqrt(q)|`; the two papers genuinely use
    different eikonal losses.
  - `L_TD = (T - stopgrad[T(x - dx, x') + dt])^2`, per endpoint: a Bellman
    backup one characteristic step toward the other endpoint. The step
    is `dx = td_step * S(x) * G(x)^-1 grad_x T` (which has metric length
    `td_step` exactly at convergence, where `|grad T|_x = 1/S`) and its cost
    is charged at the ground-truth `dt = td_step / S(x)`. The WHOLE target
    -- stepped point and step cost -- is stop-gradiented, and the term is
    masked wherever `T < dt`, i.e. within one step of the other endpoint,
    where stepping would overshoot past it.
  - `L_N  = (1.001 - S(x)) * |S(x) grad_x T + n(x)|_x^2`, per endpoint,
    with `n = grad S / |grad S|_x` the unit outward obstacle normal:
    inside/near an obstacle (where `S -> 0` makes the weight ~1) the
    characteristic must run along the obstacle's normal rather than through
    it. The reference reads `n` from a precomputed surface-normal dataset;
    taking it from the speed field's own gradient is the general form (and
    what makes this work for any `speed_fn`, not just an obstacle mesh).
  - `exp(-lambda_C * T)` is the causality curriculum: near pairs are fit
    first, far pairs only once the near ones are cheap. It is NOT
    stop-gradiented by default, matching the reference -- safe here because
    the quasimetric head bounds `T` (a `T = base/tau` factorization would
    instead be paid to inflate `T` and kill its own loss).

Deviations from the reference implementation:

  - Collocation pairs are resampled every batch instead of being drawn from
    a precomputed dataset, and the reference's restriction of `x0` to a thin
    band around the obstacle SURFACE is dropped -- there is no mesh here,
    only a `speed_fn`. Its pair GEOMETRY is kept, and matters: `x1 = x0 +
    unit * U(0, pair_radius)` makes short-range pairs common, which is what
    the `exp(-lambda_C * T)` curriculum needs to have anything to bite on
    (drawing both endpoints independently concentrates every pair near the
    domain diameter, where the curriculum weight is uniformly tiny and the
    field never gets anchored).
  - The reference's `S` is `clip(distance_to_obstacle/margin, 0.1, 1)`,
    computed offline against a mesh; here it is whatever `domain.speed`
    returns, so that ramp (or any other cost field) is the caller's to
    write. Its data pipeline's remap `S <- alpha*(S*(2-S))^2 + (1-alpha)`
    with `alpha = 1.025` is applied on top, as there, but the result is
    floored at `cfg.min_speed`: `alpha > 1` drives it negative wherever `S`
    is small, which the reference feeds straight into a `sqrt`.
  - Retries are capped (`cfg.rollback_max_retries`); the reference's
    rollback loop is unbounded.
"""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from ...backends import time_and_grads
from ...domains import DTYPE, dual_norm


def speed_star(domain, Xn, cfg):
    """Ground-truth speed `S` at normalized points, after the reference
    data pipeline's remap (see module docstring)."""
    s = domain.speed(Xn)
    if cfg.speed_smoothstep:
        s = (s * (2.0 - s)) ** 2
    s = cfg.speed_alpha * s + (1.0 - cfg.speed_alpha)
    return jnp.clip(s, cfg.min_speed, None)


def sample_pairs(domain, rng: np.random.Generator, n: int, radius: float | None = None, max_tries: int = 20):
    """The reference's pair sampler: `x0` uniform over the domain, `x1 =
    x0 + unit_direction * U(0, radius)` with `radius = sqrt(dim)` in
    normalized units by default. Periodic axes wrap; on non-periodic axes
    out-of-box partners are rejected and redrawn (the reference's own
    accept/reject loop), so pairs stay uniform rather than piling up on
    the boundary."""
    dim = domain.dim
    radius = np.sqrt(dim) if radius is None else radius
    kept0, kept1, total = [], [], 0
    for _ in range(max_tries):
        X0 = rng.uniform(-0.5, 0.5, (4 * n, dim))
        step = rng.standard_normal((4 * n, dim))
        step /= np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        X1 = X0 + step * (radius * rng.random((4 * n, 1)))
        inside = np.ones(len(X1), dtype=bool)
        for axis, periodic in enumerate(domain.periodic):
            if periodic:
                X1[:, axis] = np.mod(X1[:, axis] + 0.5, 1.0) - 0.5
            else:
                inside &= (X1[:, axis] >= -0.5) & (X1[:, axis] <= 0.5)
        kept0.append(X0[inside])
        kept1.append(X1[inside])
        total += int(inside.sum())
        if total >= n:
            break
    X0, X1 = np.concatenate(kept0)[:n], np.concatenate(kept1)[:n]
    if len(X0) < n:  # pathologically thin domain -- resample with replacement to keep shapes static
        idx = rng.integers(0, len(X0), n)
        X0, X1 = X0[idx], X1[idx]
    return jnp.asarray(X0, dtype=DTYPE), jnp.asarray(X1, dtype=DTYPE)


def speed_normal(domain, Xn, metric_inv, cfg):
    """Unit (in the dual norm) outward obstacle normal `grad S / |grad S|`."""
    grad = jax.vmap(jax.grad(lambda x: speed_star(domain, x[None, :], cfg)[0]))(Xn)
    return grad / jnp.maximum(dual_norm(grad, metric_inv), 1e-8)[:, None]


def _endpoint_terms(backend, params, domain, cfg, time, Xn, grad, X_other):
    """The three per-endpoint loss terms at one endpoint of each pair."""
    metric_inv = domain.metric_inv(Xn)
    s = speed_star(domain, Xn, cfg)
    inv_speed = dual_norm(grad, metric_inv)

    eikonal = (jnp.sqrt(jnp.clip(s * inv_speed, 1e-12, None)) - 1.0) ** 2

    step_time = cfg.td_step / s
    disp = (cfg.td_step * s)[:, None] * jnp.einsum("nij,nj->ni", metric_inv, grad)
    stepped = domain.wrap(Xn - disp)
    target = jax.lax.stop_gradient(backend.travel_time(params, stepped, X_other, cfg) + step_time)
    td = jnp.where(time < step_time, 0.0, (time - target) ** 2)

    aligned = s[:, None] * grad + speed_normal(domain, Xn, metric_inv, cfg)
    normal = (1.001 - s) * jnp.einsum("ni,nij,nj->n", aligned, metric_inv, aligned)

    return eikonal, td, normal


def loss_terms(backend, params, X0, X1, domain, cfg):
    """`(eikonal, td, normal, causal)`, each `(n,)`, summed over both
    endpoints of every pair."""
    time, grad0, grad1 = time_and_grads(backend, params, X0, X1, cfg)
    e0, td0, n0 = _endpoint_terms(backend, params, domain, cfg, time, X0, grad0, X1)
    e1, td1, n1 = _endpoint_terms(backend, params, domain, cfg, time, X1, grad1, X0)
    weight = jax.lax.stop_gradient(time) if cfg.detach_causal else time
    return e0 + e1, td0 + td1, n0 + n1, jnp.exp(-cfg.causal_lambda * weight)


def solve(domain, cfg, backend, progress_fn=None):
    """Fit `backend`'s two-point travel time on `domain` with the
    TD-NTFields objective. Returns the trained parameters.

    `progress_fn(epoch, metrics)`, if given, is called every
    `cfg.log_every` epochs with a dict of scalar training metrics.

    The training loop is the reference's, not this repo's house style: an
    epoch is `cfg.batches_per_epoch` batches; the loss is rescaled by
    `beta = 1/loss` from the previous epoch (`cfg.adaptive_beta`); and an
    epoch whose loss grew by more than `cfg.rollback_ratio` is DISCARDED
    and re-run from a randomly chosen one of the last `cfg.rollback_queue`
    checkpoints (`cfg.rollback`). That last mechanism is what keeps the
    objective's `sqrt`/`1/S` terms from turning one bad step into a dead
    run, and is why the reference can train at a fixed learning rate.
    """
    params = backend.init(jax.random.PRNGKey(cfg.seed), domain, cfg)
    rng = np.random.default_rng(cfg.seed)

    optimizer = optax.masked(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)),
        backend.trainable_mask(params),
    )
    opt_state = optimizer.init(params)

    def loss_fn(p, X0, X1, beta):
        eikonal, td, normal, causal = loss_terms(backend, p, X0, X1, domain, cfg)
        weighted = cfg.eikonal_weight * eikonal + cfg.td_weight * td + cfg.normal_weight * normal
        objective = jnp.mean(weighted * causal)
        return beta * objective, {
            "objective": objective,
            "eikonal": jnp.mean(eikonal),
            "td": jnp.mean(td),
            "normal": jnp.mean(normal),
        }

    @jax.jit
    def step(p, state, X0, X1, beta):
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, X0, X1, beta)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, aux

    history: deque = deque(maxlen=cfg.rollback_queue)
    beta, previous = jnp.float32(1.0), None
    progress = trange(cfg.epochs, desc="td_ntfields")
    for epoch in progress:
        history.append((params, opt_state))
        for retry in range(cfg.rollback_max_retries + 1):
            trial_params, trial_state = params, opt_state
            metrics = {}
            for _ in range(cfg.batches_per_epoch):
                X0, X1 = sample_pairs(domain, rng, cfg.batch_size, cfg.pair_radius)
                trial_params, trial_state, aux = step(trial_params, trial_state, X0, X1, beta)
                metrics = {k: metrics.get(k, 0.0) + float(v) / cfg.batches_per_epoch for k, v in aux.items()}
            current = metrics["objective"]
            grew = previous is not None and not (0.0 < current / max(previous, 1e-12) < cfg.rollback_ratio)
            if not (cfg.rollback and grew) or retry == cfg.rollback_max_retries:
                break
            params, opt_state = history[rng.integers(len(history))]
            progress.write(f"[td_ntfields] epoch {epoch}: loss grew to {current:.4e} -- rolling back")
        params, opt_state = trial_params, trial_state
        previous = current
        if cfg.adaptive_beta:
            beta = jnp.asarray(1.0 / max(current, 1e-12), dtype=DTYPE)
        if epoch % cfg.log_every == 0:
            progress.set_description(f"td_ntfields -- log10(loss) = {np.log10(current + 1e-12):.3f}")
            if progress_fn is not None:
                progress_fn(epoch, metrics)
    return params
