"""Refine every parameter of a chained wavefront field by gradient descent.

The chain (`chain.py`) is already a good field; training polishes it. The
loss at uniform free collocation points `x` is

    residual_weight    * mean (speed(x) |grad T(x)|_{G^-1} - 1)^2
  + consistency_weight * mean sum_j pi_j(x) (L_j(x) - T(x))^2
  + coverage_weight    * mean relu(min_coverage - sum_j w_j(x))^2
  + source_weight      * (sum_{relays j} w_j(source))^2

(`T(source) = 0` holds exactly by construction, see `field.value`.)
The residual alone cannot pin a wrong LEVEL of a splat whose window sits
inside a single other window, and the blend's gradient mixes the local
models' slopes with the windows' slopes; the consistency term (the local
models must agree where they overlap -- the usual partition-of-unity /
mixture-of-experts coupling) is what ties neighbouring emission times
together. Coverage keeps training from opening gaps between windows; the
source term keeps relay windows off the source's kink, where their
one-sided models are wrong.

**Measured limitation.** This objective does not share its minimizer with
the exact field. On the 1-D slow patch it helps a coarse chain (19 splats,
max error 8e-4 -> 3e-4) but degrades a fine one (67 splats, 7e-6 -> 1e-4)
while the residual itself drops 2e-8 -> 4e-9: the residual cannot see a
slowly drifting LEVEL, and there is no longer an upper-bound objective (as
in `huygens`) to pin it. Hence `solve`'s `train_steps=0` default. A
level-aware term -- e.g. the chain's own Hermite increments between
neighbouring centres as a consistency target -- is the obvious next step.

All of `B`, `log_R`, `c`, `u`, `H`, `src_log_R`, `src_g` are trained, with
the centres on their own (smaller) learning rate: they are spaced by the
error budget, and a step as large as the others' would reorder them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from ...domains import dual_norm


def losses(field, params, X):
    """The loss terms at normalized points `X`, as a dict of scalars."""
    cfg = field.cfg
    T, grad, disagreement, cover = jax.vmap(field.stats, in_axes=(None, 0))(params, X)
    speed = jnp.clip(field.domain.speed(X), cfg.min_speed, None)
    residual = speed * dual_norm(grad, field.domain.metric_inv(X)) - 1.0
    return {
        "residual": jnp.mean(residual ** 2),
        "consistency": jnp.mean(disagreement),
        "coverage": jnp.mean(jnp.maximum(cfg.min_coverage - cover, 0.0) ** 2),
        "source": field.source_leak(params) ** 2,
    }


def total(cfg, terms):
    return (cfg.residual_weight * terms["residual"]
            + cfg.consistency_weight * terms["consistency"]
            + cfg.coverage_weight * terms["coverage"]
            + cfg.source_weight * terms["source"])


def refine(field, params, rng, progress_fn=None):
    """`cfg.train_steps` of Adam on every parameter; returns `params`."""
    cfg = field.cfg
    if cfg.train_steps <= 0:
        return params

    labels = {k: ("centres" if k == "B" else "rest") for k in params}
    schedule = lambda lr: optax.cosine_decay_schedule(lr, cfg.train_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.multi_transform({"centres": optax.adam(schedule(cfg.center_lr)),
                               "rest": optax.adam(schedule(cfg.lr))}, labels),
    )
    state = optimizer.init(params)

    def loss_fn(p, X):
        terms = losses(field, p, X)
        return total(cfg, terms), terms

    @jax.jit
    def step(p, s, X):
        (loss, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, X)
        updates, s = optimizer.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss, terms

    log_every = max(1, cfg.train_steps // 20)
    for i in range(cfg.train_steps):
        X = field.domain.sample(rng, cfg.batch_size)
        params, state, loss, terms = step(params, state, X)
        if progress_fn is not None and (i % log_every == 0 or i == cfg.train_steps - 1):
            progress_fn(i, {"loss": float(loss), **{k: float(v) for k, v in terms.items()}})
    return params
