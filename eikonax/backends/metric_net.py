"""The `ntrl-demo` two-point network, ported to JAX -- a learned QUASIMETRIC
travel-time head, which is what makes TD-NTFields' Bellman loss well posed.

`T(x0, x1)` is not read off an output neuron. Both endpoints go through one
shared encoder `f`, and the travel time is a smooth-max distance between the
two embeddings::

    d      = sqrt((f(x0) - f(x1))^2 + eps)           per feature
    T      = out_scale * sum_g smoothmax_g(d)        log-sum-exp max over each group of `group` features

Two structural consequences, both load-bearing and neither obtainable from
a plain regression head:

  - `T` is symmetric, non-negative, and minimal on the diagonal by
    construction (`T(x, x)` is the smoothing floor `out_scale * hidden/group
    * sqrt(eps)`, not fit), so the boundary condition holds for free for
    EVERY source -- no point-source singularity, no per-source retraining. (This is the piece `srms`'s own `hntfields.py` deliberately
    did not port, keeping a fixed-source `T = base/tau` factorization
    instead; here the two-point form is the point, since the downstream
    consumer wants an all-pairs field.)
  - Every linear layer is spectrally normalized row-wise (`_lip_norm`), so
    `f` is ~1-Lipschitz and the smooth-max of coordinate distances is a
    genuine (pseudo)metric: `T` satisfies the triangle inequality
    approximately by construction rather than by training.

Faithful to the reference implementation except where noted:

  - Input is a random-Fourier feature map `[sin, cos](2*pi * x @ B)` with `B`
    FROZEN (it is a plain tensor there, not a parameter). On a PERIODIC axis
    the corresponding row of `B` is rounded to integers -- normalized
    periodic axes have period exactly 1 (see `domains`), so integer
    frequencies make the whole feature map, and therefore the whole network,
    exactly periodic. Non-periodic rows are left as sampled.
  - Hidden blocks are the reference's gated sine stack: two `u*sin(y) +
    v*(1-sin(y))` layers driven by two input-derived gates `u`, `v`,
    followed by a residual layer whose mix weight is a learned scalar
    initialized at `sigmoid(0) = 0.5`.
  - The reference's `InstanceNorm1d` on the final embedding is, for its 2-D
    input, a per-row normalization over the feature axis with no affine
    parameters -- `_row_norm` here.
  - `2 * n_freq` must equal `hidden`, because the reference's first
    residual block adds its input (the Fourier features) to its output (a
    hidden vector). `n_freq` therefore defaults to `hidden // 2`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ..domains import DTYPE

Layer = tuple[jnp.ndarray, jnp.ndarray]  # (W [out, in], b [out]), applied as x @ W.T + b


class Params(NamedTuple):
    fourier: jnp.ndarray          # [dim, n_freq], frozen (see `trainable_mask`)
    pe_gate: tuple[Layer, Layer]  # the two input-derived gates u, v
    blocks: tuple                 # n_blocks x (Layer, Layer, Layer)
    gates: tuple                  # n_blocks x [1], residual mix logits
    out: Layer


def _trunc_normal(key, shape, std):
    return (std * jax.random.truncated_normal(key, -2.0, 2.0, shape)).astype(DTYPE)


def _init_layer(key, fan_out, fan_in) -> Layer:
    std = jnp.sqrt(2.0 / (fan_out + fan_in))
    return _trunc_normal(key, (fan_out, fan_in), std), jnp.zeros((fan_out,), dtype=DTYPE)


def _softplus(x, beta):
    return jax.nn.softplus(beta * x) / beta


def _lip_norm(W, beta):
    """Row-wise soft spectral normalization: scales each output unit's
    weight row so its norm is at most ~1, with the scale treated as a
    constant (the reference `.detach()`s it)."""
    row_norm = jax.lax.stop_gradient(jnp.sqrt(jnp.sum(W ** 2, axis=1)))
    scale = 1.0 + 1e-5 - _softplus(1.0 - 1.0 / row_norm, beta)
    return W * scale[:, None]


def _dense(x, layer: Layer, beta):
    W, b = layer
    return x @ _lip_norm(W, beta).T + b


def _row_norm(y):
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.var(y, axis=-1, keepdims=True)
    return (y - mean) / jnp.sqrt(var + 1e-5)


def init(key, domain, cfg) -> Params:
    """Build the network for `domain`'s dimension, sized by `cfg.hidden`,
    `cfg.n_blocks`, `cfg.n_freq`."""
    n_freq = cfg.n_freq if cfg.n_freq is not None else cfg.hidden // 2
    if 2 * n_freq != cfg.hidden:
        raise ValueError(f"2 * n_freq must equal hidden, got {2 * n_freq} vs {cfg.hidden}")
    if cfg.hidden % cfg.group != 0:
        raise ValueError(f"hidden ({cfg.hidden}) must be divisible by group ({cfg.group})")

    keys = jax.random.split(key, 4 + 3 * cfg.n_blocks)
    B = _trunc_normal(keys[0], (domain.dim, n_freq), 1.0)
    periodic = jnp.asarray(domain.periodic)[:, None]
    B = jnp.where(periodic, jnp.round(B), B)

    h = cfg.hidden
    pe_gate = (_init_layer(keys[1], h, 2 * n_freq), _init_layer(keys[2], h, 2 * n_freq))
    blocks = tuple(
        tuple(_init_layer(keys[3 + 3 * i + j], h, h) for j in range(3))
        for i in range(cfg.n_blocks)
    )
    gates = tuple(jnp.zeros((1,), dtype=DTYPE) for _ in range(cfg.n_blocks))
    out = _init_layer(keys[-1], h, h)
    return Params(B, pe_gate, blocks, gates, out)


def trainable_mask(params: Params) -> Params:
    """`optax.masked` mask: everything but the frozen Fourier matrix."""
    return params._replace(fourier=False,
                           pe_gate=jax.tree_util.tree_map(lambda _: True, params.pe_gate),
                           blocks=jax.tree_util.tree_map(lambda _: True, params.blocks),
                           gates=jax.tree_util.tree_map(lambda _: True, params.gates),
                           out=jax.tree_util.tree_map(lambda _: True, params.out))


def embed(params: Params, Xn, cfg):
    """Shared encoder `f`: normalized points `(n, dim)` -> `(n, hidden)`."""
    beta = cfg.softplus_beta
    proj = 2.0 * jnp.pi * (Xn @ jax.lax.stop_gradient(params.fourier))
    x = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

    u = jnp.sin(_dense(x, params.pe_gate[0], beta))
    v = jnp.sin(_dense(x, params.pe_gate[1], beta))
    for (l1, l2, l3), gate in zip(params.blocks, params.gates):
        skip = x
        for layer in (l1, l2):
            y = jnp.sin(_dense(x, layer, beta))
            x = u * y + v * (1.0 - y)
        mix = jax.nn.sigmoid(0.1 * gate)
        x = (1.0 - mix) * skip + mix * jnp.sin(_dense(x, l3, beta))
    return _row_norm(_dense(x, params.out, beta))


def travel_time(params: Params, X0, X1, cfg):
    """`T(x0, x1)` for a batch of normalized point pairs, returned as `(n,)`."""
    e = embed(params, jnp.concatenate([X0, X1], axis=0), cfg)
    e0, e1 = e[: X0.shape[0]], e[X0.shape[0]:]
    d = jnp.sqrt((e0 - e1) ** 2 + 1e-6)
    d = d.reshape(d.shape[0], -1, cfg.group)
    smooth_max = (logsumexp(cfg.lse_scale * d, axis=2) - jnp.log(cfg.group)) / cfg.lse_scale
    return cfg.out_scale * jnp.sum(smooth_max, axis=1)


def num_params(params: Params) -> int:
    """Trainable scalars (the frozen Fourier matrix doesn't count)."""
    trainable = params._replace(fourier=None)
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(trainable)))
