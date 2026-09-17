"""The wavefront field: a partition of unity over local travel-time models.

    T(x) = sum_j pi_j(x) L_j(x),        pi_j = w_j / sum_m w_m

  - `w_j` is the splat's WINDOW, a compactly supported density
    `(1 - |x - B_j|^2 / R_j^2)^3_+` (C^2). The window is what makes this an
    SRM-style universal approximator: a normalized SRM with scalar weights
    is the degree-0 case of this model (`baselines.pu0_fit`).
  - `L_j` is the splat's WEIGHT, a local model of the arrival time instead
    of a scalar:

        L_j(x) = c_j + p_j . d + 1/2 d^T H_j d,      d = x - B_j

    `p_j` is tied to the local speed -- `p_j = u_j / (speed(B_j) *
    |u_j|_{G^-1(B_j)})`, so the eikonal equation holds exactly at every
    centre and only the DIRECTION `u_j` is free. A single speed evaluation
    per splat, never a ray integral per query. `H_j` carries the change of
    slowness along the front (and, in more than 1-D, its curvature).
  - The source splat (index 0, always present, not trainable in position)
    is a cone: `L_0(x) = n_0 l + 1/2 (g_0 . d) l`, `l = |d|_{G(s)}`,
    `n_0 = 1/speed(s)`. That is the second-order expansion of `int n ds`
    along the straight ray from the source, with `g_0` initialized to
    `grad n(s)`. It is the only splat with a kink, which is where the true
    field has one.

Optionally (`value_temperature > 0`) the weights also favour the earliest
arrival, `pi_j ~ w_j exp(-L_j / value_temperature)`: a soft-min, which is
what a kink where two fronts meet needs. `None` is the pure partition of
unity.

A relay splat's model is only right on the OUTGOING side of the front near
its centre, so a relay window must not reach back across the source (the
chain initializer guarantees this; training is penalized by `source_leak`).

Parameters are a dict pytree; everything is in the domain's NORMALIZED
coordinates, arrival times in physical units (see `eikonax.domains`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ...domains import DTYPE


def window(z2):
    """`(1 - z2)^3` inside the unit ball, `0` outside; `z2 = |d|^2 / R^2`."""
    inside = z2 < 1.0
    return jnp.where(inside, (1.0 - jnp.where(inside, z2, 0.0)) ** 3, 0.0)


class WavefrontField:
    """The model bound to a domain and a source (normalized coordinates).

    `params`:
        `B (k, dim)`, `log_R (k, 2)`, `c (k,)`, `u (k, dim)`,
        `H (k, dim, dim)` -- the `k` relay splats, `k = 0` allowed;
        `src_log_R (2,)`, `src_g (dim,)` -- the source splat's window radii
        and slowness gradient.

    Windows are one-sided in their radius: a relay uses `log_R[:, 1]`
    ahead of its centre (along `u`) and `log_R[:, 0]` behind it, so a splat
    can reach far forward where the speed is flat without reaching back
    over short-step neighbours. The source uses `src_log_R[1]` on the `+`
    side of axis 0 and `src_log_R[0]` on the `-` side (1-D; an isotropic
    radius is the natural choice in higher dimensions).
    """

    def __init__(self, domain, source_n, cfg):
        self.domain = domain
        self.cfg = cfg
        self.dim = domain.dim
        self.source = jnp.asarray(source_n, dtype=DTYPE).reshape(self.dim)

    # -- local speed / metric at single points --------------------------------

    def speed_at(self, x):
        return jnp.clip(self.domain.speed(x[None])[0], self.cfg.min_speed, None)

    def metric_inv_at(self, x):
        return self.domain.metric_inv(x[None])[0]

    def slowness_at(self, x):
        """Physical slowness `1/speed` at a normalized point."""
        return 1.0 / self.speed_at(x)

    def covector(self, B, u):
        """`p = u / (speed(B) |u|_{G^-1(B)})` -- the eikonal-consistent slope."""
        norm = jnp.sqrt(jnp.clip(u @ self.metric_inv_at(B) @ u, 1e-12, None))
        return u / (self.speed_at(B) * norm)

    # -- the model -------------------------------------------------------------

    def locals_and_windows(self, params, x):
        """`L_j(x)` and `w_j(x)` for the source then every relay, `(k+1,)` each."""
        cfg = self.cfg
        # source cone
        d0 = x - self.source
        G0 = jnp.linalg.inv(self.metric_inv_at(self.source))
        ell = jnp.sqrt(d0 @ G0 @ d0 + cfg.cone_delta ** 2) - cfg.cone_delta
        L0 = self.slowness_at(self.source) * ell + 0.5 * (params["src_g"] @ d0) * ell
        log_R0 = jnp.where(d0[0] > 0, params["src_log_R"][1], params["src_log_R"][0])
        w0 = window(d0 @ d0 / jnp.exp(2.0 * log_R0))

        # relays
        d = x[None, :] - params["B"]
        p = jax.vmap(self.covector)(params["B"], params["u"])
        H = 0.5 * (params["H"] + jnp.swapaxes(params["H"], -1, -2))
        L = params["c"] + jnp.einsum("ki,ki->k", p, d) + 0.5 * jnp.einsum("ki,kij,kj->k", d, H, d)
        ahead = jnp.einsum("ki,ki->k", d, params["u"]) > 0
        log_R = jnp.where(ahead, params["log_R"][:, 1], params["log_R"][:, 0])
        w = window(jnp.sum(d * d, axis=-1) / jnp.exp(2.0 * log_R))

        return jnp.concatenate([L0[None], L]), jnp.concatenate([w0[None], w])

    def weights(self, L, w):
        """`pi_j` from the local values and windows."""
        temp = self.cfg.value_temperature
        if temp is None or temp <= 0:
            return w / (jnp.sum(w) + 1e-12)
        pos = w > 0
        logits = jnp.where(pos, jnp.log(jnp.where(pos, w, 1.0)) - L / temp, -1e30)
        return jax.nn.softmax(logits) * jnp.any(pos)

    def _blend(self, params, x):
        L, w = self.locals_and_windows(params, x)
        pi = self.weights(L, w)
        return jnp.sum(pi * L), L, pi, w

    def value(self, params, x):
        """`T` at one normalized point (scalar), shifted so `T(source) = 0`
        exactly -- no loss term has to hold the field's overall level."""
        return self._blend(params, x)[0] - self._blend(params, self.source)[0]

    def source_leak(self, params):
        """`sum_j w_j(source)` over the relays: how much relay windows cover
        the source, where their one-sided models are wrong."""
        _, w = self.locals_and_windows(params, self.source)
        return jnp.sum(w[1:])

    def stats(self, params, x):
        """`(T, grad T, sum_j pi_j (L_j - T)^2, sum_j w_j)` at one point --
        the value, its gradient, the local models' disagreement, and how
        well the windows cover `x`."""
        raw, L, pi, w = self._blend(params, x)
        grad = jax.grad(self.value, argnums=1)(params, x)
        return self.value(params, x), grad, jnp.sum(pi * (L - raw) ** 2), jnp.sum(w)

    def evaluate(self, params, Xn):
        """`T` at normalized points `(n, dim)`, `(n,)`."""
        return jax.vmap(self.value, in_axes=(None, 0))(params, Xn)

    def grad(self, params, Xn):
        """`dT/dXn`, `(n, dim)`."""
        return jax.vmap(jax.grad(self.value, argnums=1), in_axes=(None, 0))(params, Xn)

    def num_splats(self, params) -> int:
        return int(params["B"].shape[0]) + 1
