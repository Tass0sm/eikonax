"""Model designs a training strategy can be run against. A backend is a
module exposing `init(key, domain, cfg) -> params`, `travel_time(params,
X0, X1, cfg) -> (n,)`, `trainable_mask(params)` and `num_params(params)`
-- nothing in `eikonax.ntfields.strategies` knows which one it has."""

import jax
import jax.numpy as jnp

from . import metric_net

BACKENDS = {"metric_net": metric_net}


def time_and_grads(backend, params, X0, X1, cfg):
    """`T(x0, x1)` and its gradient with respect to EACH endpoint, in one
    forward/backward pass. `T` is elementwise over the batch, so the
    cotangent of the summed output is the per-row gradient."""
    time, vjp = jax.vjp(lambda a, b: backend.travel_time(params, a, b, cfg), X0, X1)
    grad0, grad1 = vjp(jnp.ones_like(time))
    return time, grad0, grad1


__all__ = ["BACKENDS", "metric_net", "time_and_grads"]
