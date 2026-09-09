"""Scene geometry -> a JAX-jittable speed field `S(q)` for the neural solver.

`eikonax.scenarios` gives analytic 2-D `speed_fn`s for the CLI. This module
is the general robot form the downstream consumer needs: the speed at a
configuration `q` is set by how close the ROBOT'S BODY comes to the scene
geometry, not by `q` itself. So

    S(q) = ramp( clearance(q) / margin )

with

    clearance(q) = min over the robot's collision spheres of
                   ( signed distance from the sphere centre to the scene
                     obstacles  -  that sphere's radius )

and the sphere centres are `fk(q)` -- forward kinematics from the
configuration to a set of workspace points. For a field trained directly in
workspace coordinates (`coordinate_space="workspace_xyz"`) `fk` is the
identity and there is a single zero-radius sphere: the configuration point
*is* the only body point. For a configuration-space field the caller passes
a real `fk_fn` mapping joint angles to link-frame sphere centres.

`ramp` is the reference implementation's `clip(d / margin, min_speed, 1)`
(see `strategies.ntfields.td_ntfields`'s module docstring); `smoothstep=True`
shapes the `[0, 1]` part with `3t^2 - 2t^3` first, for a `C1` speed field.

Everything here is `jax.numpy` and safe under `jit`/`vmap`/`grad` -- it is
evaluated at every collocation point of every training batch.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp


def box_sdf(points, centers, half_extents):
    """Signed distance from each point to the UNION of axis-aligned boxes.

    `points` is `(..., 3)`, `centers` and `half_extents` are `(K, 3)`.
    Returns `(...,)`: negative inside a box (the negative depth below the
    nearest face), positive outside (the Euclidean gap to the surface),
    minimised over the `K` boxes.
    """
    p = jnp.asarray(points)[..., None, :]                    # (..., 1, 3)
    q = jnp.abs(p - jnp.asarray(centers)) - jnp.asarray(half_extents)  # (..., K, 3)
    # `sqrt(sum + eps)`, not `jnp.linalg.norm`: the exterior distance is
    # `norm(max(q, 0))`, which is the zero vector for every point INSIDE a
    # box -- and `norm`'s gradient there is `0/0 = nan`. The training
    # objective differentiates this field (`speed_normal` takes `grad S`),
    # so the floor is load-bearing, not cosmetic.
    exterior = jnp.maximum(q, 0.0)
    outside = jnp.sqrt(jnp.sum(exterior ** 2, axis=-1) + 1e-12)  # (..., K)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)           # (..., K)
    return jnp.min(outside + inside, axis=-1)                # (...,)


def identity_fk(coords):
    """`fk` for a workspace-coordinate field: the configuration point is the
    one and only body sphere centre. `(..., 3)` in, `(..., 1, 3)` out."""
    return jnp.asarray(coords)[..., None, :3]


def spheres_speed_fn(
        fk_fn: Callable[[jnp.ndarray], jnp.ndarray],
        sphere_radii,
        centers,
        half_extents,
        *,
        margin: float,
        min_speed: float = 0.1,
        smoothstep: bool = False,
        frame_offset=None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build `speed_fn(coords)` for `BoxDomain` from the scene geometry.

    Args:
        fk_fn: `coords (..., dq) -> sphere centres (..., n_spheres, 3)`, in
            the SAME frame as `centers` (a `frame_offset` is subtracted from
            the FK output, so pass it when the FK frame and the obstacle
            frame differ).
        sphere_radii: `(n_spheres,)` collision-sphere radii, matching
            `fk_fn`'s output order.
        centers, half_extents: `(K, 3)` obstacle boxes.
        margin: clearance (m) at which the speed reaches free-space `1`.
        min_speed: floor speed at/inside an obstacle (the reference uses
            `0.1`). Must be `> 0` so the eikonal `sqrt`/`1/S` terms stay
            finite.
        smoothstep: shape the `[0, 1]` ramp with `3t^2 - 2t^3` for a `C1`
            field before the floor clip.
        frame_offset: translation subtracted from every FK sphere centre
            before the distance query.
    """
    radii = jnp.asarray(sphere_radii, dtype=jnp.float32)
    centers_j = jnp.asarray(centers, dtype=jnp.float32)
    half_j = jnp.asarray(half_extents, dtype=jnp.float32)
    shift = (jnp.zeros(3, dtype=jnp.float32) if frame_offset is None
             else jnp.asarray(frame_offset, dtype=jnp.float32))
    margin = float(margin)
    min_speed = float(min_speed)

    def speed_fn(coords):
        sphere_centers = fk_fn(coords) - shift               # (..., n_spheres, 3)
        distance = box_sdf(sphere_centers, centers_j, half_j)  # (..., n_spheres)
        clearance = jnp.min(distance - radii, axis=-1)        # (...,)
        ratio = clearance / margin
        if smoothstep:
            t = jnp.clip(ratio, 0.0, 1.0)
            ratio = t * t * (3.0 - 2.0 * t)
        return jnp.clip(ratio, min_speed, 1.0)

    return speed_fn
