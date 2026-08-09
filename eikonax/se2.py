"""Single-source anisotropic eikonal equation on R^2 x S^1 -- see this
package's README.org for the mathematical background (the metric, why
value iteration rather than a finite-difference Hamiltonian stencil).

State space: `(x, y, theta)`, `theta` periodic. Local cost of moving in a
world-frame direction `phi` off the CURRENT heading `theta`:

    speed(phi) = sqrt(cos(phi)^2 + xi^2 * sin(phi)^2)   -- 1 when phi=0
                                                            (straight ahead),
                                                            xi when phi=+-90
                                                            deg (sideways)
    speed(turn) = xi                                    -- same ratio for
                                                            pure rotation

`xi` in (0, 1]; `xi = 1` is isotropic (uniform speed regardless of facing).

Solved via value iteration (Jacobi-style Bellman fixed point): every node's
arrival time is repeatedly relaxed to the minimum, over two disjoint
candidate-move families, of "move cost + interpolated arrival time at the
move's target" (see `Solver.solve`'s own docstring for the exact update).
Starts from a large sentinel everywhere except the source (pinned at 0) and
decreases monotonically to the converged field.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

# Arrival times accumulate over many sweep iterations; JAX's default
# float32 loses precision fast over that many additions. Same convention
# goc-mpc's own time_to_go_field.py uses for its gradient-descent tracer.
jax.config.update("jax_enable_x64", True)

#: Sentinel for both the initial "unknown" value and obstacle-touching
#: cells (pinned back to this after every sweep). Must be finite -- see
#: po_goc_mpc.experiments.objectives.fmm.OBSTACLE_FILL's own docstring for
#: why (linear interpolation against a literal inf poisons any blend with
#: nonzero weight on an obstacle cell). Same magnitude as that module's
#: constant, for consistency, though this package has no dependency on it.
OBSTACLE_FILL = 1.0e4


class Solver:
    """A compiled solver for one FIXED `(mask, resolution, n_theta, xi,
    n_directions)` grid+metric combination -- build once, then call
    `.solve(source)` for as many different source states as needed without
    triggering a fresh JIT compilation each time (`source` is a traced
    argument to the compiled sweep, not baked in at trace time). Use this
    directly instead of the module-level `solve()` convenience function
    whenever solving many sources against the same grid -- e.g. building an
    all-pairs field, which is exactly what a single-source solve baking
    `source` in at trace time would make prohibitively slow (one full
    recompilation per source).
    """

    def __init__(
            self,
            mask: np.ndarray,
            resolution: float,
            n_theta: int,
            xi: float,
            n_directions: int = 16,
            obstacle_fill: float = OBSTACLE_FILL,
    ):
        mask = np.asarray(mask, dtype=bool)
        self.mask = mask
        self.ny, self.nx = mask.shape
        self.resolution = resolution
        self.n_theta = n_theta
        self.xi = xi
        self.n_directions = n_directions
        self.obstacle_fill = obstacle_fill

        thetas = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
        phis = 2.0 * jnp.pi * jnp.arange(n_directions) / n_directions          # (n_directions,)
        translate_speed = jnp.sqrt(jnp.cos(phis) ** 2 + (xi ** 2) * jnp.sin(phis) ** 2)
        translate_cost = resolution / translate_speed                          # (n_directions,)

        world_angles = thetas[:, None] + phis[None, :]                         # (n_theta, n_directions)
        row_offsets = jnp.sin(world_angles)
        col_offsets = jnp.cos(world_angles)

        h_theta = 2.0 * jnp.pi / n_theta
        rotate_cost = h_theta / xi

        mask_j = jnp.asarray(mask)
        grid_row, grid_col = jnp.meshgrid(jnp.arange(self.ny), jnp.arange(self.nx), indexing="ij")

        def translate_candidate(u_layer: jnp.ndarray, k: int, m: int) -> jnp.ndarray:
            target_row = grid_row + row_offsets[k, m]
            target_col = grid_col + col_offsets[k, m]
            coords = jnp.stack([target_row.ravel(), target_col.ravel()])
            # mode="nearest": a translate step off the grid edge clamps to
            # the boundary value instead of wrapping (xy is NOT periodic,
            # unlike theta) -- same convention fmm.make_fmm_edge_cost_fn
            # uses.
            interp = map_coordinates(u_layer, coords, order=1, mode="nearest")
            return interp.reshape(self.ny, self.nx) + translate_cost[m]

        def sweep(u: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
            layers = []
            for k in range(n_theta):
                u_layer_k = u[:, :, k]
                cand = u_layer_k
                for m in range(n_directions):
                    cand = jnp.minimum(cand, translate_candidate(u_layer_k, k, m))
                # Rotate candidates: exact grid lookup (theta already
                # discrete), periodic wrap via modular indexing.
                cand = jnp.minimum(cand, u[:, :, (k + 1) % n_theta] + rotate_cost)
                cand = jnp.minimum(cand, u[:, :, (k - 1) % n_theta] + rotate_cost)
                layers.append(cand)
            new_u = jnp.stack(layers, axis=2)
            new_u = jnp.where(mask_j[:, :, None], obstacle_fill, new_u)
            return new_u.at[source[0], source[1], source[2]].set(0.0)

        self._sweep = jax.jit(sweep)

    def solve(self, source: tuple[int, int, int], n_iters: int = 300, tol: float = 1e-5) -> np.ndarray:
        """Returns `field`, shape `(ny, nx, n_theta)`: `field[i, j, k]` is
        the arrival time from state `source` (grid indices `(sy, sx, sk)`,
        NOT physical coordinates -- same convention `skfmm.travel_time`'s
        own single-point-source `phi` array uses) to state `(x[j], y[i],
        theta[k])`, `theta[k] = 2*pi*k/n_theta`. See class docstring for
        why this is cheap to call repeatedly with different `source`s.
        """
        sy, sx, sk = source
        if not (0 <= sy < self.ny and 0 <= sx < self.nx and 0 <= sk < self.n_theta):
            raise ValueError(f"source {source} out of bounds for grid ({self.ny}, {self.nx}, {self.n_theta})")
        if self.mask[sy, sx]:
            raise ValueError(f"source {source} is inside an obstacle cell")

        source_arr = jnp.asarray(source, dtype=jnp.int32)
        u = jnp.full((self.ny, self.nx, self.n_theta), self.obstacle_fill, dtype=jnp.float64)
        u = u.at[sy, sx, sk].set(0.0)

        for _ in range(n_iters):
            new_u = self._sweep(u, source_arr)
            diff = float(jnp.max(jnp.abs(new_u - u)))
            u = new_u
            if diff < tol:
                break

        return np.asarray(u)


def solve(
        mask: np.ndarray,
        resolution: float,
        n_theta: int,
        xi: float,
        source: tuple[int, int, int],
        n_directions: int = 16,
        n_iters: int = 300,
        tol: float = 1e-5,
        obstacle_fill: float = OBSTACLE_FILL,
) -> np.ndarray:
    """Convenience wrapper for a single solve -- builds a fresh `Solver`
    (see its docstring for the args) and calls `.solve(source, n_iters,
    tol)` on it. Prefer `Solver` directly when solving many sources against
    the same `(mask, resolution, n_theta, xi, n_directions)` grid -- e.g.
    an all-pairs field -- to avoid recompiling the sweep once per source.
    """
    solver = Solver(mask, resolution, n_theta, xi, n_directions, obstacle_fill)
    return solver.solve(source, n_iters, tol)
