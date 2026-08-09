"""General N-dimensional Fast Sweeping Method (value iteration / Bellman
fixed-point, semi-Lagrangian) for a static eikonal/Hamilton-Jacobi equation
under an arbitrary, per-node Riemannian metric matrix `G(x)` AND an
arbitrary, JAX-jittable speed field `speed_fn(x)` -- generalizes what
`se2.py` used to hand-derive as SE(2)-specific "translate xor rotate"
candidate moves into one general mechanism any manifold/metric can be
built on top of (see `se2.py` for the SE(2) application, with a
customizable metric matrix).

Candidate moves: every INTEGER grid-offset vector within a max radius per
axis (excluding the zero vector) -- e.g. for a 3-D grid, `radius=2` gives
all of `{-2,...,2}^3 \\ {0}`, 124 directions. Every such offset lands
EXACTLY on a grid-neighbor node (no interpolation needed anywhere, unlike a
scheme sampling arbitrary real-valued angles), and -- this is the actual
point of this refactor -- an offset can be nonzero in MULTIPLE axes at
once, e.g. simultaneously translating AND rotating in a single candidate
move. A scheme that only ever offers "translate-only" or "rotate-only"
moves (an earlier version of this package) cannot represent that at all,
which is why it could never produce a shortest path that actually turns
into its direction of travel: any use of rotation always paid its full
cost immediately, with no way to blend it into forward progress. Larger
`radius` gives finer angular resolution at the cost of `(2*radius+1)^n`
candidates -- exponential in dimension, so keep it modest beyond 2-3 axes.

Obstacles, speed field: rather than a binary obstacle mask, `Solver` takes
an arbitrary `speed_fn(coords) -> speed` -- a JAX-jittable function of
physical coordinates (obstacles are just `speed_fn(x) ~ 0` there; anything
in between is slow terrain, a soft margin, whatever the caller wants).
Evaluated once, at every grid node, at construction time (a static field --
this package has no notion of a time-varying domain). The cost of taking
offset `o` from node `X` (metric length `sqrt(disp^T G(X) disp)`, `disp =
o * resolution`) is divided by the WORSE (lower) of `X`'s own speed and the
offset target's speed -- so a move touching a zero-speed node anywhere
costs `inf`, naturally keeping the value iteration from ever finding a
cheap path through an obstacle without needing a separate masking/pinning
step. `inf`/`nan` are clipped back to `obstacle_fill` (a finite sentinel)
after every sweep, so the returned/cached field and the convergence check
both stay numerically well-behaved -- see `fmm.OBSTACLE_FILL`'s own
docstring in `po_goc_mpc.experiments.objectives.fmm` for why a literal
`inf` is a problem for anything downstream that linearly interpolates this
field.

Solved via value iteration (Jacobi-style Bellman fixed point): every
node's arrival time is repeatedly relaxed to the minimum, over every
candidate offset, of "move cost + value at the offset target" (read from
the PREVIOUS full-grid iterate). Non-periodic axes clamp at the boundary
(Neumann-like); periodic axes wrap. Starts from a large sentinel
everywhere except the source (pinned at 0) and decreases monotonically to
the converged field.
"""

import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

# Arrival times accumulate over many sweep iterations; JAX's default
# float32 loses precision fast over that many additions. Same convention
# goc-mpc's own time_to_go_field.py uses for its gradient-descent tracer.
jax.config.update("jax_enable_x64", True)

#: Sentinel for both the initial "unknown" value everywhere and wherever
#: the raw update comes out non-finite (a move touching a zero-speed node
#: -- see module docstring). Must be finite -- see
#: po_goc_mpc.experiments.objectives.fmm.OBSTACLE_FILL's own docstring for
#: why. Same magnitude as that module's constant, for consistency, though
#: this package has no dependency on it.
OBSTACLE_FILL = 1.0e4


def default_offsets(n: int, radius: int) -> list[tuple[int, ...]]:
    """Every integer offset vector in `{-radius,...,radius}^n`, excluding
    the all-zero vector -- see module docstring."""
    axis_range = range(-radius, radius + 1)
    return [o for o in itertools.product(axis_range, repeat=n) if any(o)]


def _segment_lattice_points(offset: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Integer lattice points along the straight segment from the origin
    to `offset` (inclusive of both ends), via simple parametric rounding
    (a DDA/Bresenham-style line rasterization, generalized to N-D).

    Needed because a `radius > 1` offset can otherwise "tunnel" through a
    thin (sub-radius) low-speed region without ever landing on it -- a
    real bug caught directly by this package's own tests: a 1-cell-wide
    wall, with `radius=3` candidate moves, was getting jumped clean over,
    since only the two move ENDPOINTS were being checked for speed. Every
    intermediate lattice point the move passes near now gets checked too."""
    k = max(abs(c) for c in offset)
    if k == 0:
        return [offset]
    points = []
    seen = set()
    for j in range(k + 1):
        t = j / k
        pt = tuple(int(round(t * c)) for c in offset)
        if pt not in seen:
            seen.add(pt)
            points.append(pt)
    return points


def _build_coords(grid_shape: tuple[int, ...], resolutions: jnp.ndarray, origin: jnp.ndarray) -> jnp.ndarray:
    """`(*grid_shape, n)` physical-coordinate array: `coords[idx] = origin
    + idx * resolutions` (elementwise), the array `speed_fn` is evaluated
    against."""
    axes = [origin[i] + resolutions[i] * jnp.arange(grid_shape[i]) for i in range(len(grid_shape))]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack(mesh, axis=-1)


class Solver:
    """A compiled Fast Sweeping solver for one FIXED `(grid_shape,
    resolutions, periodic, metric, speed_fn, origin, offsets)`
    grid/metric/speed combination -- build once, then call `.solve(source)`
    for as many different source states as needed without triggering a
    fresh JIT compilation each time (`source` is a traced argument to the
    compiled sweep, not baked in at trace time -- needed to make an
    all-pairs field build tractable; see `se2.py`'s own module docstring
    for why this mattered before).

    Args:
        grid_shape: length-`n` tuple, grid size per axis.
        resolutions: length-`n` sequence of per-axis physical grid spacing
            (e.g. metres for a translation axis, radians for an angular
            one -- axes need not share units).
        periodic: length-`n` sequence of bool, whether each axis wraps
            (e.g. True for an S^1 heading axis, False for a bounded
            translation axis).
        metric: `(*grid_shape, n, n)` array -- the (symmetric positive
            definite) Riemannian metric matrix at every grid node. Built
            however the caller likes; see `se2.py` for the SE(2)
            construction, with a customizable metric matrix.
        speed_fn: `coords -> speed`, a JAX-jittable function of physical
            coordinates (shape `(..., n)` in, `(...,)` out) -- see module
            docstring for how this replaces a binary obstacle mask.
        origin: length-`n` sequence, physical coordinate of grid index 0
            on each axis. Defaults to all zeros.
        offsets: explicit list of integer grid-offset tuples (each length
            `n`, not all zero) to use as candidate moves. Defaults to
            `default_offsets(n, radius)` if not given.
        radius: only used to auto-generate `offsets` when it's not given.
        obstacle_fill: sentinel for the initial "unknown" value and for
            clipping non-finite updates -- see module docstring.
    """

    def __init__(
            self,
            grid_shape: tuple[int, ...],
            resolutions,
            periodic,
            metric: np.ndarray,
            speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
            origin=None,
            offsets: list[tuple[int, ...]] | None = None,
            radius: int = 2,
            obstacle_fill: float = OBSTACLE_FILL,
    ):
        self.grid_shape = tuple(grid_shape)
        self.n = len(self.grid_shape)
        self.resolutions = np.asarray(resolutions, dtype=float)
        self.periodic = list(periodic)
        self.obstacle_fill = obstacle_fill
        if self.resolutions.shape != (self.n,):
            raise ValueError(f"resolutions must have length {self.n}, got shape {self.resolutions.shape}")
        if len(self.periodic) != self.n:
            raise ValueError(f"periodic must have length {self.n}, got {len(self.periodic)}")
        metric = np.asarray(metric, dtype=float)
        if metric.shape != (*self.grid_shape, self.n, self.n):
            raise ValueError(
                f"metric must have shape {(*self.grid_shape, self.n, self.n)}, got {metric.shape}")
        origin = np.zeros(self.n) if origin is None else np.asarray(origin, dtype=float)
        if origin.shape != (self.n,):
            raise ValueError(f"origin must have length {self.n}, got shape {origin.shape}")

        if offsets is None:
            offsets = default_offsets(self.n, radius)
        if not offsets:
            raise ValueError("offsets must be non-empty")
        self.offsets = offsets
        # Per-axis padding: the max magnitude any offset uses on that axis
        # -- covers every candidate's neighbor lookup with one pad, reused
        # across all offsets every sweep.
        pad_width = [max(abs(o[i]) for o in offsets) for i in range(self.n)]
        self._pad_width = pad_width

        metric_j = jnp.asarray(metric)
        resolutions_j = jnp.asarray(self.resolutions)
        origin_j = jnp.asarray(origin)

        periodic_local = self.periodic
        n = self.n
        grid_shape_t = self.grid_shape

        def pad(field: jnp.ndarray) -> jnp.ndarray:
            out = field
            for axis in range(n):
                if pad_width[axis] == 0:
                    continue
                pw = [(0, 0)] * n
                pw[axis] = (pad_width[axis], pad_width[axis])
                out = jnp.pad(out, pw, mode="wrap" if periodic_local[axis] else "edge")
            return out

        def shift(padded: jnp.ndarray, o: tuple[int, ...]) -> jnp.ndarray:
            """The shifted-by-`o` view of `padded` -- a STATIC index range
            (`o` is fixed at construction time), so this is plain, cheap
            array slicing, not a gather."""
            sl = tuple(
                slice(pad_width[i] + o[i], pad_width[i] + o[i] + grid_shape_t[i])
                for i in range(n)
            )
            return padded[sl]

        def neighbors(padded: jnp.ndarray) -> jnp.ndarray:
            """Stacks `shift(padded, o)` for every offset into one
            `(n_offsets, *grid_shape)` array."""
            return jnp.stack([shift(padded, o) for o in offsets], axis=0)

        # Speed field: evaluated once, at construction -- see module
        # docstring. jax.jit'd explicitly (not just relying on speed_fn
        # happening to be traceable) since the caller was told this can be
        # an arbitrary JAX-jittable function.
        coords = _build_coords(grid_shape_t, resolutions_j, origin_j)
        speed_field = jax.jit(speed_fn)(coords)
        if speed_field.shape != grid_shape_t:
            raise ValueError(
                f"speed_fn(coords) must return shape {grid_shape_t}, got {speed_field.shape}")
        speed_np = np.asarray(speed_field)
        if np.any(speed_np < 0):
            raise ValueError("speed_fn returned a negative speed somewhere -- speeds must be >= 0")
        self._speed_field = speed_np

        # Cost per offset doesn't depend on u -- precompute once, stacked
        # into one (n_offsets, *grid_shape) array for a single vectorized
        # min-reduction in the sweep below.
        disp = jnp.asarray(offsets, dtype=jnp.float64) * resolutions_j[None, :]  # (n_offsets, n)
        metric_length = jnp.sqrt(jnp.einsum("oi,...ij,oj->o...", disp, metric_j, disp))  # (n_offsets, *grid_shape)

        # Effective speed per offset: the WORST (lowest) speed at any
        # lattice point the move passes through, not just its two
        # endpoints -- see _segment_lattice_points's own docstring for why
        # (tunneling through a thin obstacle otherwise).
        padded_speed = pad(speed_field)
        effective_speed_list = []
        for o in offsets:
            samples = [shift(padded_speed, pt) for pt in _segment_lattice_points(o)]
            eff = samples[0]
            for s in samples[1:]:
                eff = jnp.minimum(eff, s)
            effective_speed_list.append(eff)
        effective_speed = jnp.stack(effective_speed_list, axis=0)  # (n_offsets, *grid_shape)
        # effective_speed == 0 -> cost == inf (division by zero is not an
        # error under IEEE-754 float semantics, and jax doesn't raise on
        # it either) -- exactly the "obstacle" case, with no separate
        # masking step needed.
        costs = metric_length / effective_speed                                 # (n_offsets, *grid_shape)

        def sweep(u: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
            candidates = neighbors(pad(u)) + costs  # (n_offsets, *grid_shape)
            raw = jnp.minimum(u, jnp.min(candidates, axis=0))
            new_u = jnp.where(jnp.isfinite(raw), raw, obstacle_fill)
            return new_u.at[tuple(source[i] for i in range(n))].set(0.0)

        self._sweep = jax.jit(sweep)

    def solve(self, source, n_iters: int = 300, tol: float = 1e-5) -> np.ndarray:
        """Returns `field`, shape `grid_shape`: `field[idx]` is the arrival
        time from state `source` (grid indices, length `n` -- NOT physical
        coordinates, same convention `skfmm.travel_time`'s own
        single-point-source `phi` array uses) to state `idx`. See class
        docstring for why this is cheap to call repeatedly with different
        `source`s.
        """
        source = tuple(int(s) for s in source)
        if len(source) != self.n:
            raise ValueError(f"source must have length {self.n}, got {source}")
        for i, s in enumerate(source):
            if not (0 <= s < self.grid_shape[i]):
                raise ValueError(f"source {source} out of bounds for grid {self.grid_shape}")
        if self._speed_field[source] <= 0.0:
            raise ValueError(f"source {source} has zero speed (inside an obstacle)")

        source_arr = jnp.asarray(source, dtype=jnp.int32)
        u = jnp.full(self.grid_shape, self.obstacle_fill, dtype=jnp.float64)
        u = u.at[source].set(0.0)

        for _ in range(n_iters):
            new_u = self._sweep(u, source_arr)
            diff = float(jnp.max(jnp.abs(new_u - u)))
            u = new_u
            if diff < tol:
                break

        return np.asarray(u)


def solve(
        grid_shape: tuple[int, ...],
        resolutions,
        periodic,
        metric: np.ndarray,
        speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
        source,
        origin=None,
        offsets: list[tuple[int, ...]] | None = None,
        radius: int = 2,
        n_iters: int = 300,
        tol: float = 1e-5,
        obstacle_fill: float = OBSTACLE_FILL,
) -> np.ndarray:
    """Convenience wrapper for a single solve -- builds a fresh `Solver`
    (see its docstring for the args) and calls `.solve(source, n_iters,
    tol)` on it. Prefer `Solver` directly when solving many sources against
    the same grid/metric/speed field -- e.g. an all-pairs field -- to
    avoid recompiling the sweep once per source.
    """
    solver = Solver(grid_shape, resolutions, periodic, metric, speed_fn, origin, offsets, radius, obstacle_fill)
    return solver.solve(source, n_iters, tol)
