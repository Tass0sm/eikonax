"""The 2-D wavefront experiment: cone splats grown around obstacles.

    python -m eikonax.scripts.wavefront_2d --out wavefront_2d.png
    python -m eikonax.scripts.wavefront_2d --scenarios wall --min-window 0.005

For each scenario, one row of panels:

  1. the field: exact arrival time (filled) with the model's contours over it;
  2. the error against the reference -- exact around the scenarios'
     rectangles (a visibility graph, `baselines.exact_rects`), which is
     sharper than `fsm` on these grids; `fsm`'s own error is printed for
     comparison;
  3. the layout: splat centres sized by window radius, and the VIRTUAL
     SOURCES the growth found (red), which is where diffraction happened.

Prints splat counts, build time and error norms per scenario.
"""

from __future__ import annotations

import argparse
import time

import jax.numpy as jnp
import numpy as np

from ..domains import plane_domain
from ..scenarios import SCENARIOS
from ..strategies import fsm as fsm_strategy
from ..strategies import wavefront
from ..strategies.wavefront import baselines


def run(scenario, args):
    speed_fn = SCENARIOS[scenario]()
    domain = plane_domain(speed_fn, ny=args.ny, nx=args.nx, resolution=args.resolution)
    source_idx = tuple(args.source)
    t0 = time.time()
    model = wavefront.solve(domain, source=source_idx, min_window=args.min_window,
                            max_window=args.max_window, tol=args.tol,
                            value_temperature=args.value_temperature)
    build = time.time() - t0

    shape = (args.n_eval, args.n_eval)
    nodes = np.asarray(domain.from_normalized(domain.grid(shape)))
    T = model.time(nodes)
    exact = baselines.exact_rects(speed_fn.rects, model.source, nodes)
    free = np.isfinite(exact) & (np.asarray(domain.speed_fn(jnp.asarray(nodes))) > 0)
    err = np.where(free, T - exact, np.nan)

    fsm_field = fsm_strategy.solve(domain, source=source_idx, radius=args.radius).ravel()
    fsm_nodes = np.asarray(domain.from_normalized(domain.grid(domain.grid_shape)))
    fsm_exact = baselines.exact_rects(speed_fn.rects, model.source, fsm_nodes)
    fsm_free = (np.isfinite(fsm_exact) & (np.asarray(domain.speed_fn(jnp.asarray(fsm_nodes))) > 0)
                & (fsm_field < fsm_strategy.fsm.OBSTACLE_FILL / 2))  # drop its unreachable sentinel
    fsm_err = fsm_field[fsm_free] - fsm_exact[fsm_free]

    finite = err[np.isfinite(err)]
    stats = {
        "splats": model.num_splats,
        "build_s": round(build, 1),
        "max": float(np.abs(finite).max()),
        "rms": float(np.sqrt((finite ** 2).mean())),
        "min": float(finite.min()),
        "uncovered": int(np.sum(free & ~np.isfinite(err))),
        "fsm_rms": float(np.sqrt((fsm_err ** 2).mean())),
        "fsm_nodes": int(fsm_free.sum()),
    }
    return model, nodes, exact, err, free, stats


def draw(axes, scenario, model, nodes, exact, err, free, stats, args):
    shape = (args.n_eval, args.n_eval)
    Y, X = nodes[:, 0].reshape(shape), nodes[:, 1].reshape(shape)
    masked = lambda v: np.where(free, v, np.nan).reshape(shape)

    a = axes[0]
    a.contourf(X, Y, masked(exact), 24)
    a.contour(X, Y, masked(np.nan_to_num(err) + exact), 24, colors="w", linewidths=0.6)
    a.plot(*model.source[::-1], "r*", ms=12)
    a.set_title(f"{scenario}: exact (fill) + model (lines)")

    a = axes[1]
    lim = max(float(np.nanmax(np.abs(err))), 1e-12)
    im = a.pcolormesh(X, Y, err.reshape(shape), cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.figure.colorbar(im, ax=a)
    a.set_title(f"error (max {stats['max']:.1e}, rms {stats['rms']:.1e})")

    a = axes[2]
    B = np.asarray(model.params["B"])
    R = np.asarray(model.params["R"])
    S = B - np.asarray(model.params["rho"])[:, None] * np.asarray(model.params["p"])
    a.scatter(B[:, 1], B[:, 0], s=np.clip(400 * R ** 2, 0.5, 40), c="C0", alpha=0.45, lw=0)
    uniq = np.unique(np.round(S, 2), axis=0)
    a.plot(uniq[:, 1], uniq[:, 0], "rx", ms=5, mew=1.2)
    a.set_title(f"{stats['splats']} splats, {len(uniq)} virtual sources")

    for a in axes:
        a.set_xlim(nodes[:, 1].min(), nodes[:, 1].max())
        a.set_ylim(nodes[:, 0].min(), nodes[:, 0].max())
        a.set_aspect("equal")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eikonax.scripts.wavefront_2d")
    ap.add_argument("--scenarios", nargs="+", default=["free", "wall", "gap"])
    ap.add_argument("--ny", type=int, default=41)
    ap.add_argument("--nx", type=int, default=41)
    ap.add_argument("--resolution", type=float, default=0.1)
    ap.add_argument("--source", type=int, nargs=2, default=(20, 5))
    ap.add_argument("--min-window", type=float, default=0.02)
    ap.add_argument("--max-window", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--value-temperature", type=float, default=None)
    ap.add_argument("--radius", type=int, default=2, help="fsm stencil radius for the comparison")
    ap.add_argument("--n-eval", type=int, default=161)
    ap.add_argument("--out", default="wavefront_2d.png")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(args.scenarios), 3, figsize=(15, 4.6 * len(args.scenarios)),
                             squeeze=False)
    for row, scenario in enumerate(args.scenarios):
        model, nodes, exact, err, free, stats = run(scenario, args)
        print(f"{scenario}: {stats}", flush=True)
        draw(axes[row], scenario, model, nodes, exact, err, free, stats, args)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
