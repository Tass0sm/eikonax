"""The 1-D wavefront experiment: chained local-model splats vs. scalar SRMs.

    python -m eikonax.scripts.wavefront_1d
    python -m eikonax.scripts.wavefront_1d --depth 0.9 --tol 1e-2 1e-3 --plot wavefront_1d.png

On a `line_domain` with a smooth `slow` patch (and on uniform speed), for
each chain tolerance: the wavefront field as chained (`train_steps=0`) and
after refinement, against the exact field (quadrature). Baselines, all
ORACLE least-squares fits to the exact field:

  - `pu0`   -- scalar-weight normalized SRM on the SAME windows;
  - `ridge` -- the literal ridge-shaped-mother sum on the same windows;
  - `pu0 uniform (k)` -- scalar-weight SRM with k evenly spaced windows,
    for the splat count that accuracy costs;
  - `fsm`   -- the grid sweep on the domain's own grid (interpolated).
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

from ..domains import line_domain
from ..scenarios import SCENARIOS
from ..strategies import fsm as fsm_strategy
from ..strategies import wavefront
from ..strategies.wavefront import baselines


def _errors(T, T_true):
    err = np.abs(np.asarray(T, dtype=np.float64) - T_true)
    return f"{err.max():9.2e} {np.sqrt(np.mean(err ** 2)):9.2e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eikonax.scripts.wavefront_1d")
    ap.add_argument("--n", type=int, default=201, help="line_domain nodes (fsm grid, default source)")
    ap.add_argument("--resolution", type=float, default=0.025)
    ap.add_argument("--center", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--depth", type=float, default=0.7)
    ap.add_argument("--source", type=int, default=None, help="source node (default: centre)")
    ap.add_argument("--tol", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--uniform-k", type=int, nargs="+", default=[20, 70, 200, 700])
    ap.add_argument("--n-eval", type=int, default=4001)
    ap.add_argument("--plot", default=None, help="save a figure of the finest run here")
    args = ap.parse_args(argv)

    scenes = {
        "free": SCENARIOS["free"](),
        "slow": SCENARIOS["slow"](center_y=args.center, radius=args.radius, depth=args.depth),
    }
    source = None if args.source is None else (args.source,)
    last = None
    for name, speed_fn in scenes.items():
        domain = line_domain(speed_fn, n=args.n, resolution=args.resolution)
        X = np.asarray(domain.grid((args.n_eval,)))
        X_phys = np.asarray(domain.from_normalized(jnp.asarray(X)))
        print(f"\n== {name} ==   {'max':>9} {'rms':>9}")

        T_fsm = fsm_strategy.solve(domain, source=source)
        nodes = np.asarray(domain.from_normalized(domain.grid(domain.grid_shape)))[:, 0]
        source_n = None
        for tol in args.tol:
            chained = wavefront.solve(domain, source=source, tol=tol, train_steps=0)
            trained = wavefront.solve(domain, source=source, tol=tol, train_steps=args.train_steps)
            f, p = chained.wavefront, chained.params
            source_n = float(f.source[0])
            T_true = baselines.exact_1d(domain, source_n, X)
            k = chained.num_splats
            print(f"tol {tol:g}: {k} splats")
            print(f"  wavefront (chained)   {_errors(chained.time(X_phys), T_true)}")
            print(f"  wavefront (trained)   {_errors(trained.time(X_phys), T_true)}")
            print(f"  pu0 (same windows)    {_errors(baselines.pu0_fit(f, p, X, T_true), T_true)}")
            print(f"  ridge (same windows)  {_errors(baselines.ridge_fit(f, p, X, T_true), T_true)}")
            last = (name, tol, X_phys[:, 0], T_true, chained, trained)

        T_true = baselines.exact_1d(domain, source_n, X)
        for k in args.uniform_k:
            print(f"  pu0 uniform ({k:4d})    {_errors(baselines.pu0_uniform_fit(X, T_true, k), T_true)}")
        print(f"  fsm ({args.n} nodes)       {_errors(np.interp(X_phys[:, 0], nodes, T_fsm), T_true)}")

    if args.plot and last is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        name, tol, x, T_true, chained, trained = last
        fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        a.plot(x, T_true, "k", lw=2, label="exact")
        a.plot(x, trained.time(x[:, None]), "C1--", label="wavefront (trained)")
        centres = np.concatenate([[chained.source[0]],
                                  np.asarray(chained.domain.from_normalized(chained.params["B"]))[:, 0]])
        a.plot(centres, chained.time(centres[:, None]), "C1|", ms=12, label="splat centres")
        a.set_ylabel("T")
        a.set_xlim(x.min(), x.max())
        a.legend()
        b.semilogy(x, np.abs(chained.time(x[:, None]) - T_true) + 1e-12, "C0", label="chained")
        b.semilogy(x, np.abs(trained.time(x[:, None]) - T_true) + 1e-12, "C1", label="trained")
        b.set_xlabel("x")
        b.set_ylabel("|error|")
        b.legend()
        fig.suptitle(f"{name}, tol {tol:g}, {chained.num_splats} splats")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
