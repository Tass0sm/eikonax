# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based numerical solvers for eikonal and Hamilton-Jacobi-Bellman (HJB) equations on regular grids -- see README.org for the mathematical background: a general N-dimensional Fast Sweeping Method (`eikonax.fsm`), its SE(2) application (`eikonax.se2`, an anisotropic eikonal equation on R^2 x S^1 with a customizable metric matrix), and a learned all-pairs alternative (`eikonax.strategies.ntfields`, a TD-NTFields port).

## Package Management

This project uses [Poetry](https://python-poetry.org/). Python 3.12 is required (pinned to `>=3.12,<3.13`).

```bash
poetry install          # install dependencies
poetry add <pkg>        # add a dependency
poetry run pytest       # run tests
```

## Architecture

Two layers. The **numerics** (`fsm`/`se2`) are the stable, positionally-called core the downstream consumer imports directly. The **strategy layer** (`strategies/`, `domains.py`, `config.py`, `scenarios.py`, `scripts/`) is a uniform `solve(domain, *, <config kwargs>)` + CLI wrapper on top, modelled on `~/phd/research/10-dynamic-hierarchy/dynamic-hierarchy` (`utils/config_utils.py`, `scripts/train.py`).

- `eikonax/fsm.py` -- general N-dimensional Fast Sweeping Method (value iteration / semi-Lagrangian). Takes a per-node metric matrix and a JAX-jittable speed field (obstacles are just `speed_fn(x) ~ 0`); candidate moves are wide-stencil integer grid offsets that can translate AND rotate (or whatever the axes represent) in a single move -- see its own module docstring for why that matters. No opinion on caching or how a caller builds an all-pairs field from repeated single-source solves.
- `eikonax/se2.py` -- thin SE(2)-specific layer on top of `fsm.py`: builds the `(ny, nx, n_theta)` grid/periodicity plumbing and a default "soft preference for the current heading" metric, but accepts any custom metric matrix.
- `eikonax/domains.py` -- the shared environmental-constraint object both strategies take: an axis-aligned box with per-axis periodicity, a `speed_fn`, and a metric. `BoxDomain` serves the continuous (normalized-coordinate) neural side AND, via `.fsm_solver()`, the grid side; `se2_domain(speed_fn, ...)` bakes in `grid_shape = (ny, nx, n_theta)`, `plane_domain` is its 2-D `(ny, nx)` counterpart and `line_domain` a 1-D `(n,)` segment. `DOMAINS` is the `--domain` registry. `clearance_fn` / `obstacle_rects` expose the obstacle description a speed field carries (see `scenarios.py`), for solvers that need distances rather than occupancy.
- `eikonax/strategies/fsm.py` -- `solve(domain, *, radius, source, all_pairs, ...)`: the grid sweep (delegates to `domain.fsm_solver()`), optionally the all-pairs per-source loop.
- `eikonax/strategies/ntfields/` -- `solve(domain, *, objective="td_ntfields", backend="metric_net", <~40 flat kwargs>, progress_fn=None) -> Model`. `td_ntfields.py` is the objective; `backends/metric_net.py` the quasimetric network. `make_config(**overrides)` builds the internal `cfg` namespace (used by tests that call backend/objective internals directly). `roadmap.py` is the optional PRM weak-supervision prior: `roadmap_weight > 0` builds a probabilistic roadmap once and adds `roadmap_weight * mean((T - d_PRM)^2)` (an obstacle-aware anchor, outside the causal curriculum); `roadmap_weight=0` (default) builds nothing.
- `eikonax/strategies/wavefront/` -- `solve(domain, *, source, tol, ...) -> Model`: a single-source field as a partition of unity over LOCAL travel-time models (pure JAX). Each splat is a compact window (the SRM density) whose weight is a local model of the arrival time rather than a scalar, with its slope tied to the speed at its own centre -- so smoothly varying speed and obstacles are handled without ever marching a ray through the speed field. Two layouts:
  - **1-D (`field.py` + `chain.py`)**: quadratic weights `c + p.d + 1/2 d^T H d` in normalized coordinates, chained outward from the source ridge to ridge -- steps sized so `|n''| r^3/6 <= tol`, emission times by a Hermite rule (`O(r^5)`), one-sided window radii. `train.py` refines every parameter (residual + consistency + coverage) but is off by default (`train_steps=0`): it lowers its loss while RAISING the true error on an accurate chain (the residual cannot see level drift). `scripts/wavefront_1d.py`: uniform speed exact with 5 splats; the `slow` patch ~7e-6 with 67 splats vs ~2e-3 for 700 oracle-fitted scalar splats.
  - **N-D (`cones.py` + `grow.py`, 2-D tested)**: the weight is the wavefront itself, a cone from a per-splat VIRTUAL SOURCE (exact at any distance under uniform speed, so windows can be large), in physical coordinates/float64. Layout is a `2^dim`-tree refined until a window fits the obstacle CLEARANCE (`domain.clearance_fn`, from `scenarios`; `2*min_window` must stay under the thinnest obstacle or windows leak through it). Growth is Dijkstra over the splats: a neighbour inherits the wave when the virtual source is visible (conservative advancement along the segment, per splat pair, never per query), and otherwise becomes a fresh source -- which is how diffraction at a corner appears without looking for corners. `scripts/wavefront_2d.py` + `baselines.exact_rects` (visibility graph) score it: `free` is exact (5e-10) with 26 splats and ONE virtual source; `wall`/`gap` reach ~4e-3 RMS, the residue being a near-constant bias over the shadow set by how far the diffraction splat sits from the true corner.
- `eikonax/config.py` -- `add_function_args` / `build_subcommand_parser`: introspect a `solve`/constructor/factory signature (resolving `from __future__ import annotations` string hints and PEP 604 `X | None`) into argparse flags. `IGNORED_KWARGS` skips `domain`, `progress_fn`, and callable constraint plumbing.
- `eikonax/scenarios.py` -- `SCENARIOS` registry of analytic `speed_fn` factories (`free`, `wall`, `gap`, and the smooth `slow` patch) for `--scenario`, each also carrying `.clearance` (distance to the nearest obstacle -- the analytic stand-in for a collision checker's distance query) and `.rects` (the obstacles, for exact references), since a callable can't be a CLI arg.
- `eikonax/scripts/solve.py` -- the CLI (`python -m eikonax.scripts.solve` / `eikonax-solve`): `--strategy` + `--domain` + `--scenario` pick the callables, one combined parser is built from their signatures, `strat.solve(domain, **kw)` runs, the field is written to `--out` (`.npz`).

## Downstream consumer

Built for `po-goc-mpc` (`~/phd/research/11-improved-gocmpc/po-goc-mpc`), pulled in as a local editable `[tool.uv.sources]` path dependency there. That project's `po_goc_mpc/experiments/objectives/` owns the all-pairs source loop, disk caching, and the `edge_cost_fn(a, b)` interpolation wrapper built on top of this solver -- mirroring how it already wraps `skfmm` for the isotropic 2-D case.
