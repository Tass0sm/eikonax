# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based numerical solvers for eikonal and Hamilton-Jacobi-Bellman (HJB) equations on regular grids -- see README.org for the mathematical background: a general N-dimensional Fast Sweeping Method (`eikonax.fsm`), its SE(2) application (`eikonax.se2`, an anisotropic eikonal equation on R^2 x S^1 with a customizable metric matrix), and a learned all-pairs alternative (`eikonax.strategies.ntfields`, a TD-NTFields port).

## Package Management

This project uses [uv](https://docs.astral.sh/uv/) (hatchling build backend). Python 3.12 is required (pinned to `>=3.12,<3.13`).

```bash
uv sync --extra gpu --extra splats   # install: CUDA JAX + srms (needed by the huygens strategy)
uv add <pkg>                         # add a dependency
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest   # run tests (ROS on PYTHONPATH injects a broken pytest plugin)
```

Extras: `gpu` = `jax[cuda12]`; `splats` = `srms` from `~/phd/research/17-subriemannian-ntfields/splat-regression-modeling` (an editable `[tool.uv.sources]` path dependency; heavy -- it brings that project's mlflow/flax/jaxkan dependencies along). Without `splats`, `eikonax.strategies` simply omits `huygens`.

## Architecture

Two layers. The **numerics** (`fsm`/`se2`) are the stable, positionally-called core the downstream consumer imports directly. The **strategy layer** (`strategies/`, `domains.py`, `config.py`, `scenarios.py`, `scripts/`) is a uniform `solve(domain, *, <config kwargs>)` + CLI wrapper on top, modelled on `~/phd/research/10-dynamic-hierarchy/dynamic-hierarchy` (`utils/config_utils.py`, `scripts/train.py`).

- `eikonax/fsm.py` -- general N-dimensional Fast Sweeping Method (value iteration / semi-Lagrangian). Takes a per-node metric matrix and a JAX-jittable speed field (obstacles are just `speed_fn(x) ~ 0`); candidate moves are wide-stencil integer grid offsets that can translate AND rotate (or whatever the axes represent) in a single move -- see its own module docstring for why that matters. No opinion on caching or how a caller builds an all-pairs field from repeated single-source solves.
- `eikonax/se2.py` -- thin SE(2)-specific layer on top of `fsm.py`: builds the `(ny, nx, n_theta)` grid/periodicity plumbing and a default "soft preference for the current heading" metric, but accepts any custom metric matrix.
- `eikonax/domains.py` -- the shared environmental-constraint object both strategies take: an axis-aligned box with per-axis periodicity, a `speed_fn`, and a metric. `BoxDomain` serves the continuous (normalized-coordinate) neural side AND, via `.fsm_solver()`, the grid side; `se2_domain(speed_fn, ...)` bakes in `grid_shape = (ny, nx, n_theta)`; `plane_domain` is its 2-D `(ny, nx)` counterpart. `DOMAINS` is the `--domain` registry.
- `eikonax/strategies/fsm.py` -- `solve(domain, *, radius, source, all_pairs, ...)`: the grid sweep (delegates to `domain.fsm_solver()`), optionally the all-pairs per-source loop.
- `eikonax/strategies/ntfields/` -- `solve(domain, *, objective="td_ntfields", backend="metric_net", <~40 flat kwargs>, progress_fn=None) -> Model`. `td_ntfields.py` is the objective; `backends/metric_net.py` the quasimetric network. `make_config(**overrides)` builds the internal `cfg` namespace (used by tests that call backend/objective internals directly). `roadmap.py` is the optional PRM weak-supervision prior: `roadmap_weight > 0` builds a probabilistic roadmap once and adds `roadmap_weight * mean((T - d_PRM)^2)` (an obstacle-aware anchor, outside the causal curriculum); `roadmap_weight=0` (default) builds nothing.
- `eikonax/strategies/huygens/` -- `solve(domain, *, source, ...) -> Model`: a single-source field as a min over straight-ray "wavelets" (Huygens' principle; an `srms` `SplatModel` with a ray-travel-time mother and a min combine). Starts from the source wavelet alone (exact for uniform speed), greedily adds wavelets that most lower `mean T` (diffraction sources at obstacle corners/gaps), refines their centres by minimizing `mean T` (an upper bound -- every value is a real path length), and prunes redundant ones. Emission times are never trained: `HuygensField.values` recomputes them as the Bellman fixed point (hard Bellman-Ford + a differentiable unroll along the shortest-path tree). `wavelets.py` is the model, `train.py` grow/refine/prune. Exact on the 2-D `wall`/`gap` scenes to a few 1e-3 with 2-3 wavelets; poor on SE(2), where the heading-dependent metric curves geodesics and straight rays are only feasible paths (open problem). Registered only when `srms` is importable.
- `eikonax/config.py` -- `add_function_args` / `build_subcommand_parser`: introspect a `solve`/constructor/factory signature (resolving `from __future__ import annotations` string hints and PEP 604 `X | None`) into argparse flags. `IGNORED_KWARGS` skips `domain`, `progress_fn`, and callable constraint plumbing.
- `eikonax/scenarios.py` -- `SCENARIOS` registry of analytic `speed_fn` factories (`free`, `wall`, `gap`) for `--scenario`, since a callable can't be a CLI arg.
- `eikonax/scripts/solve.py` -- the CLI (`python -m eikonax.scripts.solve` / `eikonax-solve`): `--strategy` + `--domain` + `--scenario` pick the callables, one combined parser is built from their signatures, `strat.solve(domain, **kw)` runs, the field is written to `--out` (`.npz`).

## Downstream consumer

Built for `po-goc-mpc` (`~/phd/research/11-improved-gocmpc/po-goc-mpc`), pulled in as a local editable `[tool.uv.sources]` path dependency there. That project's `po_goc_mpc/experiments/objectives/` owns the all-pairs source loop, disk caching, and the `edge_cost_fn(a, b)` interpolation wrapper built on top of this solver -- mirroring how it already wraps `skfmm` for the isotropic 2-D case.
