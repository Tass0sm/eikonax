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
- `eikonax/domains.py` -- the shared environmental-constraint object both strategies take: an axis-aligned box with per-axis periodicity, a `speed_fn`, and a metric. `BoxDomain` serves the continuous (normalized-coordinate) neural side AND, via `.fsm_solver()`, the grid side; `se2_domain(speed_fn, ...)` bakes in `grid_shape = (ny, nx, n_theta)`. `DOMAINS` is the `--domain` registry.
- `eikonax/strategies/fsm.py` -- `solve(domain, *, radius, source, all_pairs, ...)`: the grid sweep (delegates to `domain.fsm_solver()`), optionally the all-pairs per-source loop.
- `eikonax/strategies/ntfields/` -- `solve(domain, *, objective="td_ntfields", backend="metric_net", <~35 flat kwargs>, progress_fn=None) -> Model`. `td_ntfields.py` is the objective; `backends/metric_net.py` the quasimetric network. `make_config(**overrides)` builds the internal `cfg` namespace (used by tests that call backend/objective internals directly).
- `eikonax/config.py` -- `add_function_args` / `build_subcommand_parser`: introspect a `solve`/constructor/factory signature (resolving `from __future__ import annotations` string hints and PEP 604 `X | None`) into argparse flags. `IGNORED_KWARGS` skips `domain`, `progress_fn`, and callable constraint plumbing.
- `eikonax/scenarios.py` -- `SCENARIOS` registry of analytic `speed_fn` factories (`free`, `wall`, `gap`) for `--scenario`, since a callable can't be a CLI arg.
- `eikonax/scripts/solve.py` -- the CLI (`python -m eikonax.scripts.solve` / `eikonax-solve`): `--strategy` + `--domain` + `--scenario` pick the callables, one combined parser is built from their signatures, `strat.solve(domain, **kw)` runs, the field is written to `--out` (`.npz`).

## Downstream consumer

Built for `po-goc-mpc` (`~/phd/research/11-improved-gocmpc/po-goc-mpc`), pulled in as a local editable `[tool.uv.sources]` path dependency there. That project's `po_goc_mpc/experiments/objectives/` owns the all-pairs source loop, disk caching, and the `edge_cost_fn(a, b)` interpolation wrapper built on top of this solver -- mirroring how it already wraps `skfmm` for the isotropic 2-D case.
