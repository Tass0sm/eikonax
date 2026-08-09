# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based numerical solvers for eikonal and Hamilton-Jacobi-Bellman (HJB) equations on regular grids -- see README.org for the mathematical background: a general N-dimensional Fast Sweeping Method (`eikonax.fsm`), and its SE(2) application (`eikonax.se2`, an anisotropic eikonal equation on R^2 x S^1 with a customizable metric matrix).

## Package Management

This project uses [Poetry](https://python-poetry.org/). Python 3.12 is required (pinned to `>=3.12,<3.13`).

```bash
poetry install          # install dependencies
poetry add <pkg>        # add a dependency
poetry run pytest       # run tests
```

## Architecture

- `eikonax/fsm.py` -- general N-dimensional Fast Sweeping Method (value iteration / semi-Lagrangian). Takes a per-node metric matrix and a JAX-jittable speed field (obstacles are just `speed_fn(x) ~ 0`); candidate moves are wide-stencil integer grid offsets that can translate AND rotate (or whatever the axes represent) in a single move -- see its own module docstring for why that matters. No opinion on caching or how a caller builds an all-pairs field from repeated single-source solves.
- `eikonax/se2.py` -- thin SE(2)-specific layer on top of `fsm.py`: builds the `(ny, nx, n_theta)` grid/periodicity plumbing and a default "soft preference for the current heading" metric, but accepts any custom metric matrix.

## Downstream consumer

Built for `po-goc-mpc` (`~/phd/research/11-improved-gocmpc/po-goc-mpc`), pulled in as a local editable `[tool.uv.sources]` path dependency there. That project's `po_goc_mpc/experiments/objectives/` owns the all-pairs source loop, disk caching, and the `edge_cost_fn(a, b)` interpolation wrapper built on top of this solver -- mirroring how it already wraps `skfmm` for the isotropic 2-D case.
