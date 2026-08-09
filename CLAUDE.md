# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based numerical solvers for eikonal and Hamilton-Jacobi-Bellman (HJB) equations on regular grids -- see README.org for the mathematical background and the first solver (`eikonax.se2`, an anisotropic eikonal equation on R^2 x S^1).

## Package Management

This project uses [Poetry](https://python-poetry.org/). Python 3.12 is required (pinned to `>=3.12,<3.13`).

```bash
poetry install          # install dependencies
poetry add <pkg>        # add a dependency
poetry run pytest       # run tests
```

## Architecture

- `eikonax/se2.py` -- single-source anisotropic eikonal solver over `(x, y, theta)` state space (value iteration / semi-Lagrangian fast sweeping). Plain function of numpy/JAX arrays; no opinion on caching or how a caller builds an all-pairs field from repeated single-source solves.

## Downstream consumer

Built for `po-goc-mpc` (`~/phd/research/11-improved-gocmpc/po-goc-mpc`), pulled in as a local editable `[tool.uv.sources]` path dependency there. That project's `po_goc_mpc/experiments/objectives/` owns the all-pairs source loop, disk caching, and the `edge_cost_fn(a, b)` interpolation wrapper built on top of this solver -- mirroring how it already wraps `skfmm` for the isotropic 2-D case.
