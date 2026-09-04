"""The kwargs -> CLI-flag introspection (`eikonax.config`)."""

from eikonax.config import (
    add_function_args,
    build_subcommand_parser,
    get_function_kwargs,
    resolve_type,
)
from eikonax.domains import se2_domain
from eikonax.scenarios import wall
from eikonax.strategies import fsm, ntfields


def _flags(parser):
    return {opt for a in parser._actions for opt in a.option_strings}


def test_ignored_kwargs_and_no_default_params_are_skipped():
    names = [n for n, _ in get_function_kwargs(se2_domain)]
    assert "speed_fn" not in names  # no default -> supplied by the caller
    assert "metric_inv_fn" not in names  # callable plumbing -> ignored
    assert {"ny", "nx", "resolution", "n_theta", "xi_lateral"} <= set(names)


def test_fsm_solve_flags():
    p = build_subcommand_parser("fsm", fsm.solve)
    flags = _flags(p)
    assert "--radius" in flags and "--n-iters" in flags and "--tol" in flags
    assert "--all-pairs" in flags  # bool default False -> store_true
    # source: tuple[int, ...] | None -> nargs='+', int elements
    (source_action,) = [a for a in p._actions if a.dest == "source"]
    assert source_action.nargs == "+" and source_action.type is int


def test_bool_true_default_becomes_a_no_flag():
    p = build_subcommand_parser("ntfields", ntfields.solve)
    flags = _flags(p)
    assert "--no-rollback" in flags and "--rollback" not in flags  # default True
    assert "--detach-causal" in flags  # default False
    assert "--epochs" in flags and "--lr" in flags and "--td-step" in flags


def test_optional_int_resolves_to_int():
    # n_freq: int | None = None
    p = build_subcommand_parser("ntfields", ntfields.solve)
    (n_freq_action,) = [a for a in p._actions if a.dest == "n_freq"]
    assert n_freq_action.type is int


def test_resolve_type_unwraps_optional():
    assert resolve_type(int | None) is int
    assert resolve_type(float) is float
    assert resolve_type(None.__class__) is str  # NoneType -> fallback


def test_add_function_args_reports_names_for_one_combined_parser():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    strat_names = add_function_args(parser, fsm.solve)
    domain_names = add_function_args(parser, se2_domain)
    scenario_names = add_function_args(parser, wall)
    assert set(strat_names).isdisjoint(domain_names)
    assert "wall_x" in scenario_names
    args = parser.parse_args(["--radius", "3", "--ny", "9", "--wall-x", "1.5"])
    assert args.radius == 3 and args.ny == 9 and args.wall_x == 1.5
