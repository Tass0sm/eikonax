"""Turn a `solve()` function's keyword arguments into CLI flags.

A strategy's configuration lives in its `solve(domain, *, ...)` signature --
one defaulted keyword argument per knob -- and `eikonax.scripts.solve` builds
its argument parser by introspecting that signature rather than maintaining a
parallel list. Same mechanism as `dynamic-hierarchy`'s
`utils/config_utils.py`, from which this is ported:

  - every keyword argument with a default becomes `--kebab-case-name`;
  - a `bool` default becomes a `store_true` flag, or a `--no-name`
    `store_false` flag when the default is `True`;
  - a `list`/`tuple` annotation becomes `nargs="+"`;
  - the argument type is read from the annotation (`int`/`float`/`str`/`bool`,
    `X | None` unwrapped to `X`), defaulting to `str`.

`domain` and `progress_fn` are positional/callable plumbing, not
configuration, so they are always skipped.
"""

from __future__ import annotations

import argparse
import inspect
import types
from inspect import Parameter
from typing import Callable, Union, get_args, get_origin, get_type_hints

#: `typing.Union[...]` and the PEP 604 `X | None` form have different origins.
_UNION_ORIGINS = (Union, types.UnionType)

#: Parameters that are never configuration: the domain object, the progress
#: callback, and the callable "environmental constraint" plumbing (a
#: `speed_fn` or metric function cannot come off a command line -- a
#: `--scenario` name stands in for it).
IGNORED_KWARGS = ("domain", "progress_fn", "speed_fn", "metric_inv_fn", "metric_at_theta")


def get_function_kwargs(func: Callable, ignored_kwargs=IGNORED_KWARGS):
    """The `(name, Parameter)` pairs of `func` that are genuine, overridable
    keyword arguments: they have a default, are not `**kwargs`, and are not
    in `ignored_kwargs`."""
    return [
        (name, param)
        for name, param in inspect.signature(func).parameters.items()
        if param.default is not Parameter.empty
        and param.kind is not Parameter.VAR_KEYWORD
        and name not in ignored_kwargs
    ]


def override_default_kwargs(func: Callable, args: argparse.Namespace, ignored_kwargs=IGNORED_KWARGS):
    """`{name: value}` for every configurable kwarg of `func`, taken from
    `args` where present and from the signature default otherwise."""
    kwargs = {}
    for name, param in get_function_kwargs(func, ignored_kwargs=ignored_kwargs):
        kwargs[name] = getattr(args, name, param.default)
    return kwargs


def _unwrap_optional(ann):
    """`X | None` / `Optional[X]` -> `X`; everything else unchanged."""
    if get_origin(ann) in _UNION_ORIGINS:
        non_none = [a for a in get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


def resolve_type(ann):
    """The argparse `type=` callable for a parameter annotation. `X | None`
    (and `Optional[X]`) unwrap to `X`; anything not a plain scalar falls
    back to `str`."""
    if ann is inspect.Parameter.empty:
        return str
    if get_origin(ann) in _UNION_ORIGINS:
        non_none = [a for a in get_args(ann) if a is not type(None)]
        return resolve_type(non_none[0]) if len(non_none) == 1 else str
    if ann in (int, float, str, bool):
        return ann
    return str


def add_param_as_arg(parser: argparse.ArgumentParser, name: str, param: Parameter, annotation=None):
    """Add one `--flag` to `parser` for keyword argument `name`. `annotation`
    is the resolved type (from `typing.get_type_hints`, since a module using
    `from __future__ import annotations` stores annotations as strings)."""
    flag = f"--{name.replace('_', '-')}"
    ann = _unwrap_optional(param.annotation if annotation is None else annotation)

    if ann is bool or isinstance(param.default, bool):
        if param.default is True:
            parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action="store_false",
                                help=f"disable {name} (default: true)")
            parser.set_defaults(**{name: True})
        else:
            parser.add_argument(flag, dest=name, action="store_true",
                                help=f"enable {name} (default: false)")
            parser.set_defaults(**{name: False})
        return

    if get_origin(ann) in (list, tuple):
        elem_args = get_args(ann)
        elem_type = resolve_type(elem_args[0]) if elem_args else str
        parser.add_argument(flag, type=elem_type, nargs="+", default=param.default, required=False)
        return

    parser.add_argument(flag, type=resolve_type(ann), default=param.default, required=False)


def add_function_args(parser: argparse.ArgumentParser, fn: Callable, ignored_kwargs=IGNORED_KWARGS) -> list[str]:
    """Add one flag to `parser` per configurable keyword argument of `fn`.
    Returns the argument names added, so a caller with several `fn`s in one
    parser can split the parsed namespace back out by owner."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    names = []
    for arg_name, param in get_function_kwargs(fn, ignored_kwargs=ignored_kwargs):
        add_param_as_arg(parser, arg_name, param, annotation=hints.get(arg_name))
        names.append(arg_name)
    return names


def build_subcommand_parser(name: str, fn: Callable, ignored_kwargs=IGNORED_KWARGS) -> argparse.ArgumentParser:
    """An `add_help=False` parser with one flag per configurable keyword
    argument of `fn` (a strategy's `solve`, a domain constructor, a scenario
    factory)."""
    parser = argparse.ArgumentParser(description=f"options for {name}", add_help=False)
    add_function_args(parser, fn, ignored_kwargs=ignored_kwargs)
    return parser
