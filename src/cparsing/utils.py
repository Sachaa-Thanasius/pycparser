"""Module with utilities for internal use, e.g. the Coord class."""

from collections.abc import Callable
from functools import reduce
from string import Template
from types import FunctionType
from typing import TYPE_CHECKING, Optional, Protocol

from sly.yacc import YaccProduction

from ._cluegen import Datum
from ._typing_compat import Self, override


if TYPE_CHECKING:
    from .c_parser import CParser

__all__ = ("Coord",)


class Coord(Datum):
    line_start: int
    col_start: int
    line_end: Optional[int] = None
    col_end: Optional[int] = None
    filename: str = "<unknown>"

    @classmethod
    def from_prod(cls, p: YaccProduction, parser: "CParser") -> Self:
        return cls(p.lineno, p.index, filename=parser.ctx.filename)

    @classmethod
    def from_node(cls, p: object, parser: "CParser") -> Self:
        lineno, (col_start, col_end) = parser.line_position(p), parser.index_position(p)
        filename = parser.ctx.filename

        assert lineno
        assert col_start

        return cls(lineno, col_start, None, col_end, filename)

    @override
    def __str__(self) -> str:
        line_end = self.line_end if (self.line_end is not None) else "?"
        col_end = self.col_end if (self.col_end is not None) else "?"
        return f"{self.filename} | {self.line_start}:{self.col_start} - {line_end}:{col_end}"


class SubstituteDecorator(Protocol):
    def __call__(self, sub: dict[str, str], *extra: dict[str, str]) -> Callable[[Callable[..., object]], None]: ...


class substitute:
    def __init__(self, sub: dict[str, str], *extra: dict[str, str], namespace: dict[str, object]) -> None:
        """Save the given substitution maps for use with a template rule function."""

        self.substitutions = [sub, *extra]
        self.namespace = namespace

    def __call__(self, func: Callable[..., object], /) -> None:
        """Create new rule functions based on the given template rule function and substitution maps, then put them in
        the given namespace.
        """

        for sub in self.substitutions:
            # Copy the template rule function with a new name.
            new_name = reduce(lambda nm, sb: nm.replace(sb[0], sb[1]), sub.items(), func.__name__)
            new_func = self._copy_func_with_name(func, new_name)

            # Perform substitution for the template rules and put them on the new function.
            new_func.rules = [Template(rule_templ).substitute(sub) for rule_templ in reversed(func.rules)]  # pyright: ignore

            # Put the new rule function in the given namespace.
            self.namespace[new_name] = new_func

    def __set_name__(self, owner: type, name: str, /) -> None:
        """Remove the leftover rule template variable from the class."""

        delattr(owner, name)

    @staticmethod
    def _copy_func_with_name(func: Callable[..., object], new_name: str) -> FunctionType:
        """Copy a function object but change the name."""

        new_func = FunctionType(func.__code__, func.__globals__, new_name, func.__defaults__, func.__closure__)
        if func.__annotations__:
            new_func.__annotations__ = func.__annotations__.copy()
        if func.__kwdefaults__ is not None:  # pyright: ignore # It can be None.
            new_func.__kwdefaults__ = func.__kwdefaults__.copy()
        if func.__dict__:
            new_func.__dict__ = func.__dict__.copy()

        return new_func
