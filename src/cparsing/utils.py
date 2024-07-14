"""Module with utilities for internal use, e.g. the Coord class."""

from typing import TYPE_CHECKING, Optional

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
        lineno = parser.line_position(p)
        col_start, col_end = parser.index_position(p)
        filename = parser.ctx.filename

        assert lineno
        assert col_start

        return cls(lineno, col_start, None, col_end, filename)

    @override
    def __str__(self) -> str:
        line_end = self.line_end if (self.line_end is not None) else "?"
        col_end = self.col_end if (self.col_end is not None) else "?"
        return f"{self.filename} | {self.line_start}:{self.col_start}-{line_end}:{col_end}"
