from __future__ import annotations

import io
from collections import deque
from contextlib import contextmanager
from types import GeneratorType
from typing import Any, Generator, Optional, Union

from cparsing.utils import Dataclass


class Coord:
    def __repr__(self):
        return "Coord()"


class AST(Dataclass, kw_only=True, weakref_slot=True):
    coord: Coord | None = None


class Number(AST):
    value: int


class NegNumber(AST):
    value: int


class BinOp(AST):
    op: str
    left: AST
    right: AST


class NumberTuple(AST):
    members: list[Number]


class NodeVisitor:
    def _visit(self, node: AST) -> Generator[Any, Any, Any]:
        result: Any = getattr(self, f"visit_{type(node).__name__}", self.generic_visit)(node)
        if isinstance(result, GeneratorType):
            result = yield from result
        return result

    def visit(self, node: AST) -> Any:
        """Visit a node."""

        stack: deque[Generator[Any, Any, Any]] = deque([self._visit(node)])
        result: Any = None

        while stack:
            try:
                node = stack[-1].send(result)
                stack.append(self._visit(node))
                result = None
            except StopIteration as exc:
                stack.pop()
                result = exc.value

        return result

    def generic_visit(self, node: AST) -> Generator[Any, Any, None]:
        """Called if no explicit visitor function exists for a node."""

        for field_name in node._fields:
            field = getattr(node, field_name)

            if isinstance(field, AST):
                yield field

            elif isinstance(field, list):
                for subfield in field:  # pyright: ignore [reportUnknownVariableType]
                    if isinstance(subfield, AST):
                        yield subfield

        return None


def dump(
    node: AST,
    annotate_fields: bool = True,
    include_coords: bool = False,
    *,
    indent: str | int | None = None,
) -> str:
    """Give a formatted string representation of an AST.

    Parameters
    ----------
    node: AST
        The AST to format as a string.
    annotate_fields: bool, default=True
        Whether the returned string will show the names and the values for fields. If False, the result string will be
        more compact by omitting unambiguous field names. True by default.
    include_coords: bool, default=False
        Whether to display coordinates for each node. False by default.
    indent: int | str | None, optional
        The indent level to pretty-print the tree with. Expects the integer to be positive. None (the default) selects
        the single line representation.

    Returns
    -------
    str
        The formatted string representation of the given AST.
    """

    _missing = object()

    def _format(node: Any, level: int = 0) -> tuple[str, bool]:
        assert isinstance(indent, Optional[str])

        if indent is not None:
            level += 1
            prefix = '\n' + indent * level
            sep = ',\n' + indent * level
        else:
            prefix = ''
            sep = ', '

        if isinstance(node, AST):
            cls = type(node)
            args: list[str] = []
            allsimple = True
            keywords = annotate_fields

            for name in node._fields:
                try:
                    value = getattr(node, name)
                except AttributeError:
                    keywords = True
                    continue
                if value is None and getattr(cls, name, _missing) is None:
                    keywords = True
                    continue
                value, simple = _format(value, level)
                allsimple = allsimple and simple
                if keywords:
                    args.append(f'{name}={value}')
                else:
                    args.append(value)

            if include_coords and node.coord:
                value, simple = _format(node.coord, level)
                allsimple = allsimple and simple
                args.append(f'coord={value}')

            if allsimple and len(args) <= 3:
                return f"{type(node).__name__}({', '.join(args)})", not args

            return f'{type(node).__name__}({prefix}{sep.join(args)})', False

        if isinstance(node, list):
            if not node:
                return '[]', True
            return f'[{prefix}{sep.join(_format(x, level)[0] for x in node)}]', False  # pyright: ignore [reportUnknownVariableType]

        return repr(node), True

    if not isinstance(node, AST):
        raise TypeError(f'expected AST, got {type(node).__name__!r}')

    if indent is not None and not isinstance(indent, str):
        indent = ' ' * indent

    return _format(node)[0]


# ================================================================================
# ================================================================================
# ================================================================================


class Evaluator(NodeVisitor):
    def visit_Number(self, node: Number) -> int:
        return node.value

    def visit_NegNumber(self, node: NegNumber) -> int:
        return node.value

    def visit_NumberTuple(self, node: NumberTuple) -> int:
        return sum(num.value for num in node.members)

    def visit_BinOp(self, node: BinOp) -> Generator[Any, int, Optional[Union[int, float]]]:
        left = yield node.left
        right = yield node.right
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        return None


class Sightseer(NodeVisitor):
    def visit_Number(self, node: Number) -> None:
        print(node.value)

    def visit_NumberTuple(self, node: NumberTuple) -> None:
        print(node.members)

    def visit_BinOp(self, node: BinOp) -> Generator[Any, Any, None]:
        print(node.op)
        return self.generic_visit(node)


class NodeDumper(NodeVisitor):
    def __init__(self, annotate_fields: bool = True, include_coords: bool = False, *, indent: str | None = None):
        self.annotate_fields = annotate_fields
        self.include_coords = include_coords
        self.indent = indent

        self.buffer = io.StringIO()
        self.indent_level = 0

    @property
    def prefix(self) -> str:
        if self.indent is not None:
            prefix = f"\n{self.indent * self.indent_level}"
        else:
            prefix = ""

        return prefix

    @property
    def sep(self) -> str:
        if self.indent is not None:
            sep = f",\n{self.indent * self.indent_level}"
        else:
            sep = ", "
        return sep

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object):
        self.buffer.close()

    @contextmanager
    def add_indent_level(self) -> Generator[None, Any, None]:
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1

    def generic_visit(self, node: AST) -> Generator[Any, Any, None]:
        with self.add_indent_level():
            self.buffer.write(f"{type(node).__name__}({self.prefix}")

            # Determine which fields will be displayed.
            node_fields = node._fields
            if self.include_coords and node.coord:
                node_fields += ("coord",)

            for field_name in node_fields:
                if self.annotate_fields:
                    self.buffer.write(f"{field_name}=")

                field = getattr(node, field_name)

                if isinstance(field, AST):
                    yield field

                elif isinstance(field, list):
                    if not field:
                        self.buffer.write("[]")
                    else:
                        with self.add_indent_level():
                            self.buffer.write(f"[{self.prefix}")

                            for subfield in field:  # pyright: ignore [reportUnknownVariableType]
                                if isinstance(subfield, AST):
                                    yield subfield
                                    self.buffer.write(self.sep)

                            # Remove extra separator.
                            self.buffer.seek(self.buffer.tell() - len(self.sep))
                            self.buffer.truncate()

                        self.buffer.write("]")
                else:
                    self.buffer.write(repr(field))

                self.buffer.write(self.sep)

            # Remove extra separator.
            self.buffer.seek(self.buffer.tell() - len(self.sep))
            self.buffer.truncate()

        self.buffer.write(")")

        return None


def dump2(node: AST, annotate_fields: bool = True, include_coords: bool = False, *, indent: str | None = None) -> str:
    """Give a formatted string representation of an AST.

    Parameters
    ----------
    node: AST
        The AST to format as a string.
    annotate_fields: bool, default=True
        Whether the returned string will show the names and the values for fields. If False, the result string will be
        more compact by omitting unambiguous field names. True by default.
    include_coords: bool, default=False
        Whether to display coordinates for each node. False by default.
    indent: str | None, optional
        The indent level to pretty-print the tree with. None by default, which selects the single line representation.

    Returns
    -------
    str
        The formatted string representation of the given AST.
    """

    with NodeDumper(annotate_fields, include_coords, indent=indent) as visitor:
        visitor.visit(node)
        return visitor.buffer.getvalue()


def make_test_tree(limit: int = 10) -> tuple[AST, int]:
    tree = NumberTuple([Number(i) for i in range(-1, -10, -1)])
    tree = BinOp("+", tree, Number(0))
    tree.coord = Coord()
    for i in range(1, limit):
        tree = BinOp("+", Number(i), tree)
    return tree, sum(range(1, limit)) - 1


def test() -> None:
    import time

    limit = 10
    tree, expected = make_test_tree(limit)
    # Sightseer().visit(tree)

    if limit <= 500:
        print("\n======================================================================\n")
        print("+++ builtin AST dump +++")
        start = time.perf_counter()
        print(dump(tree, indent=4))
        elapsed = time.perf_counter() - start
        print(elapsed)

    print("\n======================================================================\n")

    print("+++ my AST dump +++")
    start = time.perf_counter()
    print(dump2(tree, indent="    "))
    elapsed = time.perf_counter() - start
    print(elapsed)

    print("\n======================================================================\n")

    result = Evaluator().visit(tree)
    print(f"{result=}, {expected=}")


if __name__ == "__main__":
    raise SystemExit(test())
