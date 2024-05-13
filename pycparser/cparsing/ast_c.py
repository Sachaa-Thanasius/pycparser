# ruff: noqa: T201
from __future__ import annotations

from collections import deque
from types import GeneratorType
from typing import Any, ClassVar, Generator, Optional, Union


class Coord: ...


class ASTMeta(type):
    @classmethod
    def __create_init(cls, annotations: dict[str, Any], global_ns: dict[str, Any]) -> Any:
        partial_signature = ", ".join(
            f"{name}: {getattr(ann, '__qualname__', ann)}" for name, ann in annotations.items()
        )
        signature = f"def __init__(self, {partial_signature}) -> None:\n"
        body = "\n    ".join(f"self.{name} = {name}" for name in annotations)
        init_source = signature + body

        local_ns: dict[str, Any] = {}
        return eval(init_source, global_ns, local_ns)  # noqa: S307

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        try:
            annotations: dict[str, Any] = namespace["__annotations__"]
        except KeyError:
            return super().__new__(cls, name, bases, namespace)

        try:
            cls_global_ns: dict[str, Any] = namespace["__module__"].__dict__
        except (KeyError, AttributeError):
            cls_global_ns = {}

        cls_match_args = tuple(annotations)
        cls_init = cls.__create_init(annotations, cls_global_ns)

        namespace.update(
            {
                "__slots__": namespace.get("__slots__", ()) + cls_match_args,
                "__match_args__": cls_match_args,
                "__init__": cls_init,
            }
        )
        return super().__new__(cls, name, bases, namespace)


class AST:
    __slots__ = ("__weakref__", "coord")

    _fields: ClassVar[tuple[str, ...]]
    coord: Coord | None


class Number(AST):
    __slots__ = ("value",)
    __match_args__ = ("value",)

    _fields = ()

    def __init__(self, value: int):
        self.value: int = value


class NegNumber(AST):
    __slots__ = ("value",)
    __match_args__ = ("value",)

    _fields = ()

    def __init__(self, value: int):
        self.value: int = value


class BinOp(AST):
    __slots__ = ("op", "left", "right")
    __match_args__ = ("op", "left", "right")

    _fields = ("left", "right")

    def __init__(self, op: str, left: AST, right: AST):
        self.op: str = op
        self.left: AST = left
        self.right: AST = right


class NodeVisitor:
    def _visit(self, node: AST) -> Generator[Any, Any, Any]:
        result: Any = getattr(self, f"visit_{type(node).__name__}", self.generic_visit)(node)
        if isinstance(result, GeneratorType):
            result = yield from result
        return result

    def visit(self, node: AST) -> Any:
        """Visit all the nodes."""

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

        for attr in node._fields:
            yield getattr(node, attr)
        return None


class Evaluator(NodeVisitor):
    def visit_Number(self, node: Number) -> int:
        return node.value

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

    def visit_BinOp(self, node: BinOp) -> Generator[Any, Any, None]:
        print(node.op)
        return self.generic_visit(node)


def make_test_tree() -> AST:
    tree = BinOp("+", NegNumber(-1), Number(0))
    for i in range(1, 100):
        tree = BinOp("+", Number(i), tree)
    return tree


def test() -> None:
    tree = make_test_tree()
    Sightseer().visit(tree)


if __name__ == "__main__":
    raise SystemExit(test())
