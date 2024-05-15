import io
import sys
from collections import deque
from contextlib import ExitStack, contextmanager
from types import GeneratorType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    List,
    NoReturn,
    Optional,
    Tuple,
)
from typing import Union as TUnion

from pycparser.cparsing.parser_c import (
    Coord,  # noqa: F401 # pyright: ignore [reportUnusedImport] # Used in _create_init.
)

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
elif TYPE_CHECKING:
    from typing_extensions import dataclass_transform
else:
    from typing import Callable

    def dataclass_transform(**kwargs: bool) -> Callable[[type], type]:
        def inner(cls: type) -> type:
            return cls


if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeGuard
elif TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeGuard
else:
    if sys.version_info >= (3, 9, 2):
        from types import GenericAlias as _GenericAlias
    else:
        from typing import _GenericAlias

    class _TypeGuardGenericAlias(_GenericAlias):
        def __repr__(self):
            return f"<placeholder for {super().__repr__()}>"

    class _TypeGuardMeta(type):
        def __getitem__(self, item: object) -> _TypeGuardGenericAlias:
            return _TypeGuardGenericAlias(self, item)

        def __repr__(self):
            return f"<placeholder for {super().__repr__()}>"

    class TypeGuard(metaclass=_TypeGuardMeta): 
        pass

    class TypeAlias:
        def __repr__(self):
            return f"<placeholder for {super().__repr__()}"


_SimpleNode: TypeAlias = "TUnion[Constant, Identifier, ArrayRef, StructRef, FuncCall]"


def _create_init(annotations: Dict[str, Any]) -> Any:
    """Generate and evaluate an __init__ function based on a given annotation dict and global namespace."""

    init_lines: list[str] = []

    partial_signature_lines: list[str] = []
    for name, ann in annotations.items():
        # Account for string annotations.
        if isinstance(ann, str):
            actual_ann = repr(ann)
        else:
            actual_ann = getattr(ann, '__qualname__', ann)

        partial_signature_lines.append(f"{name}: {actual_ann}, ")

    partial_signature = "".join(partial_signature_lines)

    init_lines.append(f"def __init__(self, {partial_signature}*, coord: Optional[Coord] = None) -> None:")
    init_lines.extend(f"    self.{name} = {name}" for name in annotations)
    init_lines.append("    self.coord: Optional[Coord] = coord")

    return "\n".join(init_lines)


@dataclass_transform(eq_default=False)
class ASTMeta(type):
    """Custom metaclass for AST classes that generates __slots__, __match_args__, __init__, and _fields attributes/
    methods based on the classes's annotations.

    Notes
    -----
    This does not work exactly like dataclasses do. It doesn't provide options to change what gets generated, and it
    can't generate everything that dataclasses can. However, the dataclass_transform decorator is needed for
    type-checkers to infer the signature of the generated __init__.
    """

    _fields: Tuple[str, ...]

    def __new__(mcls, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any], **kwargs: NoReturn):
        if kwargs:
            msg = (
                "These kwargs are for show. cast.AST implements dataclass-like behavior, but only a subset and "
                "unilaterally at that: __init__, __slots__, and __match_args__ are always generated."
            )
            raise TypeError(msg)

        cls_annotations: dict[str, Any] = namespace.get("__annotations__", {})

        try:
            global_ns = sys.modules[namespace["__module__"]].__dict__
        except (KeyError, AttributeError):
            global_ns = {}

        init_code = _create_init(cls_annotations)
        exec(init_code, global_ns, namespace)  # noqa: S102

        cls_fields = tuple(cls_annotations)
        cls_slots = cls_fields + tuple(namespace.get("__slots__", ()))

        namespace.update(
            {
                "__slots__": cls_slots,
                "__match_args__": cls_fields,
                "_fields": cls_fields,
            }
        )
        return super().__new__(mcls, name, bases, namespace)


class AST(metaclass=ASTMeta):
    __slots__ = ("__weakref__", "coord")


class File(AST):
    ext: List[AST]


class ExprList(AST):
    exprs: List[AST]


class Enumerator(AST):
    name: str
    value: AST


class EnumeratorList(AST):
    enumerators: List[Enumerator]


class ParamList(AST):
    params: List[AST]


class EllipsisParam(AST):
    pass


class Compound(AST):
    block_items: List[AST]


# ======== Flow control constructs

# ==== Looping


class For(AST):
    init: AST
    cond: AST
    next: AST
    stmt: AST


class While(AST):
    cond: AST
    stmt: AST


class DoWhile(AST):
    cond: AST
    stmt: AST


# ==== Other


class GoTo(AST):
    name: str


class Label(AST):
    name: str
    stmt: AST


class Switch(AST):
    cond: AST
    stmt: AST


class Case(AST):
    expr: AST
    stmts: List[AST]


class Default(AST):
    stmts: List[AST]


class If(AST):
    cond: AST
    iftrue: AST
    iffalse: Optional[AST]


class Continue(AST):
    pass


class Break(AST):
    pass


class Return(AST):
    expr: AST


# ======== Operations


class Assignment(AST):
    op: str
    left: AST
    right: AST


class UnaryOp(AST):
    op: str
    expr: AST


class BinaryOp(AST):
    op: str
    left: AST
    right: AST


class TernaryOp(AST):
    cond: AST
    iftrue: AST
    iffalse: AST


# ======== Base


class Pragma(AST):
    string: str


class Identifier(AST):
    name: str


class Constant(AST):
    type: str
    value: str


class EmptyStatement(AST):
    pass


# ======== Other


class ArrayDecl(AST):
    type: AST
    dim: AST
    dim_quals: List[str]


class ArrayRef(AST):
    name: AST
    subscript: AST


class Alignas(AST):
    alignment: AST


class Cast(AST):
    to_type: AST
    expr: AST


class CompoundLiteral(AST):
    type: AST
    init: AST


class Decl(AST):
    name: Optional[str]
    quals: List[str]
    align: List[Any]
    storage: List[Any]
    funcspec: List[Any]
    type: "Decl"
    init: Optional[AST]
    bitsize: Optional[AST]


class DeclList(AST):
    decls: List[Decl]


class Enum(AST):
    name: Optional[str]
    values: Optional[EnumeratorList]


class FuncCall(AST):
    name: Identifier
    args: ExprList


class FuncDecl(AST):
    args: AST
    type: AST


class FuncDef(AST):
    decl: AST
    param_decls: List[AST]
    body: AST


class IdentifierType(AST):
    names: List[str]


class InitList(AST):
    exprs: List[AST]


class NamedInitializer(AST):
    name: List[AST]
    expr: AST


class PtrDecl(AST):
    quals: Any
    type: AST


class StaticAssert(AST):
    cond: AST
    message: AST


class Struct(AST):
    name: str
    decls: Optional[List[AST]]


class StructRef(AST):
    name: AST
    type: Any
    field: AST


class TypeDecl(AST):
    declname: Optional[str]
    quals: Optional[Any]
    align: Optional[Any]
    type: Optional[AST]


class Typedef(AST):
    name: Optional[str]
    quals: List[str]
    storage: List[Any]
    type: AST


class Typename(AST):
    name: str
    quals: List[str]
    align: Optional[Any]
    type: AST


class Union(AST):
    name: str
    decls: Optional[List[AST]]


# ======== AST utilities


class NodeVisitor:
    """Visitor pattern for the AST implemented based on a talk by David Beazley called
    "Generators: The Final Frontier".
    """

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

        for field in node._fields:
            potential_subnode = getattr(node, field)

            if isinstance(potential_subnode, AST):
                yield getattr(node, field)

            elif isinstance(potential_subnode, list):
                for subsub in potential_subnode:  # pyright: ignore [reportUnknownVariableType]
                    if isinstance(subsub, AST):
                        yield subsub

        return None


class NodePrettyPrinter(NodeVisitor):
    def __init__(self, annotate_fields: bool = True, include_coords: bool = False, *, indent: Optional[str] = None):
        self.annotate_fields = annotate_fields
        self.include_coords = include_coords
        self.indent = indent

        self.buffer = io.StringIO()
        self.level = 0

    @property
    def prefix(self) -> str:
        if self.indent is not None:
            prefix = f"\n{self.indent * self.level}"
        else:
            prefix = ""

        return prefix

    @property
    def sep(self) -> str:
        if self.indent is not None:
            sep = f",\n{self.indent * self.level}"
        else:
            sep = ", "
        return sep

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object):
        self.buffer.close()

    @contextmanager
    def add_level(self) -> Generator[None, Any, None]:
        self.level += 1
        try:
            yield
        finally:
            self.level -= 1

    def generic_visit(self, node: AST) -> Generator[Any, Any, None]:
        with self.add_level():
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
                        with self.add_level():
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


def dump(
    node: AST,
    annotate_fields: bool = True,
    include_coords: bool = False,
    *,
    indent: Optional[str] = None,
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
    indent: str | None, optional
        The indent level to pretty-print the tree with. None by default, which selects the single line representation.

    Returns
    -------
    str
        The formatted string representation of the given AST.
    """

    with NodePrettyPrinter(annotate_fields, include_coords, indent=indent) as visitor:
        visitor.visit(node)
        return visitor.buffer.getvalue()


class Unparser(NodeVisitor):
    # fmt: off
    # Precedence map of binary operators:
    precedence_map: ClassVar[Dict[str, int]] = {
        # Should be in sync with c_parser.CParser.precedence
        # Higher numbers are stronger binding
        '||': 0,  # weakest binding
        '&&': 1,
        '|': 2,
        '^': 3,
        '&': 4,
        '==': 5, '!=': 5,
        '>': 6, '>=': 6, '<': 6, '<=': 6,
        '>>': 7, '<<': 7,
        '+': 8, '-': 8,
        # strongest binding
        '*': 9, '/': 9, '%': 9,
    }
    # fmt: on

    def __init__(self, *, reduce_parentheses: bool = False):
        self.indent_level = 0
        self.reduce_parentheses = reduce_parentheses

    @property
    def indent(self) -> str:
        return " " * self.indent_level

    @contextmanager
    def add_indent_level(self, val: int = 1) -> Generator[None, Any, None]:
        self.indent_level += val
        try:
            yield
        finally:
            self.indent_level -= val

    @staticmethod
    def _is_simple_node(node: AST) -> TypeGuard[_SimpleNode]:
        return isinstance(node, (Constant, Identifier, ArrayRef, StructRef, FuncCall))

    def _visit_expr(self, node: AST) -> str:
        if isinstance(node, InitList):
            return "{%s}"
        elif isinstance(node, ExprList):  # noqa: RET505
            return "(%s)"
        else:
            return "%s"

    def _parenthesize_if(self, node: AST, condition: Callable[[AST], bool]) -> str:
        """Visits 'n' and returns its string representation, parenthesized
        if the condition function applied to the node returns True.
        """

        result = self._visit_expr(node)
        return f'({result})' if condition(node) else result

    def _parenthesize_unless_simple(self, node: AST) -> str:
        return self._parenthesize_if(node, lambda n: not self._is_simple_node(n))

    def _generate_struct_union_body(self, members: List[AST]) -> Generator[AST, str, str]:
        results: list[str] = []
        for decl in members:
            decl_str = yield from self._generate_stmt(decl)
            results.append(decl_str)
        return ''.join(results)

    def _generate_stmt(self, node: AST, add_indent: bool = False) -> Generator[AST, str, str]:
        """Generation from a statement node. This method exists as a wrapper
        for individual visit_* methods to handle different treatment of
        some statements in this context.
        """

        typ = type(node)
        indent_stack = ExitStack()
        if add_indent:
            indent_stack.push(self.add_indent_level(2))

        with indent_stack:
            indent = self.indent

        node_str = yield node

        if typ in {
            Decl,
            Assignment,
            Cast,
            UnaryOp,
            BinaryOp,
            TernaryOp,
            FuncCall,
            ArrayRef,
            StructRef,
            Constant,
            Identifier,
            Typedef,
            ExprList,
        }:
            # These can also appear in an expression context so no semicolon
            # is added to them automatically
            #
            return f"{indent}{node_str};\n"
        elif typ in {Compound}:
            # No extra indentation required before the opening brace of a
            # compound - because it consists of multiple lines it has to
            # compute its own indentation.
            #
            return node_str
        elif typ in {If}:
            return f"{indent}{node_str}"
        else:
            return f"{indent}{node_str}\n"

    def _generate_decl(self, node: Decl) -> Generator[Any, str, str]:
        """Generation from a Decl node."""

        results: list[str] = []

        if node.funcspec:
            results.append(' '.join(node.funcspec) + ' ')
        if node.storage:
            results.append(' '.join(node.storage) + ' ')
        if node.align:
            align = yield node.align[0]
            results.append(f"{align} ")

        type_ = yield from self._generate_type(node.type)
        results.append(type_)
        return "".join(results)

    def _generate_type(
        self,
        node: AST,
        modifiers: Optional[List[Any]] = None,
        emit_declname: bool = True,
    ) -> Generator[AST, str, str]:
        """Recursive generation from a type node. n is the type node.
        modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers
        encountered on the way down to a TypeDecl, to allow proper
        generation from it.
        """

        if modifiers is None:
            modifiers = []

        # ~ print(n, modifiers)

        if isinstance(node, TypeDecl):
            results: list[str] = []
            if node.quals:
                results.append(' '.join(node.quals) + ' ')
            type_ = yield node.type
            results.append(type_)

            nstr = [node.declname if (node.declname and emit_declname) else ""]
            # Resolve modifiers.
            # Wrap in parens to distinguish pointer to array and pointer to
            # function syntax.
            #
            for i, modifier in enumerate(modifiers):
                if isinstance(modifier, ArrayDecl):
                    if i != 0 and isinstance(modifiers[i - 1], PtrDecl):
                        nstr.insert(0, "(")
                        nstr.append(")")

                    nstr.append('[')
                    if modifier.dim_quals:
                        nstr.append(' '.join(modifier.dim_quals) + ' ')
                    nstr.append(self.visit(modifier.dim) + ']')
                elif isinstance(modifier, FuncDecl):
                    if i != 0 and isinstance(modifiers[i - 1], PtrDecl):
                        nstr.insert(0, "(")
                        nstr.append(")")
                    nstr.append('(' + self.visit(modifier.args) + ')')
                elif isinstance(modifier, PtrDecl):
                    if modifier.quals:
                        modifier_quals_str = " ".join(modifier.quals)
                        if nstr:
                            nstr.insert(0, f"* {modifier_quals_str} ")
                        else:
                            nstr = [f"* {modifier_quals_str}"]
                    else:
                        nstr.insert(0, "*")
            if nstr:
                results.append(" " + "".join(nstr))
            return "".join(results)
        elif isinstance(node, Decl):
            result = yield from self._generate_decl(node.type)
            return result
        elif isinstance(node, Typename):
            result = yield from self._generate_type(node.type, emit_declname=emit_declname)
            return result
        elif isinstance(node, IdentifierType):
            return ' '.join(node.names) + ' '
        elif isinstance(node, (ArrayDecl, PtrDecl, FuncDecl)):
            result = yield from self._generate_type(node.type, [*modifiers, node], emit_declname=emit_declname)
            return result
        else:
            result = yield node
            return result

    def visit_Constant(self, node: Constant) -> str:
        return node.value

    def visit_Identifier(self, node: Identifier) -> str:
        return node.name

    def visit_Pragma(self, node: Pragma) -> str:
        ret = "#pragma"
        if node.string:
            ret += f" {node.string}"
        return ret

    def visit_ArrayRef(self, node: ArrayRef) -> Generator[AST, str, str]:
        name = yield node.name
        arrref = self._parenthesize_unless_simple(node.name) % name
        subscript = yield node.subscript
        return f"{arrref}[{subscript}]"

    def visit_StructRef(self, node: StructRef) -> Generator[AST, str, str]:
        name = yield node.name
        sref = self._parenthesize_unless_simple(node.name) % name
        field = yield node.field
        return f"{sref}{node.type}{field}"

    def visit_FuncCall(self, node: FuncCall) -> Generator[AST, str, str]:
        name = yield node.name
        fref = self._parenthesize_unless_simple(node.name) % name
        args = yield node.args
        return f"{fref}({args})"

    def visit_UnaryOp(self, node: UnaryOp) -> Generator[AST, str, str]:
        expr = yield node.expr

        if node.op == "sizeof":
            return f"sizeof{expr}"
        else:
            operand = self._parenthesize_unless_simple(node.expr) % expr
            if node.op == "p++":
                return f"{operand}++"
            elif node.op == "p--":
                return f"{operand}--"
            else:
                return f"{node.op}{operand}"

    def visit_BinaryOp(self, node: BinaryOp) -> Generator[AST, str, str]:
        # Note: all binary operators are left-to-right associative
        #
        # If `n.left.op` has a stronger or equally binding precedence in
        # comparison to `n.op`, no parenthesis are needed for the left:
        # e.g., `(a*b) + c` is equivalent to `a*b + c`, as well as
        #       `(a+b) - c` is equivalent to `a+b - c` (same precedence).
        # If the left operator is weaker binding than the current, then
        # parentheses are necessary:
        # e.g., `(a+b) * c` is NOT equivalent to `a+b * c`.
        left_str = self._parenthesize_if(
            node.left,
            lambda d: not (
                self._is_simple_node(d)
                or (
                    self.reduce_parentheses
                    and isinstance(d, BinaryOp)
                    and self.precedence_map[d.op] >= self.precedence_map[node.op]
                )
            ),
        )
        left = yield node.left
        left_str = left_str % left

        # If `n.right.op` has a stronger -but not equal- binding precedence,
        # parenthesis can be omitted on the right:
        # e.g., `a + (b*c)` is equivalent to `a + b*c`.
        # If the right operator is weaker or equally binding, then parentheses
        # are necessary:
        # e.g., `a * (b+c)` is NOT equivalent to `a * b+c` and
        #       `a - (b+c)` is NOT equivalent to `a - b+c` (same precedence).
        right_str = self._parenthesize_if(
            node.right,
            lambda d: not (
                self._is_simple_node(d)
                or (
                    self.reduce_parentheses
                    and isinstance(d, BinaryOp)
                    and self.precedence_map[d.op] > self.precedence_map[node.op]
                )
            ),
        )
        right = yield node.right
        right_str = right_str % right
        return f'{left_str} {node.op} {right_str}'

    def visit_Assignment(self, node: Assignment) -> Generator[AST, str, str]:
        right = yield node.right
        right_str = self._parenthesize_if(node.right, lambda n: isinstance(n, Assignment)) % right
        left = yield node.left
        return f"{left} {node.op} {right_str}"

    def visit_IdentifierType(self, node: IdentifierType) -> str:
        return " ".join(node.names)

    def visit_Decl(self, node: Decl, *, no_type: bool = False) -> Generator[AST, str, str]:
        # no_type is used when a Decl is part of a DeclList, where the type is
        # explicitly only for the first declaration in a list.
        #
        if no_type:
            result: list[str] = [node.name]
        else:
            decl = yield from self._generate_decl(node)
            result = [decl]

        if node.bitsize:
            bitsize = yield node.bitsize
            result.append(f' : {bitsize}')
        if node.init:
            init = yield node.init
            result.append(' = ' + self._visit_expr(node.init) % init)
        return "".join(result)

    def visit_DeclList(self, node: DeclList) -> Generator[AST, str, str]:
        result: list[str] = []
        first_decl = yield node.decls[0]
        result.append(first_decl)

        if len(node.decls) > 1:
            result.append(", ")

            decls_list: list[str] = []
            for decl in node.decls[1:]:
                decl_str = yield from self.visit_Decl(decl, no_type=True)
                decls_list.append(decl_str)
            result.append(", ".join(decls_list))

        return "".join(result)

    def visit_Typedef(self, node: Typedef) -> Generator[AST, str, str]:
        result: list[str] = []
        if node.storage:
            result.append(' '.join(node.storage) + ' ')

        type_ = yield from self._generate_type(node.type)

        result.append(type_)
        return "".join(result)

    def visit_Cast(self, node: Cast) -> Generator[AST, str, str]:
        expr = yield node.expr
        expr_str = self._parenthesize_unless_simple(node.expr) % expr
        type_ = yield from self._generate_type(node.to_type, emit_declname=False)
        return f"({type_}) {expr_str}"

    def visit_ExprList(self, node: ExprList) -> Generator[AST, str, str]:
        visited_subexprs: list[str] = []
        for expr in node.exprs:
            expr_str = yield expr
            visited_subexprs.append(self._visit_expr(expr) % expr_str)

        return ", ".join(visited_subexprs)

    def visit_InitList(self, node: InitList) -> Generator[AST, str, str]:
        visited_subexprs: list[str] = []
        for expr in node.exprs:
            expr_str = yield expr
            visited_subexprs.append(self._visit_expr(expr) % expr_str)

        return ", ".join(visited_subexprs)

    def visit_Enum(self, node: Enum) -> Generator[AST, str, str]:
        results: List[str] = [f"enum {(node.name or '')}"]

        members = None if node.values is None else node.values.enumerators
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            results.append('\n')
            results.append(self.indent)
            with self.add_indent_level(2):
                results.append('{\n')

                # `[:-2] + '\n'` removes the final `,` from the enumerator list
                enum_body: list[str] = []
                for value in members:
                    body_part = yield value
                    enum_body.append(body_part)
                results.append("".join(enum_body)[:-2] + "\n")

            results.append(self.indent + '}')
        return "".join(results)

    def visit_Alignas(self, node: Alignas) -> Generator[AST, str, str]:
        alignment = yield node.alignment
        return f"_Alignas({alignment})"

    def visit_Enumerator(self, node: Enumerator) -> Generator[AST, str, str]:
        if not node.value:
            return f"{self.indent}{node.name},\n"
        else:
            value = yield node.value
            return f"{self.indent}{node.name} = {value}"

    def visit_FuncDef(self, node: FuncDef) -> Generator[AST, str, str]:
        decl = yield node.decl
        self.indent_level = 0
        body = yield node.body
        if node.param_decls:
            knrdecls: list[str] = []
            for p in node.param_decls:
                knrdecls.append((yield p))

            krndecls_str = ';\n'.join(knrdecls)
            return f"{decl}\n{krndecls_str};\n{body}\n"
        else:
            return f"{decl}\n{body}\n"

    def visit_File(self, node: File) -> Generator[AST, str, str]:
        results: list[str] = []
        for ext in node.ext:
            result = yield ext
            if isinstance(ext, FuncDef):
                results.append(result)
            elif isinstance(ext, Pragma):
                results.append(f"{result}\n")
            else:
                results.append(f"{result};\n")
        return "".join(results)

    def visit_Compound(self, node: Compound) -> Generator[AST, str, str]:
        results: list[str] = [self.indent + "{\n"]
        with self.add_indent_level(2):
            if node.block_items:
                block_statements: list[str] = []
                for stmt in node.block_items:
                    block_stmt_str = yield from self._generate_stmt(stmt)
                    block_statements.append(block_stmt_str)

                results.append(''.join(block_statements))

        results.append(self.indent + '}\n')
        return "".join(results)

    def visit_CompoundLiteral(self, n: CompoundLiteral) -> Generator[AST, str, str]:
        type_ = yield n.type
        init = yield n.init

        return f'({type_})' + '{' + init + '}'

    def visit_EmptyStatement(self, n: EmptyStatement) -> str:
        return ';'

    def visit_ParamList(self, node: ParamList) -> Generator[AST, str, str]:
        results: list[str] = []
        for param in node.params:
            param_str = yield param
            results.append(param_str)
        return ', '.join(results)

    def visit_Return(self, node: Return) -> Generator[AST, str, str]:
        result = 'return'
        if node.expr:
            expr = yield node.expr
            result += f' {expr}'
        return f"{result};"

    def visit_Break(self, node: Break) -> str:
        return 'break;'

    def visit_Continue(self, node: Continue) -> str:
        return 'continue;'

    def visit_TernaryOp(self, node: TernaryOp) -> Generator[AST, str, str]:
        cond = yield node.cond
        iftrue = yield node.iftrue
        iffalse = yield node.iffalse

        return (
            f"({self._visit_expr(node.cond) % cond}) ? "
            f"({self._visit_expr(node.iftrue) % iftrue}) : "
            f"({self._visit_expr(node.iffalse) % iffalse})"
        )

    def visit_If(self, n: If) -> Generator[AST, str, str]:
        results = ['if (']
        if n.cond:
            cond = yield n.cond
            results.append(cond)
        results.append(')\n')
        stmt = yield from self._generate_stmt(n.iftrue, add_indent=True)
        results.append(stmt)
        if n.iffalse:
            results.append(self.indent + 'else\n')
            stmt = yield from self._generate_stmt(n.iffalse, add_indent=True)
            results.append(stmt)
        return "".join(results)

    def visit_For(self, node: For) -> Generator[AST, str, str]:
        results = ['for (']

        if node.init:
            init = yield node.init
            results.append(init)
        results.append(';')

        if node.cond:
            cond = yield node.cond
            results.append(f' {cond}')
        results.append(';')

        if node.next:
            next_ = yield node.next
            results.append(f' {next_}')
        results.append(')\n')

        stmt = yield from self._generate_stmt(node.stmt, add_indent=True)
        results.append(stmt)
        return "".join(results)

    def visit_While(self, node: While) -> Generator[AST, str, str]:
        results = ['while (']
        if node.cond:
            cond = yield node.cond
            results.append(cond)
        results.append(')\n')
        stmt = yield from self._generate_stmt(node.stmt, add_indent=True)
        results.append(stmt)
        return "".join(results)

    def visit_DoWhile(self, node: DoWhile) -> Generator[AST, str, str]:
        results = ['do\n']
        stmt = yield from self._generate_stmt(node.stmt, add_indent=True)
        results.append(stmt)
        results.append(f'{self.indent}while (')

        if node.cond:
            results.append((yield node.cond))
        results.append(');')
        return "".join(results)

    def visit_StaticAssert(self, node: StaticAssert) -> Generator[AST, str, str]:
        results = ['_Static_assert(', (yield node.cond)]

        if node.message:
            results.append(',')
            results.append((yield node.message))
        results.append(')')
        return "".join(results)

    def visit_Switch(self, node: Switch) -> Generator[AST, str, str]:
        cond = yield node.cond
        stmt = yield from self._generate_stmt(node.stmt, add_indent=True)

        return "".join((f'switch ({cond})\n', stmt))

    def visit_Case(self, node: Case) -> Generator[AST, str, str]:
        expr = yield node
        stmts: list[str] = []
        for stmt in node.stmts:
            stmt_str = yield from self._generate_stmt(stmt, add_indent=True)
            stmts.append(stmt_str)

        return "".join((f'case {expr}:\n', *stmts))

    def visit_Default(self, node: Default) -> Generator[AST, str, str]:
        s = 'default:\n'
        stmts: list[str] = []
        for stmt in node.stmts:
            stmt_str = yield from self._generate_stmt(stmt, add_indent=True)
            stmts.append(stmt_str)

        return "".join((s, *stmts))

    def visit_Label(self, node: Label) -> Generator[AST, str, str]:
        stmt = yield from self._generate_stmt(node.stmt)
        return f'{node.name}:\n{stmt}'

    def visit_Goto(self, node: GoTo) -> str:
        return f'goto {node.name};'

    def visit_EllipsisParam(self, node: EllipsisParam) -> str:
        return '...'

    def visit_Struct(self, node: Struct) -> Generator[AST, str, str]:
        results: List[str] = [f"struct {(node.name or '')}"]

        members = node.decls
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            results.append('\n')
            results.append(self.indent)
            with self.add_indent_level(2):
                results.append('{\n')
                body = yield from self._generate_struct_union_body(members)
                results.append(body)

            results.append(self.indent + '}')
        return "".join(results)

    def visit_Union(self, node: Union) -> Generator[AST, str, str]:
        results: List[str] = [f"union {(node.name or '')}"]

        members = node.decls
        if members is not None:
            # None means no members
            # Empty sequence means an empty list of members
            results.append('\n')
            results.append(self.indent)
            with self.add_indent_level(2):
                results.append('{\n')
                body = yield from self._generate_struct_union_body(members)
                results.append(body)

            results.append(self.indent + '}')
        return "".join(results)

    def visit_Typename(self, node: Typename) -> Generator[AST, str, str]:
        result = yield from self._generate_type(node.type)
        return result

    def visit_NamedInitializer(self, node: NamedInitializer) -> Generator[AST, str, str]:
        results: list[str] = []

        for name in node.name:
            if isinstance(name, Identifier):
                results.append(f".{name.name}")
            else:
                name_str = yield name
                results.append(f"[{name_str}]")

        expr = yield node.expr
        results.append(f' = {self._visit_expr(node.expr) % expr}')

        return "".join(results)

    def visit_FuncDecl(self, node: FuncDecl) -> Generator[AST, str, str]:
        result = yield from self._generate_type(node)
        return result

    def visit_ArrayDecl(self, node: ArrayDecl) -> Generator[AST, str, str]:
        result = yield from self._generate_type(node, emit_declname=False)
        return result

    def visit_TypeDecl(self, node: TypeDecl) -> Generator[AST, str, str]:
        result = yield from self._generate_type(node, emit_declname=False)
        return result

    def visit_PtrDecl(self, node: PtrDecl) -> Generator[AST, str, str]:
        result = yield from self._generate_type(node, emit_declname=False)
        return result


def unparse(node: AST, *, reduce_parentheses: bool = False) -> str:
    """Unparse an AST object and generate a string with code that would produce an equivalent AST object if parsed back."""

    unparser = Unparser(reduce_parentheses=reduce_parentheses)
    return unparser.visit(node)


def compare_asts(first_node: TUnion[AST, List[AST], Any], second_node: TUnion[AST, List[AST], Any]) -> bool:
    """Compare two AST nodes for equality, to see if they have the same field structure with the same values.

    This only takes into account fields present in a node's _fields list.

    Notes
    -----
    The algorithm is modified from https://stackoverflow.com/a/19598419 to be iterative instead of recursive.
    """

    nodes = deque([(first_node, second_node)])

    while nodes:
        node1, node2 = nodes.pop()

        if type(node1) is not type(node2):
            return False

        if isinstance(node1, AST):
            nodes.extend((getattr(node1, field), getattr(node2, field)) for field in node1._fields if field != "ctx")
            continue

        if isinstance(node1, list):
            assert isinstance(node2, list)
            try:
                if sys.version_info >= (3, 10):
                    nodes.extend(zip(node1, node2, strict=True))
                else:
                    if len(node1) != len(node2):
                        return False
                    nodes.extend(zip(node1, node2))
            except ValueError:
                return False

            continue

        if node1 != node2:
            return False

    return True
