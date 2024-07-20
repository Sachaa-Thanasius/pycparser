# pyright: reportRedeclaration=none, reportUndefinedVariable=none
"""Module for parsing C tokens into an AST."""

from typing import TYPE_CHECKING, Any, NoReturn, Optional, TypedDict, TypeVar, Union, cast, overload

from sly import Parser
from sly.lex import Token
from sly.yacc import YaccProduction as YaccProd, YaccSymbol

from . import c_ast, c_context
from ._cluegen import Datum
from ._typing_compat import NotRequired, Self
from .c_lexer import CLexer
from .utils import Coord, substitute


if TYPE_CHECKING:
    from sly.types import _

_DeclarationT = TypeVar("_DeclarationT", bound=Union[c_ast.Typedef, c_ast.Decl, c_ast.Typename])
_ModifierT = TypeVar("_ModifierT", bound=c_ast.TypeModifier)


__all__ = ("CParser",)


# ============================================================================
# region -------- Helpers and AST fixers
# ============================================================================


class _StructDeclaratorDict(TypedDict):
    decl: Optional[Union[c_ast.TypeDecl, c_ast.TypeModifier, c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdType]]
    bitsize: NotRequired[Optional[c_ast.AST]]
    init: NotRequired[Optional[c_ast.AST]]


class _DeclarationSpecifiers(Datum):
    """Declaration specifiers for C declarations.

    Attributes
    ----------
    qual: list[str], default=[]
        A list of type qualifiers.
    storage: list[str], default=[]
        A list of storage type qualifiers.
    type: list[c_ast.IdType], default=[]
        A list of type specifiers.
    function: list[str], default=[]
        A list of function specifiers.
    alignment: list[c_ast.Alignas], default=[]
        A list of alignment specifiers.
    """

    qual: list[str] = []
    storage: list[str] = []
    type: list[Union[c_ast.IdType, c_ast.Typename]] = []
    function: list[str] = []
    alignment: list[c_ast.Alignas] = []

    @classmethod
    def add(cls, decl_spec: Optional[Self], new_item: Any, kind: str, *, append: bool = False) -> Self:
        """Given a declaration specifier and a new specifier of a given kind, add the specifier to its respective list.

        If `append` is True, the new specifier is added to the end of the specifiers list, otherwise it's added at the
        beginning. Returns the declaration specifier, with the new specifier incorporated.
        """

        if decl_spec is None:
            return cls(**{kind: [new_item]})
        else:
            subspec_list: list[Any] = getattr(decl_spec, kind)
            if append:
                subspec_list.append(new_item)
            else:
                subspec_list.insert(0, new_item)

            return decl_spec


def _fix_atomic_specifiers_once(decl: Union[c_ast.Typedef, c_ast.Decl]) -> tuple[Any, bool]:
    """Performs one "fix" round of atomic specifiers.

    Returns
    -------
    (modified_decl, found): tuple[Any, bool]
        A tuple with the original decl, possibly modified, and a bool indicating whether a fix was made.
    """

    grandparent: Any = None
    parent = decl
    node = decl.type

    while node is not None:
        if isinstance(node, c_ast.Typename) and "_Atomic" in node.quals:
            break

        grandparent = parent
        parent = node

        try:
            node = node.type
        except AttributeError:
            # If we've reached a node without a `type` field, it means we won't find what we're looking for
            # at this point; give up the search and return the original decl unmodified.
            return decl, False

    assert isinstance(parent, c_ast.TypeDecl)
    grandparent.type = node.type
    if "_Atomic" not in node.type.quals:
        node.type.quals.append("_Atomic")
    return decl, True


def fix_atomic_specifiers(decl: Union[c_ast.Typedef, c_ast.Decl]) -> Any:
    """Atomic specifiers like _Atomic(type) are unusually structured, conferring a qualifier upon the contained type.

    This function fixes a decl with atomic specifiers to have a sane AST structure, by removing spurious
    Typename->TypeDecl pairs and attaching the _Atomic qualifier in the right place.
    """

    # There can be multiple levels of _Atomic in a decl; fix them until a fixed point is reached.
    found = True
    while found:
        decl, found = _fix_atomic_specifiers_once(decl)

    # Make sure to add an _Atomic qual on the topmost decl if needed.
    # Also restore the declname on the innermost TypeDecl, since it gets placed in the wrong place during construction.
    typ = decl
    while not isinstance(typ, c_ast.TypeDecl):
        try:
            typ = typ.type
        except AttributeError:
            return decl

    assert typ.quals is not None
    if "_Atomic" in typ.quals and "_Atomic" not in decl.quals:
        decl.quals.append("_Atomic")
    if typ.declname is None:
        typ.declname = decl.name

    return decl


def fix_switch_cases(switch_node: c_ast.Switch) -> c_ast.Switch:
    """Fix the mess of case statements created for a switch node by default.

    Parameters
    ----------
    switch_node: c_ast.Switch
        An unfixed switch node. May be modified by the function.

    Returns
    -------
    c_ast.Switch
        The fixed switch node.

    Notes
    -----
    The "case" statements in a "switch" come out of parsing with one child node, so subsequent statements are just
    tucked to the parent Compound. Additionally, consecutive (fall-through) case statements come out messy. This is a
    peculiarity of the C grammar.

    The following:

        switch (myvar) {
            case 10:
                k = 10;
                p = k + 1;
                return 10;
            case 20:
            case 30:
                return 20;
            default:
                break;
        }

    creates this tree (pseudo-dump):

        Switch
            ID: myvar
            Compound:
                Case 10:
                    k = 10
                p = k + 1
                return 10
                Case 20:
                    Case 30:
                        return 20
                Default:
                    break

    The goal of this transform is to fix this mess, turning it into the
    following:

        Switch
            ID: myvar
            Compound:
                Case 10:
                    k = 10
                    p = k + 1
                    return 10
                Case 20:
                Case 30:
                    return 20
                Default:
                    break
    """

    if not isinstance(switch_node.stmt, c_ast.Compound):
        return switch_node

    # The new Compound child for the Switch, which will collect children in the correct order.
    new_compound = c_ast.Compound([], coord=switch_node.stmt.coord)
    assert isinstance(new_compound.block_items, list)

    # The last Case/Default node.
    latest_case: Optional[Union[c_ast.Case, c_ast.Default]] = None

    # Goes over the children of the Compound below the Switch, adding them either directly below new_compound
    # or below the last Case as appropriate (for `switch(cond) {}`, block_items would have been None).
    for child in switch_node.stmt.block_items or []:
        if isinstance(child, (c_ast.Case, c_ast.Default)):
            # If it's a Case or Default:
            # 1. Add it to the Compound and mark as "last case"
            # 2. If its immediate child is also a Case or Default, promote it to a sibling.
            new_compound.block_items.append(child)
            while isinstance(child.stmts[0], (c_ast.Case, c_ast.Default)):
                new_compound.block_items.append(child.stmts.pop())
                child = new_compound.block_items[-1]  # noqa: PLW2901

            print(new_compound.block_items)
            latest_case = cast(Union[c_ast.Case, c_ast.Default], new_compound.block_items[-1])
        else:
            # Other statements are added as children to the last case, if it exists.
            if latest_case is not None:
                latest_case.stmts.append(child)
            else:
                new_compound.block_items.append(child)

    switch_node.stmt = new_compound
    return switch_node


# endregion


class CParser(Parser):
    tokens = CLexer.tokens
    precedence = (
        ("left", LOR),
        ("left", LAND),
        ("left", OR),
        ("left", XOR),
        ("left", AND),
        ("left", EQ, NE),
        ("left", GT, GE, LT, LE),
        ("left", RSHIFT, LSHIFT),
        ("left", PLUS, MINUS),
        ("left", TIMES, DIVIDE, MOD),
    )
    debugfile = "sly_cparser.out"

    subst = substitute(vars())

    def __init__(self, ctx: "c_context.CContext") -> None:
        self.ctx = ctx

    # ============================================================================
    # region ---- Scope helpers
    # ============================================================================

    def is_type_in_scope(self, name: str) -> bool:
        return self.ctx.scope_stack.get(name, False)

    def add_identifier_to_scope(self, name: str, coord: Optional[Coord]) -> None:
        """Add a new object, function, or enum member name (i.e. an ID) to the current scope."""

        if self.ctx.scope_stack.maps[0].get(name, False):
            msg = f"Non-typedef {name!r} previously declared as typedef in this scope."
            self.ctx.error(msg, coord)

        self.ctx.scope_stack[name] = False

    def add_typedef_name_to_scope(self, name: str, coord: Optional[Coord]) -> None:
        """Add a new typedef name (i.e. a TYPEID) to the current scope."""

        if not self.ctx.scope_stack.maps[0].get(name, True):
            msg = f"Typedef {name!r} previously declared as non-typedef in this scope."
            self.ctx.error(msg, coord)

        self.ctx.scope_stack[name] = True

    # endregion

    # ============================================================================
    # region ---- AST helpers
    # ============================================================================

    def _fix_decl_name_type(
        self,
        decl: _DeclarationT,
        typename: Union[list[Union[c_ast.IdType, c_ast.Typename]], list[c_ast.IdType]],
    ) -> _DeclarationT:
        """Fixes a declaration. Modifies decl."""

        # Reach the underlying basic type
        type_ = decl
        while not isinstance(type_, c_ast.TypeDecl):
            type_ = type_.type

        decl.name = type_.declname
        type_.quals = decl.quals[:]

        # The typename is a list of types. If any type in this list isn't an IdType,
        # it must be the only type in the list (it's illegal to declare "int enum ..").
        # If all the types are basic, they're collected in the IdType holder.
        for tn in typename:
            if not isinstance(tn, c_ast.IdType):
                if len(typename) > 1:
                    msg = "Invalid multiple types specified"
                    self.ctx.error(msg, tn.coord)
                else:
                    type_.type = tn
                    return decl

        typename = cast(list[c_ast.IdType], typename)  # We know the type at this point.
        if not typename:
            # Functions default to returning int
            if not isinstance(decl.type, c_ast.FuncDecl):
                msg = "Missing type in declaration"
                self.ctx.error(msg, decl.coord)

            type_.type = c_ast.IdType(["int"], coord=decl.coord)
        else:
            # At this point, we know that typename is a popoulated list of IdType nodes.
            # Concatenate all the names into a single list.
            type_.type = c_ast.IdType([name for id_ in typename for name in id_.names], coord=typename[0].coord)
        return decl

    def _build_declarations(
        self,
        spec: _DeclarationSpecifiers,
        decls: list[_StructDeclaratorDict],
        *,
        typedef_namespace: bool = False,
    ) -> list[c_ast.Decl]:
        """Builds a list of declarations all sharing the given specifiers.

        If typedef_namespace is true, each declared name is added to the "typedef namespace", which also includes
        objects, functions, and enum constants.
        """

        is_typedef = "typedef" in spec.storage
        declarations: list[c_ast.Decl] = []

        decls_0 = decls[0]

        if decls_0.get("bitsize") is not None:
            # Bit-fields are allowed to be unnamed.
            pass

        elif decls_0["decl"] is None:
            # When redeclaring typedef names as identifiers in inner scopes, a problem can occur where the identifier
            # gets grouped into spec.type, leaving decl as None. This can only occur for the first declarator.
            if len(spec.type) < 2 or len(spec.type[-1].names) != 1 or not self.is_type_in_scope(spec.type[-1].names[0]):
                coord = next((t.coord for t in spec.type if hasattr(t, "coord")), None)
                msg = "Invalid declaration"
                self.ctx.error(msg, coord)

            # Make this look as if it came from "direct_declarator:ID"
            decls_0["decl"] = c_ast.TypeDecl(
                declname=spec.type[-1].names[0],
                quals=None,
                align=spec.alignment,
                type=None,
                coord=spec.type[-1].coord,
            )
            # Remove the "new" type's name from the end of spec.type
            del spec.type[-1]

        elif not isinstance(decls_0["decl"], (c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdType)):
            # A similar problem can occur where the declaration ends up looking like an abstract declarator.
            # Give it a name if this is the case.
            decls_0_tail = decls_0["decl"]
            while not isinstance(decls_0_tail, c_ast.TypeDecl):
                decls_0_tail = decls_0_tail.type
            if decls_0_tail.declname is None:
                decls_0_tail.declname = spec.type.pop(-1).names[0]

        for decl in decls:
            typ = decl["decl"]
            assert typ is not None

            if is_typedef:
                declaration = c_ast.Typedef(None, spec.qual, spec.storage, typ, coord=typ.coord)
            else:
                declaration = c_ast.Decl(
                    None,
                    typ,
                    spec.qual,
                    spec.alignment,
                    spec.storage,
                    spec.function,
                    decl.get("init"),
                    decl.get("bitsize"),
                    coord=typ.coord,
                )

            if isinstance(declaration.type, (c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdType)):
                fixed_decl = declaration
            else:
                fixed_decl = self._fix_decl_name_type(declaration, spec.type)

            # Add the type name defined by typedef to a symbol table (for usage in the lexer)
            if typedef_namespace:
                assert isinstance(fixed_decl.name, str)
                if is_typedef:
                    self.add_typedef_name_to_scope(fixed_decl.name, fixed_decl.coord)
                else:
                    self.add_identifier_to_scope(fixed_decl.name, fixed_decl.coord)

            fixed_decl = fix_atomic_specifiers(fixed_decl)
            declarations.append(fixed_decl)

        return declarations

    def _build_function_definition(
        self,
        spec: _DeclarationSpecifiers,
        decl: Union[c_ast.TypeDecl, c_ast.TypeModifier, c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdType],
        param_decls: Optional[list[c_ast.Decl]],
        body: c_ast.Compound,
    ) -> c_ast.FuncDef:
        """Builds a function definition."""

        if "typedef" in spec.storage:
            msg = "Invalid typedef"
            self.ctx.error(msg, decl.coord)

        declaration = self._build_declarations(spec, decls=[{"decl": decl, "init": None}], typedef_namespace=True)[0]

        return c_ast.FuncDef(decl=declaration, param_decls=param_decls, body=body, coord=decl.coord)

    @overload
    def _type_modify_decl(self, decl: c_ast.TypeDecl, modifier: _ModifierT) -> _ModifierT: ...
    @overload
    def _type_modify_decl(self, decl: _ModifierT, modifier: c_ast.TypeModifier) -> _ModifierT: ...
    def _type_modify_decl(
        self,
        decl: Union[c_ast.TypeDecl, _ModifierT],
        modifier: Union[c_ast.TypeModifier, _ModifierT],
    ) -> _ModifierT:
        """Tacks a type modifier on a declarator, and returns the modified declarator.

        The declarator and modifier may be modified.

        Notes
        -----
        (This is basically an insertion into a linked list.)

        To understand what's going on here, read sections A.8.5 and A.8.6 of K&R2 very carefully.

        A C type consists of a basic type declaration, with a list of modifiers. For example:

            int *c[5];

        The basic declaration here is "int c", and the pointer and the array are the modifiers.

        Basic declarations are represented by `c_ast.TypeDecl`, and the modifiers are by
        `c_ast.FuncDecl`, `c_ast.PtrDecl` and `c_ast.ArrayDecl`.

        The standard states that whenever a new modifier is parsed, it should be added to the end of the list of
        modifiers. For example:

            K&R2 A.8.6.2: Array Declarators

            In a declaration T D where D has the form
                D1 [constant-expression-opt]
            and the type of the identifier in the declaration T D1 is "type-modifier T",
            the type of the identifier of D is "type-modifier array of T".

        This is what this method does. The declarator it receives can be a list of declarators ending with
        `c_ast.TypeDecl`. It tacks the modifier to the end of this list, just before the `TypeDecl`.

        Additionally, the modifier may be a list itself. This is useful for pointers, that can come as a chain from
        the rule "pointer". In this case, the whole modifier list is spliced into the new location.
        """

        modifier_head = modifier

        # The modifier may be a nested list. Reach its tail.
        modifier_tail = modifier
        while modifier_tail.type:
            modifier_tail = modifier_tail.type  # pyright: ignore

        if isinstance(decl, c_ast.TypeDecl):
            # If the decl is a basic type, just tack the modifier onto it.
            modifier_tail.type = decl  # pyright: ignore
            return modifier  # pyright: ignore
        else:
            # Otherwise, the decl is a list of modifiers.
            # Reach its tail and splice the modifier onto the tail, pointing to the underlying basic type.
            decl_tail = decl
            while not isinstance(decl_tail.type, c_ast.TypeDecl):
                decl_tail = decl_tail.type

            modifier_tail.type = decl_tail.type  # pyright: ignore
            decl_tail.type = modifier_head  # pyright: ignore
            return decl

    # endregion

    # ============================================================================
    # region ---- Grammar productions
    #
    # Implementation of the BNF defined in K&R2 A.13
    # ============================================================================

    @_("{ external_declaration }")
    def translation_unit(self, p: YaccProd):
        """Handle a translation unit.

        Notes
        -----
        This allows empty input. Not strictly part of the C99 Grammar, but useful in practice.
        """

        # NOTE: external_declaration is already a list, so now it's a list of lists.
        return c_ast.File([e for ext_decl in p.external_declaration for e in ext_decl])

    @_("function_definition")
    def external_declaration(self, p: YaccProd):
        """Handle an external declaration.

        Notes
        -----
        Declarations always come as lists (because they can be several in one line), so we wrap the function definition
        into a list as well, to make the return value of external_declaration homogeneous.
        """

        return [p.function_definition]

    @_("declaration")
    def external_declaration(self, p: YaccProd):
        return p.declaration

    @_("pp_directive", "pp_pragma_directive")
    def external_declaration(self, p: YaccProd):
        return [p[0]]

    @_('";"')
    def external_declaration(self, p: YaccProd) -> list[c_ast.AST]:
        return []

    @_("static_assert")
    def external_declaration(self, p: YaccProd):
        return [p.static_assert]

    @_('STATIC_ASSERT_ "(" constant_expression [ "," unified_string_literal ] ")"')
    def static_assert(self, p: YaccProd):
        return c_ast.StaticAssert(p.constant_expression, p.unified_string_literal, coord=Coord.from_prod(p, self))

    @_("PP_HASH")
    def pp_directive(self, p: YaccProd):
        msg = "Directives not supported yet"
        self.ctx.error(msg, Coord.from_prod(p, self))

    @_("PP_PRAGMA")
    def pp_pragma_directive(self, p: YaccProd):
        """Handle a preprocessor pragma directive or a _Pragma operator.

        Notes
        -----
        These encompass two types of C99-compatible pragmas:
        - The #pragma directive: "# pragma character_sequence"
        - The _Pragma unary operator: '_Pragma ( " string_literal " )'
        """

        return c_ast.Pragma("", coord=Coord.from_prod(p, self))

    @_("PP_PRAGMA PP_PRAGMASTR")
    def pp_pragma_directive(self, p: YaccProd):
        return c_ast.Pragma(p.PP_PRAGMASTR, coord=Coord.from_prod(p, self))

    @_('PRAGMA_ "(" unified_string_literal ")"')
    def pp_pragma_directive(self, p: YaccProd):
        return c_ast.Pragma(p.unified_string_literal, coord=Coord.from_prod(p, self))

    @_("pp_pragma_directive { pp_pragma_directive }")
    def pp_pragma_directive_list(self, p: YaccProd):
        return [p.pp_pragma_directive0, *p.pp_pragma_directive1]

    @_("[ declaration_specifiers ] id_declarator { declaration } compound_statement")
    def function_definition(self, p: YaccProd):
        """Handle a function declaration.

        Notes
        -----
        In function definitions, the declarator can be followed by a declaration list, for old "K&R style"
        function definitions.
        """

        print("---- Function declaration")

        if p.declaration_specifiers:
            print("decl specifiers exist")
            spec: _DeclarationSpecifiers = p.declaration_specifiers
        else:
            print("decl specifiers don't exist")
            # no declaration specifiers - "int" becomes the default type
            spec = _DeclarationSpecifiers(type=[c_ast.IdType(["int"], coord=p.id_declarator.coord)])

        return self._build_function_definition(
            spec=spec,
            decl=p.id_declarator,
            param_decls=[decl for decl_list in p.declaration for decl in decl_list] if p.declaration else None,
            body=p.compound_statement,
        )

    @_(
        "labeled_statement",
        "expression_statement",
        "compound_statement",
        "selection_statement",
        "iteration_statement",
        "jump_statement",
        "pp_pragma_directive",
        "static_assert",
    )
    def statement(self, p: YaccProd):
        """Handle a statement.

        Notes
        -----
        According to C18 A.2.2 6.7.10 static_assert-declaration, _Static_assert is a declaration, not a statement.
        We additionally recognise it as a statement to fix parsing of _Static_assert inside the functions.
        """

        return p[0]

    @_("pp_pragma_directive_list statement")
    def pragmacomp_or_statement(self, p: YaccProd):
        """Handles a pragma or a statement.

        Notes
        -----
        A pragma is generally considered a decorator rather than an actual statement. Still, for the purposes of
        analyzing an abstract syntax tree of C code, pragma's should not be ignored and were previously treated as a
        statement. This presents a problem for constructs that take a statement such as labeled_statements,
        selection_statements, and iteration_statements, causing a misleading structure in the AST. For example,
        consider the following C code.

            for (int i = 0; i < 3; i++)
                #pragma omp critical
                sum += 1;

        This code will compile and execute "sum += 1;" as the body of the for loop. Previous implementations of
        PyCParser would render the AST for this block of code as follows:

            For:
                DeclList:
                    Decl: i, [], [], []
                        TypeDecl: i, []
                            IdentifierType: ["int"]
                    Constant: int, 0
                BinaryOp: <
                    ID: i
                    Constant: int, 3
                UnaryOp: p++
                    ID: i
                Pragma: omp critical
            Assignment: +=
                ID: sum
                Constant: int, 1

        This AST misleadingly takes the Pragma as the body of the loop, and the assignment then becomes a sibling of
        the loop.

        To solve edge cases like these, the pragmacomp_or_statement rule groups a pragma and its following statement
        (which would otherwise be orphaned) using a compound block, effectively turning the above code into:

            for (int i = 0; i < 3; i++) {
                #pragma omp critical
                sum += 1;
            }
        """

        return c_ast.Compound(
            block_items=[*p.pp_pragma_directive_list, p.statement],
            coord=p.pp_pragma_directive_list.coord,
        )

    @_("statement")
    def pragmacomp_or_statement(self, p: YaccProd):
        return p.statement

    @_(
        "declaration_specifiers [ init_declarator_list ]",
        "declaration_specifiers_no_type [ id_init_declarator_list ]",
    )
    def decl_body(self, p: YaccProd):
        """Handle declaration bodies.

        Notes
        -----
        In C, declarations can come several in a line:

            int x, *px, romulo = 5;

        However, for the AST, we will split them to separate Declnodes.

        This rule splits its declarations and always returns a list of Decl nodes, even if it's one element long.
        """

        spec: _DeclarationSpecifiers = p[0]

        # p[1] is either a list or None
        #
        # NOTE: Accessing optional components via index puts the component in a 1-tuple,
        # so it's now being accessed with p[1][0].
        #
        declarator_list = p[1][0]
        if declarator_list is None:
            # By the standard, you must have at least one declarator unless declaring a structure tag, a union tag,
            # or the members of an enumeration.
            #
            spec_ty = spec.type
            if len(spec_ty) == 1 and isinstance(spec_ty[0], (c_ast.Struct, c_ast.Union, c_ast.Enum)):
                decls = [
                    c_ast.Decl(
                        None,
                        spec_ty[0],
                        quals=spec.qual,
                        align=spec.alignment,
                        storage=spec.storage,
                        funcspec=spec.function,
                        coord=spec_ty[0].coord,
                    )
                ]

            # However, this case can also occur on redeclared identifiers in an inner scope. The trouble is that
            # the redeclared type's name gets grouped into declaration_specifiers; _build_declarations compensates
            # for this.
            #
            else:
                decls = self._build_declarations(spec, decls=[{"decl": None, "init": None}], typedef_namespace=True)

        else:
            decls = self._build_declarations(spec, decls=declarator_list, typedef_namespace=True)

        return decls

    @_('decl_body ";"')
    def declaration(self, p: YaccProd):
        """Handle a declaration.

        Notes
        -----
        The declaration has been split to a decl_body sub-rule and ";", because having them in a single rule created a
        problem for defining typedefs.

        If a typedef line was directly followed by a line using the type defined with the typedef, the type would not
        be recognized. This is because to reduce the declaration rule, the parser's lookahead asked for the token
        after ";", which was the type from the next line, and the lexer had no chance to see the updated type symbol
        table.

        Splitting solves this problem, because after seeing ";", the parser reduces decl_body, which actually adds the
        new type into the table to be seen by the lexer before the next line is reached.
        """

        return p.decl_body

    @_("type_qualifier [ declaration_specifiers_no_type ]")
    def declaration_specifiers_no_type(self, p: YaccProd):
        """Handle declaration specifiers "without a type".

        Notes
        -----
        To know when declaration-specifiers end and declarators begin, we require the following:

        1. declaration-specifiers must have at least one type-specifier.
        2. No typedef-names are allowed after we've seen any type-specifier.

        These are both required by the spec.
        """

        return _DeclarationSpecifiers.add(p.declaration_specifiers_no_type, p[0], "qual")

    # region -- declaration_specifiers_no_type

    @_("storage_class_specifier [ declaration_specifiers_no_type ]")
    def declaration_specifiers_no_type(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers_no_type, p[0], "storage")

    @_("function_specifier [ declaration_specifiers_no_type ]")
    def declaration_specifiers_no_type(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers_no_type, p[0], "function")

    @_("atomic_specifier [ declaration_specifiers_no_type ]")
    def declaration_specifiers_no_type(self, p: YaccProd):
        # Without this, `typedef _Atomic(T) U` will parse incorrectly because the
        # _Atomic qualifier will match instead of the specifier.
        return _DeclarationSpecifiers.add(p.declaration_specifiers_no_type, p[0], "type")

    @_("alignment_specifier [ declaration_specifiers_no_type ]")
    def declaration_specifiers_no_type(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers_no_type, p[0], "alignment")

    # endregion

    # region -- declaration_specifiers

    @_("declaration_specifiers type_qualifier")
    def declaration_specifiers(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers, p[1], "qual", append=True)

    @_("declaration_specifiers storage_class_specifier")
    def declaration_specifiers(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers, p[1], "storage", append=True)

    @_("declaration_specifiers function_specifier")
    def declaration_specifiers(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers, p[1], "function", append=True)

    @_("declaration_specifiers type_specifier_no_typeid")
    def declaration_specifiers(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers, p[1], "type", append=True)

    @_("declaration_specifiers alignment_specifier")
    def declaration_specifiers(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers, p[1], "alignment", append=True)

    @_("type_specifier")
    def declaration_specifiers(self, p: YaccProd):
        return _DeclarationSpecifiers(type=[p.type_specifier])

    @_("declaration_specifiers_no_type type_specifier")
    def declaration_specifiers(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.declaration_specifiers_no_type, p.type_specifier, "type", append=True)

    # endregion

    @_("AUTO", "REGISTER", "STATIC", "EXTERN", "TYPEDEF", "THREAD_LOCAL_")
    def storage_class_specifier(self, p: YaccProd):
        return p[0]

    @_("INLINE", "NORETURN_")
    def function_specifier(self, p: YaccProd):
        return p[0]

    @_(
        "VOID",
        "BOOL_",
        "CHAR",
        "SHORT",
        "INT",
        "LONG",
        "FLOAT",
        "DOUBLE",
        "COMPLEX_",
        "SIGNED",
        "UNSIGNED",
        "INT128",
    )
    def type_specifier_no_typeid(self, p: YaccProd):
        return c_ast.IdType([p[0]], coord=Coord.from_prod(p, self))

    @_("typedef_name", "enum_specifier", "struct_or_union_specifier", "type_specifier_no_typeid", "atomic_specifier")
    def type_specifier(self, p: YaccProd):
        return p[0]

    @_('ATOMIC_ "(" type_name ")"')
    def atomic_specifier(self, p: YaccProd):
        """Handle an atomic specifier from C11.

        Notes
        -----
        See section 6.7.2.4 of the C11 standard.
        """

        typ: c_ast.Typename = p.type_name
        typ.quals.append(p.ATOMIC_)
        return typ

    @_("CONST", "RESTRICT", "VOLATILE", "ATOMIC_")
    def type_qualifier(self, p: YaccProd):
        return p[0]

    @_('init_declarator { "," init_declarator }')
    def init_declarator_list(self, p: YaccProd):
        return [p.init_declarator0, *p.init_declarator1]

    @_("declarator [ EQUALS initializer ]")
    def init_declarator(self, p: YaccProd):
        """Handle an init declarator.

        Returns
        -------
        dict[str, Any]
            A {decl=<declarator> : init=<initializer>} dictionary. If there's no initializer, uses None.
        """

        return {"decl": p.declarator, "init": p.initializer}

    @_('id_init_declarator { "," init_declarator }')
    def id_init_declarator_list(self, p: YaccProd):
        return [p.id_init_declarator, *p.init_declarator]

    @_("id_declarator [ EQUALS initializer ]")
    def id_init_declarator(self, p: YaccProd) -> _StructDeclaratorDict:
        return {"decl": p.id_declarator, "init": p.initializer}

    @_("specifier_qualifier_list type_specifier_no_typeid")
    def specifier_qualifier_list(self, p: YaccProd):
        """Handle a specifier qualifier list. At least one type specifier is required."""

        return _DeclarationSpecifiers.add(p.specifier_qualifier_list, p.type_specifier_no_typeid, "type", append=True)

    @_("specifier_qualifier_list type_qualifier")
    def specifier_qualifier_list(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.specifier_qualifier_list, p.type_qualifier, "qual", append=True)

    @_("type_specifier")
    def specifier_qualifier_list(self, p: YaccProd):
        return _DeclarationSpecifiers(type=[p.type_specifier])

    @_("type_qualifier_list type_specifier")
    def specifier_qualifier_list(self, p: YaccProd):
        return _DeclarationSpecifiers(qual=p.type_qualifier_list, type=[p.type_specifier])

    @_("alignment_specifier")
    def specifier_qualifier_list(self, p: YaccProd):
        return _DeclarationSpecifiers(alignment=[p.alignment_specifier])

    @_("specifier_qualifier_list alignment_specifier")
    def specifier_qualifier_list(self, p: YaccProd):
        return _DeclarationSpecifiers.add(p.specifier_qualifier_list, p.alignment_specifier, "alignment")

    @_("struct_or_union ID", "struct_or_union TYPEID")
    def struct_or_union_specifier(self, p: YaccProd):
        """Handle a struct-or-union specifier.

        Notes
        -----
        TYPEID is allowed here (and in other struct/enum related tag names), because
        struct/enum tags reside in their own namespace and can be named the same as types.
        """

        klass = c_ast.Struct if (p.struct_or_union == "struct") else c_ast.Union
        # None means no list of members
        return klass(name=p[1], decls=None, coord=Coord.from_prod(p, self))

    @_('struct_or_union "{" { struct_declaration } "}"')
    def struct_or_union_specifier(self, p: YaccProd):
        klass = c_ast.Struct if (p.struct_or_union == "struct") else c_ast.Union
        # Empty sequence means an empty list of members
        decls = [decl for decl_list in p.struct_declaration for decl in decl_list if decl is not None]
        return klass(name=None, decls=decls, coord=Coord.from_prod(p, self))

    @_(
        'struct_or_union ID "{" { struct_declaration } "}"',
        'struct_or_union TYPEID "{" { struct_declaration } "}"',
    )
    def struct_or_union_specifier(self, p: YaccProd):
        klass = c_ast.Struct if (p.struct_or_union == "struct") else c_ast.Union
        # Empty sequence means an empty list of members
        decls = [decl for decl_list in p.struct_declaration for decl in decl_list]
        return klass(name=p[1], decls=decls, coord=Coord.from_prod(p, self))

    @_("STRUCT", "UNION")
    def struct_or_union(self, p: YaccProd):
        return p[0]

    @_('specifier_qualifier_list [ struct_declarator_list ] ";"')
    def struct_declaration(self, p: YaccProd):
        spec: _DeclarationSpecifiers = p.specifier_qualifier_list

        if "typedef" in spec.storage:
            raise AssertionError

        if p.struct_declarator_list is not None:
            decls = self._build_declarations(spec, decls=p.struct_declarator_list)

        elif len(spec.type) == 1:
            # Anonymous struct/union: gcc extension, C1x feature.
            # Although the standard only allows structs/unions here, I see no reason to disallow other types since
            # some compilers have typedefs here, and pycparser isn't about rejecting all invalid code.
            #
            node = spec.type[0]
            decl_type = node if isinstance(node, c_ast.AST) else c_ast.IdType(node)
            decls = self._build_declarations(spec, decls=[{"decl": decl_type}])

        else:
            # Structure/union members can have the same names as typedefs. The trouble is that the member's name gets
            # grouped into specifier_qualifier_list; _build_declarations() compensates.
            #
            decls = self._build_declarations(spec, decls=[{"decl": None, "init": None}])

        return decls

    @_('";"')
    def struct_declaration(self, p: YaccProd) -> list[c_ast.AST]:
        return []

    @_("pp_pragma_directive")
    def struct_declaration(self, p: YaccProd):
        return [p.pp_pragma_directive]

    @_('struct_declarator { "," struct_declarator }')
    def struct_declarator_list(self, p: YaccProd):
        return [p.struct_declarator0, *p.struct_declarator1]

    @_("declarator")
    def struct_declarator(self, p: YaccProd) -> _StructDeclaratorDict:
        """Handle a struct declarator.

        Returns
        -------
        _StructDeclaratorDict
            A dict with the keys "decl", for the underlying declarator, and "bitsize", for the bitsize.
        """

        return {"decl": p.declarator, "bitsize": None}

    @_('declarator ":" constant_expression')
    def struct_declarator(self, p: YaccProd) -> _StructDeclaratorDict:
        return {"decl": p.declarator, "bitsize": p.constant_expression}

    @_('":" constant_expression')
    def struct_declarator(self, p: YaccProd) -> _StructDeclaratorDict:
        return {"decl": c_ast.TypeDecl(quals=None), "bitsize": p.constant_expression}

    @_("ENUM ID", "ENUM TYPEID")
    def enum_specifier(self, p: YaccProd):
        return c_ast.Enum(p[1], None, coord=Coord.from_prod(p, self))

    @_('ENUM "{" enumerator_list "}"')
    def enum_specifier(self, p: YaccProd):
        return c_ast.Enum(None, p.enumerator_list, coord=Coord.from_prod(p, self))

    @_(
        'ENUM ID "{" enumerator_list "}"',
        'ENUM TYPEID "{" enumerator_list "}"',
    )
    def enum_specifier(self, p: YaccProd):
        return c_ast.Enum(p[1], p.enumerator_list, coord=Coord.from_prod(p, self))

    @_("enumerator")
    def enumerator_list(self, p: YaccProd):
        return c_ast.EnumeratorList([p.enumerator], coord=p.enumerator.coord)

    @_('enumerator_list ","')
    def enumerator_list(self, p: YaccProd):
        return p.enumerator_list

    @_('enumerator_list "," enumerator')
    def enumerator_list(self, p: YaccProd):
        p.enumerator_list.enumerators.append(p.enumerator)
        return p.enumerator_list

    @_(
        'ALIGNAS_ "(" type_name ")"',
        'ALIGNAS_ "(" constant_expression ")"',
    )
    def alignment_specifier(self, p: YaccProd):
        return c_ast.Alignas(p[2], coord=Coord.from_prod(p, self))

    @_("ID [ EQUALS constant_expression ]")
    def enumerator(self, p: YaccProd):
        enumerator = c_ast.Enumerator(p.ID, p.constant_expression, coord=Coord.from_prod(p, self))
        self.add_identifier_to_scope(enumerator.name, enumerator.coord)
        return enumerator

    @_("id_declarator", "typeid_declarator")
    def declarator(self, p: YaccProd):
        return p[0]

    # ========
    # region -- Experimental usage of `subst()` for $$$_declarator and direct_$$$_declarator rules
    #
    # Note: $$$ is substituted with id, typeid, and typeid_noparen, depending on the rule.
    # ========

    # fmt: off
    subst_ids = subst(
        {"_SUB1": "id",              "_SUB2": "ID"},
        {"_SUB1": "typeid",          "_SUB2": "TYPEID"},
        {"_SUB1": "typeid_noparen",  "_SUB2": "TYPEID"},
    )
    # fmt: on

    @subst_ids
    @_("direct_${_SUB1}_declarator")
    def _SUB1_declarator(self, p: YaccProd) -> Union[c_ast.TypeDecl, c_ast.TypeModifier]:
        return p[0]

    @subst_ids
    @_("pointer direct_${_SUB1}_declarator")
    def _SUB1_declarator(self, p: YaccProd) -> Union[c_ast.TypeDecl, c_ast.TypeModifier]:
        return self._type_modify_decl(p[1], p.pointer)

    @subst_ids
    @_("${_SUB2}")
    def direct__SUB1_declarator(self, p: YaccProd):
        return c_ast.TypeDecl(declname=p[0], quals=None, coord=Coord.from_prod(p, self))

    @subst({"_SUB1": "id"}, {"_SUB1": "typeid"})
    @_('"(" ${_SUB1}_declarator ")"')
    def direct__SUB1_declarator(self, p: YaccProd):
        return p[1]

    @subst_ids
    @_('direct_${_SUB1}_declarator "[" [ type_qualifier_list ] [ assignment_expression ] "]"')
    def direct__SUB1_declarator(self, p: YaccProd):
        # Accept dimension qualifiers
        # Per C99 6.7.5.3 p7
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=p.assignment_expression,
            dim_quals=p.type_qualifier_list or [],
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @subst_ids
    @_('direct_${_SUB1}_declarator "[" STATIC [ type_qualifier_list ] assignment_expression "]"')
    def direct__SUB1_declarator(self, p: YaccProd):
        if p.type_qualifier_list is not None:
            dim_quals = [qual for qual in (p.STATIC, *p.type_qualifier_list) if qual is not None]
        else:
            dim_quals = [p.STATIC]
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=p.assignment_expression,
            dim_quals=dim_quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @subst_ids
    @_('direct_${_SUB1}_declarator "[" type_qualifier_list STATIC assignment_expression "]"')
    def direct__SUB1_declarator(self, p: YaccProd):
        dim_quals = [qual for qual in (*p.type_qualifier_list, p.STATIC) if qual is not None]
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=p.assignment_expression,
            dim_quals=dim_quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @subst_ids
    @_('direct_${_SUB1}_declarator "[" [ type_qualifier_list ] TIMES "]"')
    def direct__SUB1_declarator(self, p: YaccProd):
        """Special for VLAs."""

        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=c_ast.Id(p.TIMES, coord=p[0].coord),
            dim_quals=p.type_qualifier_list or [],
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @subst_ids
    @_(
        'direct_${_SUB1}_declarator "(" parameter_type_list ")"',
        'direct_${_SUB1}_declarator "(" [ identifier_list ] ")"',
    )
    def direct__SUB1_declarator(self, p: YaccProd):
        # print("---- here 9")
        # print(f"{self.lookahead=}")

        # NOTE: This first line depends on an implementation detail -- optional components are in tuples when accessed
        # with numerical index -- to determine the difference.
        args = p[2] if not isinstance(p[2], tuple) else p[2][0]
        func = c_ast.FuncDecl(args, type=None, coord=p[0].coord)  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.

        # To see why the lookahead token is needed, consider:
        #   typedef char TT;
        #   void foo(int TT) { TT = 10; }
        # Outside the function, TT is a typedef, but inside (starting and ending with the braces) it's a parameter.
        # The trouble begins with yacc's lookahead token. We don't know if we're declaring or defining a function
        # until we see "{", but if we wait for yacc to trigger a rule on that token, then TT will have already been
        # read and incorrectly interpreted as TYPEID.
        # We need to add the parameters to the scope the moment the lexer sees "{".
        #
        if (self.lookahead is not None) and (self.lookahead.type == "{") and (func.args is not None):
            for param in func.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    break

                self.add_identifier_to_scope(param.name, param.coord)

        return self._type_modify_decl(decl=p[0], modifier=func)

    del subst_ids  # Explicit cleanup required.

    # endregion

    @_("TIMES [ type_qualifier_list ] [ pointer ]")
    def pointer(self, p: YaccProd) -> c_ast.PtrDecl:
        """Handle a pointer.

        Notes
        -----
        Pointer decls nest from inside out. This is important when different levels have different qualifiers.
        For example:

            char * const * p;

        Means "pointer to const pointer to char"

        While:

            char ** const p;

        Means "const pointer to pointer to char"

        So when we construct PtrDecl nestings, the leftmost pointer goes in as the most nested type.
        """

        nested_type = c_ast.PtrDecl(
            quals=p.type_qualifier_list or [],
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            coord=Coord.from_prod(p, self),
        )

        pointer: Optional[c_ast.PtrDecl] = p.pointer
        if pointer is not None:
            tail_type = pointer
            while tail_type.type is not None:
                tail_type = tail_type.type
            tail_type.type = nested_type  # pyright: ignore
            return pointer
        else:
            return nested_type

    @_("type_qualifier { type_qualifier }")
    def type_qualifier_list(self, p: YaccProd):
        return [p.type_qualifier0, *p.type_qualifier1]

    @_("parameter_list")
    def parameter_type_list(self, p: YaccProd):
        return p.parameter_list

    @_('parameter_list "," ELLIPSIS')
    def parameter_type_list(self, p: YaccProd):
        p.parameter_list.params.append(c_ast.EllipsisParam(coord=Coord.from_prod(p, self)))
        return p.parameter_list

    @_('parameter_declaration { "," parameter_declaration }')
    def parameter_list(self, p: YaccProd):
        coord = p.parameter_declaration0.coord
        return c_ast.ParamList([p.parameter_declaration0, *p.parameter_declaration1], coord=coord)

    @_(
        "declaration_specifiers id_declarator",
        "declaration_specifiers typeid_noparen_declarator",
    )
    def parameter_declaration(self, p: YaccProd):
        """Handle a parameter declaration.

        Notes
        -----
        From ISO/IEC 9899:TC2, 6.7.5.3.11:

            "If, in a parameter declaration, an identifier can be treated either as a typedef name or as a
            parameter name, it shall be taken as a typedef name."

        Inside a parameter declaration, once we've reduced declaration specifiers, if we shift in an "(" and see
        a TYPEID, it could be either an abstract declarator or a declarator nested inside parens. This rule tells us to
        always treat it as an abstract declarator. Therefore, we only accept `id_declarator`s and
        `typeid_noparen_declarator`s.
        """

        spec: _DeclarationSpecifiers = p.declaration_specifiers
        if not spec.type:
            spec.type.append(c_ast.IdType(["int"], coord=Coord.from_node(p.declaration_specifiers, self)))
        return self._build_declarations(spec, decls=[{"decl": p[1]}])[0]

    @_("declaration_specifiers [ abstract_declarator ]")
    def parameter_declaration(self, p: YaccProd):
        spec: _DeclarationSpecifiers = p.declaration_specifiers
        if not spec.type:
            spec.type.append(c_ast.IdType(["int"], coord=Coord.from_node(p.declaration_specifiers, self)))

        # Parameters can have the same names as typedefs. The trouble is that the parameter's name gets grouped into
        # declaration_specifiers, making it look like an old-style declaration; compensate.
        if len(spec.type) > 1 and len(spec.type[-1].names) == 1 and self.is_type_in_scope(spec.type[-1].names[0]):
            decl = self._build_declarations(spec, decls=[{"decl": p.abstract_declarator, "init": None}])[0]

        # This truly is an old-style parameter declaration.
        else:
            decl = c_ast.Typename(
                name="",
                quals=spec.qual,
                align=None,
                type=p.abstract_declarator or c_ast.TypeDecl(quals=None),
                coord=Coord.from_node(p.declaration_specifiers, self),
            )
            typename = spec.type
            decl = self._fix_decl_name_type(decl, typename)

        return decl

    @_('identifier { "," identifier }')
    def identifier_list(self, p: YaccProd):
        return c_ast.ParamList([p.identifier0, *p.identifier1], coord=p.identifier0.coord)

    @_("assignment_expression")
    def initializer(self, p: YaccProd):
        return p.assignment_expression

    @_(
        '"{" [ initializer_list ] "}"',
        '"{" initializer_list "," "}"',
    )
    def initializer(self, p: YaccProd):
        if p.initializer_list is not None:
            return p.initializer_list
        else:
            return c_ast.InitList([], coord=Coord.from_prod(p, self))

    @_("[ designation ] initializer")
    def initializer_list(self, p: YaccProd):
        init = p.initializer if (p.designation is None) else c_ast.NamedInitializer(p.designation, p.initializer)
        return c_ast.InitList([init], coord=p.initializer.coord)

    @_('initializer_list "," [ designation ] initializer')
    def initializer_list(self, p: YaccProd):
        init = p.initializer if (p.designation is None) else c_ast.NamedInitializer(p.designation, p.initializer)
        p.initializer_list.exprs.append(init)
        return p.initializer_list

    @_("designator_list EQUALS")
    def designation(self, p: YaccProd):
        return p.designator_list

    @_("designator { designator }")
    def designator_list(self, p: YaccProd):
        """Handle a list of designators.

        Notes
        -----
        Designators are represented as a list of nodes, in the order in which
        they're written in the code.
        """

        return [p.designator0, *p.designator1]

    @_('"[" constant_expression "]"', '"." identifier')
    def designator(self, p: YaccProd):
        return p[1]

    @_("specifier_qualifier_list [ abstract_declarator ]")
    def type_name(self, p: YaccProd):
        spec_list: _DeclarationSpecifiers = p.specifier_qualifier_list
        typename = c_ast.Typename(
            name="",
            quals=spec_list.qual[:],
            align=None,
            type=p.abstract_declarator or c_ast.TypeDecl(quals=None),
            coord=Coord.from_node(spec_list, self),
        )

        return self._fix_decl_name_type(typename, spec_list.type)

    @_("pointer")
    def abstract_declarator(self, p: YaccProd):
        dummytype = c_ast.TypeDecl(quals=None)
        return self._type_modify_decl(decl=dummytype, modifier=p.pointer)

    @_("pointer direct_abstract_declarator")
    def abstract_declarator(self, p: YaccProd):
        return self._type_modify_decl(p.direct_abstract_declarator, p.pointer)

    @_("direct_abstract_declarator")
    def abstract_declarator(self, p: YaccProd):
        return p.direct_abstract_declarator

    # Creating and using direct_abstract_declarator_opt here
    # instead of listing both direct_abstract_declarator and the
    # lack of it in the beginning of _1 and _2 caused two
    # shift/reduce errors.
    #
    @_('"(" abstract_declarator ")"')
    def direct_abstract_declarator(self, p: YaccProd):
        return p.abstract_declarator

    @_('direct_abstract_declarator "[" [ assignment_expression ] "]"')
    def direct_abstract_declarator(self, p: YaccProd):
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=p.assignment_expression,
            dim_quals=[],
            coord=p.direct_abstract_declarator.coord,
        )

        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=arr)

    @_('"[" [ type_qualifier_list ] [ assignment_expression ] "]"')
    def direct_abstract_declarator(self, p: YaccProd):
        return c_ast.ArrayDecl(
            type=c_ast.TypeDecl(quals=None),
            dim=p.assignment_expression,
            dim_quals=p.type_qualifier_list or [],
            coord=Coord.from_prod(p, self),
        )

    @_('direct_abstract_declarator "[" TIMES "]"')
    def direct_abstract_declarator(self, p: YaccProd):
        arr = c_ast.ArrayDecl(
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            dim=c_ast.Id(p.TIMES, coord=Coord.from_prod(p, self)),
            dim_quals=[],
            coord=p.direct_abstract_declarator.coord,
        )

        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=arr)

    @_('"[" TIMES "]"')
    def direct_abstract_declarator(self, p: YaccProd):
        return c_ast.ArrayDecl(
            type=c_ast.TypeDecl(quals=None),
            dim=c_ast.Id(p[2], coord=Coord.from_prod(p, self)),
            dim_quals=[],
            coord=Coord.from_prod(p, self),
        )

    @_('direct_abstract_declarator "(" [ parameter_type_list ] ")"')
    def direct_abstract_declarator(self, p: YaccProd):
        func = c_ast.FuncDecl(
            args=p.parameter_type_list,
            type=None,  # pyright: ignore [reportArgumentType] # Gets fixed in _type_modify_decl.
            coord=p.direct_abstract_declarator.coord,
        )
        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=func)

    @_('"(" [ parameter_type_list ] ")"')
    def direct_abstract_declarator(self, p: YaccProd):
        return c_ast.FuncDecl(
            args=p.parameter_type_list,
            type=c_ast.TypeDecl(quals=None),
            coord=Coord.from_prod(p, self),
        )

    @_("declaration", "statement")
    def block_item(self, p: YaccProd) -> list[c_ast.AST]:
        """Handle a block item.

        Notes
        -----
        declaration is a list, statement isn't. To make it consistent, block_item will always be a list.
        """

        item = p[0]
        return item if isinstance(item, list) else [item]  # pyright: ignore [reportUnknownVariableType]

    @_('"{" { block_item } "}"')
    def compound_statement(self, p: YaccProd):
        """Handle a compound statement.

        Notes
        -----
        Since we made block_item a list, this just combines lists. Empty block items (plain ";") produce `[None]`,
        so ignore them.
        """

        print("---- function body")
        block_items = [item for item_list in p.block_item for item in item_list if item is not None]
        return c_ast.Compound(block_items=block_items, coord=Coord.from_prod(p, self))

    @_('ID ":" pragmacomp_or_statement')
    def labeled_statement(self, p: YaccProd):
        return c_ast.Label(p.ID, p.pragmacomp_or_statement, coord=Coord.from_prod(p, self))

    @_('CASE constant_expression ":" pragmacomp_or_statement')
    def labeled_statement(self, p: YaccProd):
        return c_ast.Case(p.constant_expression, [p.pragmacomp_or_statement], coord=Coord.from_prod(p, self))

    @_('DEFAULT ":" pragmacomp_or_statement')
    def labeled_statement(self, p: YaccProd):
        return c_ast.Default([p.pragmacomp_or_statement], coord=Coord.from_prod(p, self))

    @_('IF "(" expression ")" pragmacomp_or_statement')
    def selection_statement(self, p: YaccProd):
        return c_ast.If(p[2], p[4], None, coord=Coord.from_prod(p, self))

    @_('IF "(" expression ")" statement ELSE pragmacomp_or_statement')
    def selection_statement(self, p: YaccProd):
        return c_ast.If(p[2], p[4], p[6], coord=Coord.from_prod(p, self))

    @_('SWITCH "(" expression ")" pragmacomp_or_statement')
    def selection_statement(self, p: YaccProd):
        return fix_switch_cases(c_ast.Switch(p.expression, p.pragmacomp_or_statement, coord=Coord.from_prod(p, self)))

    @_('WHILE "(" expression ")" pragmacomp_or_statement')
    def iteration_statement(self, p: YaccProd):
        return c_ast.While(p.expression, p.pragmacomp_or_statement, coord=Coord.from_prod(p, self))

    @_('DO pragmacomp_or_statement WHILE "(" expression ")" ";"')
    def iteration_statement(self, p: YaccProd):
        return c_ast.DoWhile(p.expression, p.pragmacomp_or_statement, coord=Coord.from_prod(p, self))

    @_('FOR "(" [ expression ] ";" [ expression ] ";" [ expression ] ")" pragmacomp_or_statement')
    def iteration_statement(self, p: YaccProd):
        return c_ast.For(p.expression0, p.expression1, p.expression2, p[8], coord=Coord.from_prod(p, self))

    @_('FOR "(" declaration [ expression ] ";" [ expression ] ")" pragmacomp_or_statement')
    def iteration_statement(self, p: YaccProd):
        coord = Coord.from_prod(p, self)
        return c_ast.For(
            c_ast.DeclList(p.declaration, coord=coord),
            p.expression0,
            p.expression1,
            p.expression2,
            coord=coord,
        )

    @_('GOTO ID ";"')
    def jump_statement(self, p: YaccProd):
        return c_ast.Goto(p.ID, coord=Coord.from_prod(p, self))

    @_('BREAK ";"')
    def jump_statement(self, p: YaccProd):
        return c_ast.Break(coord=Coord.from_prod(p, self))

    @_('CONTINUE ";"')
    def jump_statement(self, p: YaccProd):
        return c_ast.Continue(coord=Coord.from_prod(p, self))

    @_('RETURN [ expression ] ";"')
    def jump_statement(self, p: YaccProd):
        return c_ast.Return(p.expression, coord=Coord.from_prod(p, self))

    @_('[ expression ] ";"')
    def expression_statement(self, p: YaccProd):
        if p.expression is not None:
            return p.expression
        else:
            return c_ast.EmptyStatement(coord=Coord.from_prod(p, self))

    @_("assignment_expression")
    def expression(self, p: YaccProd):
        return p.assignment_expression

    @_('expression "," assignment_expression')
    def expression(self, p: YaccProd):
        if not isinstance(p.expression, c_ast.ExprList):
            p.expression = c_ast.ExprList([p.expression], coord=p.expression.coord)

        p.expression.exprs.append(p.assignment_expression)
        return p.expression

    @_("TYPEID")
    def typedef_name(self, p: YaccProd):
        return c_ast.IdType([p.TYPEID], coord=Coord.from_prod(p, self))

    @_('"(" compound_statement ")"')
    def assignment_expression(self, p: YaccProd):
        # TODO: Verify that the original name "parenthesized_compound_expression", isn't meaningful.
        return p.compound_statement

    @_("conditional_expression")
    def assignment_expression(self, p: YaccProd):
        return p.conditional_expression

    @_("unary_expression assignment_operator assignment_expression")
    def assignment_expression(self, p: YaccProd):
        coord = p.assignment_operator.coord
        return c_ast.Assignment(p.assignment_operator, p.unary_expression, p.assignment_expression, coord=coord)

    @_(
        "EQUALS",
        "XOREQUAL",
        "TIMESEQUAL",
        "DIVEQUAL",
        "MODEQUAL",
        "PLUSEQUAL",
        "MINUSEQUAL",
        "LSHIFTEQUAL",
        "RSHIFTEQUAL",
        "ANDEQUAL",
        "OREQUAL",
    )
    def assignment_operator(self, p: YaccProd):
        """Handle assignment operators.

        Notes
        -----
        K&R2 defines these as many separate rules, to encode precedence and associativity. However, in our case,
        SLY's built-in precedence/associativity specification feature can take care of it (see `CParser.precedence`).
        """

        return p[0]

    @_("conditional_expression")
    def constant_expression(self, p: YaccProd):
        return p.conditional_expression

    @_("binary_expression")
    def conditional_expression(self, p: YaccProd):
        return p.binary_expression

    @_('binary_expression CONDOP expression ":" conditional_expression')
    def conditional_expression(self, p: YaccProd):
        coord = p.binary_expression.coord
        return c_ast.TernaryOp(p.binary_expression, p.expression, p.conditional_expression, coord=coord)

    @_("cast_expression")
    def binary_expression(self, p: YaccProd):
        return p.cast_expression

    @_(
        "binary_expression TIMES binary_expression",
        "binary_expression DIVIDE binary_expression",
        "binary_expression MOD binary_expression",
        "binary_expression PLUS binary_expression",
        "binary_expression MINUS binary_expression",
        "binary_expression RSHIFT binary_expression",
        "binary_expression LSHIFT binary_expression",
        "binary_expression LT binary_expression",
        "binary_expression LE binary_expression",
        "binary_expression GE binary_expression",
        "binary_expression GT binary_expression",
        "binary_expression EQ binary_expression",
        "binary_expression NE binary_expression",
        "binary_expression AND binary_expression",
        "binary_expression OR binary_expression",
        "binary_expression XOR binary_expression",
        "binary_expression LAND binary_expression",
        "binary_expression LOR binary_expression",
    )
    def binary_expression(self, p: YaccProd):
        return c_ast.BinaryOp(p[1], p[0], p[2], coord=p[0].coord)

    @_("unary_expression")
    def cast_expression(self, p: YaccProd):
        return p.unary_expression

    @_('"(" type_name ")" cast_expression')
    def cast_expression(self, p: YaccProd):
        return c_ast.Cast(p.type_name, p.cast_expression, coord=Coord.from_prod(p, self))

    @_("postfix_expression")
    def unary_expression(self, p: YaccProd):
        return p.postfix_expression

    @_("PLUSPLUS unary_expression", "MINUSMINUS unary_expression", "unary_operator cast_expression")
    def unary_expression(self, p: YaccProd):
        return c_ast.UnaryOp(p[0], p[1], coord=p[1].coord)

    @_("SIZEOF unary_expression")
    def unary_expression(self, p: YaccProd):
        return c_ast.UnaryOp(p[0], p[1], coord=Coord.from_prod(p, self))

    @_('SIZEOF "(" type_name ")"', 'ALIGNOF_ "(" type_name ")"')
    def unary_expression(self, p: YaccProd):
        return c_ast.UnaryOp(p[0], p.type_name, coord=Coord.from_prod(p, self))

    @_("AND", "TIMES", "PLUS", "MINUS", "NOT", "LNOT")
    def unary_operator(self, p: YaccProd):
        return p[0]

    @_("primary_expression")
    def postfix_expression(self, p: YaccProd):
        return p.primary_expression

    @_('postfix_expression "[" expression "]"')
    def postfix_expression(self, p: YaccProd):
        return c_ast.ArrayRef(p.postfix_expression, p.expression, coord=p.postfix_expression.coord)

    @_('postfix_expression "(" assignment_expression { "," assignment_expression } ")"')
    def postfix_expression(self, p: YaccProd):
        arg_expr = c_ast.ExprList(
            [p.assignment_expression0, *p.assignment_expression1],
            coord=p.assignment_expression0.coord,
        )
        return c_ast.FuncCall(p.postfix_expression, arg_expr, coord=p.postfix_expression.coord)

    @_(
        'postfix_expression "." ID',
        'postfix_expression "." TYPEID',
        "postfix_expression ARROW ID",
        "postfix_expression ARROW TYPEID",
    )
    def postfix_expression(self, p: YaccProd):
        field = c_ast.Id(p[2], coord=Coord.from_prod(p, self))
        return c_ast.StructRef(p.postfix_expression, p[1], field, coord=p.postfix_expression.coord)

    @_("postfix_expression PLUSPLUS", "postfix_expression MINUSMINUS")
    def postfix_expression(self, p: YaccProd):
        return c_ast.UnaryOp("p" + p[1], p.postfix_expression, coord=p[1].coord)

    @_('"(" type_name ")" "{" initializer_list [ "," ] "}"')
    def postfix_expression(self, p: YaccProd):
        return c_ast.CompoundLiteral(p.type_name, p.initializer_list)

    @_("identifier", "constant", "unified_string_literal", "unified_wstring_literal")
    def primary_expression(self, p: YaccProd):
        return p[0]

    @_('"(" expression ")"')
    def primary_expression(self, p: YaccProd):
        return p.expression

    @_('OFFSETOF "(" type_name "," offsetof_member_designator ")"')
    def primary_expression(self, p: YaccProd):
        coord = Coord.from_prod(p, self)
        return c_ast.FuncCall(
            c_ast.Id(p.OFFSETOF, coord=coord),
            c_ast.ExprList([p.type_name, p.offsetof_member_designator], coord=coord),
            coord=coord,
        )

    @_("identifier")
    def offsetof_member_designator(self, p: YaccProd):
        return p.identifier

    @_('offsetof_member_designator "." identifier')
    def offsetof_member_designator(self, p: YaccProd):
        coord = p.offsetof_member_designator.coord
        return c_ast.StructRef(p.offsetof_member_designator, p[1], p.identifer, coord=coord)

    @_('offsetof_member_designator "[" expression "]"')
    def offsetof_member_designator(self, p: YaccProd):
        return c_ast.ArrayRef(p.offsetof_member_designator, p.expression, coord=p.offsetof_member_designator.coord)

    @_("ID")
    def identifier(self, p: YaccProd):
        return c_ast.Id(p.ID, coord=Coord.from_prod(p, self))

    @_("INT_CONST_DEC", "INT_CONST_OCT", "INT_CONST_HEX", "INT_CONST_BIN", "INT_CONST_CHAR")
    def constant(self, p: YaccProd):
        uCount = 0
        lCount = 0
        for x in p[0][-3:]:
            if x in {"l", "L"}:
                lCount += 1
            elif x in {"u", "U"}:
                uCount += 1

        if uCount > 1:
            msg = "Constant cannot have more than one u/U suffix."
            raise ValueError(msg)
        if lCount > 2:
            msg = "Constant cannot have more than two l/L suffix."
            raise ValueError(msg)
        prefix = "unsigned " * uCount + "long " * lCount
        return c_ast.Constant(prefix + "int", p[0], coord=Coord.from_prod(p, self))

    @_("FLOAT_CONST", "HEX_FLOAT_CONST")
    def constant(self, p: YaccProd):
        if "x" in p[0].lower():
            t = "float"
        else:
            if p[0][-1] in {"f", "F"}:
                t = "float"
            elif p[0][-1] in {"l", "L"}:
                t = "long double"
            else:
                t = "double"

        return c_ast.Constant(t, p[0], coord=Coord.from_prod(p, self))

    @_("CHAR_CONST", "WCHAR_CONST", "U8CHAR_CONST", "U16CHAR_CONST", "U32CHAR_CONST")
    def constant(self, p: YaccProd):
        return c_ast.Constant("char", p[0], coord=Coord.from_prod(p, self))

    @_("STRING_LITERAL")
    def unified_string_literal(self, p: YaccProd):
        """Handle "unified" string literals.

        Notes
        -----
        The "unified" string and wstring literal rules are for supporting concatenation of adjacent string literals.
        For example, `"hello " "world"` is seen by the C compiler as a single string literal with the value
        "hello world".
        """

        # single literal
        return c_ast.Constant("string", p[0], coord=Coord.from_prod(p, self))

    @_("unified_string_literal STRING_LITERAL")
    def unified_string_literal(self, p: YaccProd):
        p.unified_string_literal.value = p.unified_string_literal.value[:-1] + p.STRING_LITERAL[1:]
        return p.unified_string_literal

    @_(
        "WSTRING_LITERAL",
        "U8STRING_LITERAL",
        "U16STRING_LITERAL",
        "U32STRING_LITERAL",
    )
    def unified_wstring_literal(self, p: YaccProd):
        return c_ast.Constant("string", p[0], coord=Coord.from_prod(p, self))

    @_(
        "unified_wstring_literal WSTRING_LITERAL",
        "unified_wstring_literal U8STRING_LITERAL",
        "unified_wstring_literal U16STRING_LITERAL",
        "unified_wstring_literal U32STRING_LITERAL",
    )
    def unified_wstring_literal(self, p: YaccProd):
        p.unified_wstring_literal.value = p.unified_wstring_literal.value.rstrip()[:-1] + p[1][2:]
        og_col_end: Optional[int] = p.unified_wstring_literal.coord.col_end
        if og_col_end is not None:
            p.unified_wstring_literal.coord.col_end = og_col_end + len(p.unified_wstring_literal.value)
        return p.unified_wstring_literal

    # endregion

    def error(self, token: Optional[Union[Token, YaccSymbol]]) -> NoReturn:
        if token:
            msg = "Syntax error."
            location = Coord(getattr(token, "lineno", 0), token.index, None, token.end)  # type: ignore
        else:
            msg = "Parse error in input. EOF."
            location = Coord(-1, -1)

        self.ctx.error(msg, location, token)
