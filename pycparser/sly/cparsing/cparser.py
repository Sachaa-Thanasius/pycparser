# pyright: reportRedeclaration=none
# ruff: noqa: ANN201, F811, S105, RET505
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pycparser import c_ast
from pycparser.sly import Parser
from pycparser.sly.cparsing.clexer import CLexer

if TYPE_CHECKING:
    from typing import Callable, Protocol, TypeVar, runtime_checkable

    CallableT = TypeVar("CallableT")

    @runtime_checkable
    class _RuleDecorator(Protocol):
        def __call__(self, rule: str, *extras: str) -> Callable[[CallableT], CallableT]: ...

    _ = object()

    assert isinstance(_, _RuleDecorator)


class Coord:
    __slots__ = ('filename', 'lineno', 'col_span', '__weakref__')

    def __init__(self, filename: str, line: int, col_offset: tuple[int, int] | None = None):
        self.filename = filename
        self.lineno = line
        self.col_span = col_offset

    def __str__(self):
        result = f"{self.filename}:{self.lineno}"
        if self.col_span is not None:
            result += f":{self.col_span}"
        return result


def fix_switch_cases(*args: Any, **kwargs: Any) -> Any: ...


class CParser(Parser):
    debugfile = 'sly_cparser.out'

    # Get the token list from the lexer (required)
    tokens = CLexer.tokens  # | PPLineLexer.tokens | PPPragmaLexer.tokens

    def _select_struct_union_class(self, token: str) -> type[c_ast.Struct | c_ast.Union]:
        """Given a token (either STRUCT or UNION), selects the appropriate AST class."""

        return c_ast.Struct if (token == 'struct') else c_ast.Union

    def _token_coord(self, p: Any, tokenpos: int | None = None) -> Any:
        return Coord("filename", self.line_position(p), self.index_position(p))

    def _build_function_definition(self, *args: Any, **kwargs: Any) -> Any: ...

    def _build_declarations(self, *args: Any, **kwargs: Any) -> Any: ...

    def _add_declaration_specifier(
        self,
        declspec: dict[str, list[Any]] | None,
        newspec: Any,
        kind: str,
        *,
        append: bool = False,
    ) -> Any:
        """Declaration specifiers are represented by a dictionary with the entries:
        * qual: a list of type qualifiers
        * storage: a list of storage type qualifiers
        * type: a list of type specifiers
        * function: a list of function specifiers
        * alignment: a list of alignment specifiers

        This method is given a declaration specifier, and a
        new specifier of a given kind.
        If `append` is True, the new specifier is added to the end of
        the specifiers list, otherwise it's added at the beginning.
        Returns the declaration specifier, with the new
        specifier incorporated.
        """

        spec = declspec or {"qual": [], "storage": [], "type": [], "function": [], "alignment": []}

        if append:
            spec[kind].append(newspec)
        else:
            spec[kind].insert(0, newspec)

        return spec

    def _add_identifier(self, *args: Any, **kwargs: Any) -> Any: ...

    # To understand what's going on here, read sections A.8.5 and
    # A.8.6 of K&R2 very carefully.
    #
    # A C type consists of a basic type declaration, with a list
    # of modifiers. For example:
    #
    # int *c[5];
    #
    # The basic declaration here is 'int c', and the pointer and
    # the array are the modifiers.
    #
    # Basic declarations are represented by TypeDecl (from module c_ast) and the
    # modifiers are FuncDecl, PtrDecl and ArrayDecl.
    #
    # The standard states that whenever a new modifier is parsed, it should be
    # added to the end of the list of modifiers. For example:
    #
    # K&R2 A.8.6.2: Array Declarators
    #
    # In a declaration T D where D has the form
    #   D1 [constant-expression-opt]
    # and the type of the identifier in the declaration T D1 is
    # "type-modifier T", the type of the
    # identifier of D is "type-modifier array of T"
    #
    # This is what this method does. The declarator it receives
    # can be a list of declarators ending with TypeDecl. It
    # tacks the modifier to the end of this list, just before
    # the TypeDecl.
    #
    # Additionally, the modifier may be a list itself. This is
    # useful for pointers, that can come as a chain from the rule
    # p_pointer. In this case, the whole modifier list is spliced
    # into the new location.
    def _type_modify_decl(self, decl: Any, modifier: Any) -> Any:
        """Tacks a type modifier on a declarator, and returns
        the modified declarator.

        Note: the declarator and modifier may be modified
        """
        # ~ print '****'
        # ~ decl.show(offset=3)
        # ~ modifier.show(offset=3)
        # ~ print '****'

        modifier_head = modifier
        modifier_tail = modifier

        # The modifier may be a nested list. Reach its tail.
        while modifier_tail.type:
            modifier_tail = modifier_tail.type

        # If the decl is a basic type, just tack the modifier onto it.
        if isinstance(decl, c_ast.TypeDecl):
            modifier_tail.type = decl
            return modifier
        else:
            # Otherwise, the decl is a list of modifiers. Reach
            # its tail and splice the modifier onto the tail,
            # pointing to the underlying basic type.
            decl_tail = decl

            while not isinstance(decl_tail.type, c_ast.TypeDecl):
                decl_tail = decl_tail.type

            modifier_tail.type = decl_tail.type
            decl_tail.type = modifier_head
            return decl

    def _get_yacc_lookahead_token(self, *args: Any, **kwargs: Any) -> Any: ...

    def _fix_decl_name_type(self, *args: Any, **kwargs: Any) -> Any: ...

    def _is_type_in_scope(self, *args: Any, **kwargs: Any) -> Any: ...

    precedence = (
        ('left', LOR),
        ('left', LAND),
        ('left', OR),
        ('left', XOR),
        ('left', AND),
        ('left', EQ, NE),
        ('left', GT, GE, LT, LE),
        ('left', RSHIFT, LSHIFT),
        ('left', PLUS, MINUS),
        ('left', TIMES, DIVIDE, MOD),
    )

    @_('translation_unit')
    def optional_translation_unit(self, p: Any):
        return c_ast.FileAST(p.translation_unit)

    @_('empty')
    def optional_translation_unit(self, p: Any):
        return c_ast.FileAST([])

    @_('external_declaration')
    def translation_unit(self, p: Any):
        # Note: external_declaration is already a list
        return p.external_declaration

    @_('translation_unit external_declaration')
    def translation_unit(self, p: Any):
        p.translation_unit.extend(p.external_declaration)
        return p.translation_unit

    # Declarations always come as lists (because they can be
    # several in one line), so we wrap the function definition
    # into a list as well, to make the return value of
    # external_declaration homogeneous.
    @_('function_definition')
    def external_declaration(self, p: Any):
        return [p.function_definition]

    @_('declaration')
    def external_declaration(self, p: Any):
        return p.declaration

    @_('pp_directive', 'pppragma_directive')
    def external_declaration(self, p: Any):
        return [p[0]]

    @_('SEMI')
    def external_declaration(self, p: Any) -> list[Any]:
        return []

    @_('static_assert')
    def external_declaration(self, p: Any):
        return p.static_assert

    @_('_STATIC_ASSERT LPAREN constant_expression COMMA unified_string_literal RPAREN')
    def static_assert(self, p: Any):
        return c_ast.StaticAssert(p.constant_expression, p.unified_string_literal, self._token_coord(p, 1))

    @_('_STATIC_ASSERT LPAREN constant_expression RPAREN')
    def static_assert(self, p: Any):
        return c_ast.StaticAssert(p.constant_expression, None, self._token_coord(p, 1))

    @_('PPHASH')
    def pp_directive(self, p: Any):
        raise RuntimeError('Directives not supported yet', self._token_coord(p, 1))

    # This encompasses two types of C99-compatible pragmas:
    # - The #pragma directive:
    #       # pragma character_sequence
    # - The _Pragma unary operator:
    #       _Pragma ( " string_literal " )
    @_('PPPRAGMA')
    def pppragma_directive(self, p: Any):
        return c_ast.Pragma("", self._token_coord(p, 2))

    @_('PPPRAGMA PPPRAGMASTR')
    def pppragma_directive(self, p: Any):
        return c_ast.Pragma(p[1], self._token_coord(p, 2))

    @_('_PRAGMA LPAREN unified_string_literal RPAREN')
    def pppragma_directive(self, p: Any):
        return c_ast.Pragma(p[2], self._token_coord(p, 1))

    @_('pppragma_directive { pppragma_directive }')
    def pppragma_directive_list(self, p: Any):
        return [p.pppragma_directive0, *p.pppragma_directive1]

    # In function definitions, the declarator can be followed by
    # a declaration list, for old "K&R style" function definitios.
    @_('id_declarator declaration_list_opt compound_statement')
    def function_definition(self, p: Any):
        # no declaration specifiers - 'int' becomes the default type
        spec = {
            "qual": [],
            "alignment": [],
            "storage": [],
            "type": [c_ast.IdentifierType(['int'], coord=self._token_coord(p, 1))],
            "function": [],
        }

        return self._build_function_definition(
            spec=spec,
            decl=p.id_declarator,
            param_decls=p.declaration_list_opt,
            body=p.compound_statement,
        )

    @_('declaration_specifiers id_declarator declaration_list_opt compound_statement')
    def function_definition(self, p: Any):
        spec = p.declaration_specifiers

        return self._build_function_definition(
            spec=spec,
            decl=p.id_declarator,
            param_decls=p.declaration_list_opt,
            body=p.compound_statement,
        )

    # Note, according to C18 A.2.2 6.7.10 static_assert-declaration _Static_assert
    # is a declaration, not a statement. We additionally recognise it as a statement
    # to fix parsing of _Static_assert inside the functions.
    #
    @_(
        'labeled_statement',
        'expression_statement',
        'compound_statement',
        'selection_statement',
        'iteration_statement',
        'jump_statement',
        'pppragma_directive',
        'static_assert',
    )
    def statement(self, p: Any):
        return p[0]

    # A pragma is generally considered a decorator rather than an actual
    # statement. Still, for the purposes of analyzing an abstract syntax tree of
    # C code, pragma's should not be ignored and were previously treated as a
    # statement. This presents a problem for constructs that take a statement
    # such as labeled_statements, selection_statements, and
    # iteration_statements, causing a misleading structure in the AST. For
    # example, consider the following C code.
    #
    #   for (int i = 0; i < 3; i++)
    #       #pragma omp critical
    #       sum += 1;
    #
    # This code will compile and execute "sum += 1;" as the body of the for
    # loop. Previous implementations of PyCParser would render the AST for this
    # block of code as follows:
    #
    #   For:
    #     DeclList:
    #       Decl: i, [], [], []
    #         TypeDecl: i, []
    #           IdentifierType: ['int']
    #         Constant: int, 0
    #     BinaryOp: <
    #       ID: i
    #       Constant: int, 3
    #     UnaryOp: p++
    #       ID: i
    #     Pragma: omp critical
    #   Assignment: +=
    #     ID: sum
    #     Constant: int, 1
    #
    # This AST misleadingly takes the Pragma as the body of the loop and the
    # assignment then becomes a sibling of the loop.
    #
    # To solve edge cases like these, the pragmacomp_or_statement rule groups
    # a pragma and its following statement (which would otherwise be orphaned)
    # using a compound block, effectively turning the above code into:
    #
    #   for (int i = 0; i < 3; i++) {
    #       #pragma omp critical
    #       sum += 1;
    #   }
    #
    @_('pppragma_directive_list statement')
    def pragmacomp_or_statement(self, p: Any):
        return c_ast.Compound(
            block_items=p[1] + [p[2]],
            coord=self._token_coord(p, 1),
        )

    @_('statement')
    def pragmacomp_or_statement(self, p: Any):
        return p.statement

    # In C, declarations can come several in a line:
    #   int x, *px, romulo = 5;
    #
    # However, for the AST, we will split them to separate Decl
    # nodes.
    #
    # This rule splits its declarations and always returns a list
    # of Decl nodes, even if it's one element long.
    #
    @_(
        'declaration_specifiers init_declarator_list_opt',
        'declaration_specifiers_no_type id_init_declarator_list_opt',
    )
    def decl_body(self, p: Any):
        spec = p[0]

        # p[2] (init_declarator_list_opt) is either a list or None
        #
        if p.init_declarator_list_opt is None:
            # By the standard, you must have at least one declarator unless
            # declaring a structure tag, a union tag, or the members of an
            # enumeration.
            #
            ty = spec['type']
            s_u_or_e = (c_ast.Struct, c_ast.Union, c_ast.Enum)
            if len(ty) == 1 and isinstance(ty[0], s_u_or_e):
                decls = [
                    c_ast.Decl(
                        name=None,
                        quals=spec['qual'],
                        align=spec['alignment'],
                        storage=spec['storage'],
                        funcspec=spec['function'],
                        type=ty[0],
                        init=None,
                        bitsize=None,
                        coord=ty[0].coord,
                    )
                ]

            # However, this case can also occur on redeclared identifiers in
            # an inner scope.  The trouble is that the redeclared type's name
            # gets grouped into declaration_specifiers; _build_declarations
            # compensates for this.
            #
            else:
                decls = self._build_declarations(
                    spec=spec, decls=[{"decl": None, "init": None}], typedef_namespace=True
                )

        else:
            decls = self._build_declarations(spec=spec, decls=p.init_declarator_list_opt, typedef_namespace=True)

        return decls

    # The declaration has been split to a decl_body sub-rule and
    # SEMI, because having them in a single rule created a problem
    # for defining typedefs.
    #
    # If a typedef line was directly followed by a line using the
    # type defined with the typedef, the type would not be
    # recognized. This is because to reduce the declaration rule,
    # the parser's lookahead asked for the token after SEMI, which
    # was the type from the next line, and the lexer had no chance
    # to see the updated type symbol table.
    #
    # Splitting solves this problem, because after seeing SEMI,
    # the parser reduces decl_body, which actually adds the new
    # type into the table to be seen by the lexer before the next
    # line is reached.
    @_('decl_body SEMI')
    def declaration(self, p: Any):
        return p.decl_body

    # Since each declaration is a list of declarations, this
    # rule will combine all the declarations and return a single
    # list
    #
    @_('declaration { declaration }')
    def declaration_list(self, p: Any):
        return p.declaration0 + p.declaration1

    @_('empty', 'declaration_list')
    def declaration_list_opt(self, p: Any):
        return p[0]

    # To know when declaration-specifiers end and declarators begin,
    # we require declaration-specifiers to have at least one
    # type-specifier, and disallow typedef-names after we've seen any
    # type-specifier. These are both required by the spec.
    #
    @_('type_qualifier declaration_specifiers_no_type_opt')
    def declaration_specifiers_no_type(self, p: Any):
        return self._add_declaration_specifier(
            p.declaration_specifiers_no_type_opt,
            p.type_qualifier,
            'qual',
        )

    @_('storage_class_specifier declaration_specifiers_no_type_opt')
    def declaration_specifiers_no_type(self, p: Any):
        return self._add_declaration_specifier(
            p.declaration_specifiers_no_type_opt,
            p.storage_class_specifier,
            'storage',
        )

    @_('function_specifier declaration_specifiers_no_type_opt')
    def declaration_specifiers_no_type(self, p: Any):
        return self._add_declaration_specifier(
            p.declaration_specifiers_no_type_opt,
            p.function_specifier,
            'function',
        )

    # Without this, `typedef _Atomic(T) U` will parse incorrectly because the
    # _Atomic qualifier will match, instead of the specifier.
    @_('atomic_specifier declaration_specifiers_no_type_opt')
    def declaration_specifiers_no_type(self, p: Any):
        return self._add_declaration_specifier(
            p.declaration_specifiers_no_type_opt,
            p.atomic_specifier,
            'type',
        )

    @_('alignment_specifier declaration_specifiers_no_type_opt')
    def declaration_specifiers_no_type(self, p: Any):
        return self._add_declaration_specifier(
            p.declaration_specifiers_no_type_opt,
            p.alignment_specifier,
            'alignment',
        )

    @_('empty', 'declaration_specifiers_no_type')
    def declaration_specifiers_no_type_opt(self, p: Any):
        return p[0]

    @_('declaration_specifiers type_qualifier')
    def declaration_specifiers(self, p: Any):
        return self._add_declaration_specifier(p.declaration_specifiers, p.type_qualifier, 'qual', append=True)

    @_('declaration_specifiers storage_class_specifier')
    def declaration_specifiers(self, p: Any):
        return self._add_declaration_specifier(
            p.declaration_specifiers,
            p.storage_class_specifier,
            'storage',
            append=True,
        )

    @_('declaration_specifiers function_specifier')
    def declaration_specifiers(self, p: Any):
        return self._add_declaration_specifier(p.declaration_specifiers, p.function_specifier, 'function', append=True)

    @_('declaration_specifiers type_specifier_no_typeid')
    def declaration_specifiers(self, p: Any):
        return self._add_declaration_specifier(
            p.declaration_specifiers,
            p.type_specifier_no_typeid,
            'type',
            append=True,
        )

    @_('type_specifier')
    def declaration_specifiers(self, p: Any):
        return self._add_declaration_specifier(None, p.type_specifier, 'type')

    @_('declaration_specifiers_no_type type_specifier')
    def declaration_specifiers(self, p: Any):
        return self._add_declaration_specifier(p.declaration_specifiers_no_type, p.type_specifier, 'type', append=True)

    @_('declaration_specifiers alignment_specifier')
    def declaration_specifiers(self, p: Any):
        return self._add_declaration_specifier(
            p.declaration_specifiers,
            p.alignment_specifier,
            'alignment',
            append=True,
        )

    @_('AUTO', 'REGISTER', 'STATIC', 'EXTERN', 'TYPEDEF', '_THREAD_LOCAL')
    def storage_class_specifier(self, p: Any):
        return p[0]

    @_('INLINE', '_NORETURN')
    def function_specifier(self, p: Any):
        return p[0]

    @_(
        'VOID',
        '_BOOL',
        'CHAR',
        'SHORT',
        'INT',
        'LONG',
        'FLOAT',
        'DOUBLE',
        '_COMPLEX',
        'SIGNED',
        'UNSIGNED',
        '__INT128',
    )
    def type_specifier_no_typeid(self, p: Any):
        return c_ast.IdentifierType([p[0]], coord=self._token_coord(p, 1))

    @_('typedef_name', 'enum_specifier', 'struct_or_union_specifier', 'type_specifier_no_typeid', 'atomic_specifier')
    def type_specifier(self, p: Any):
        return p[0]

    # See section 6.7.2.4 of the C11 standard.
    @_('_ATOMIC LPAREN type_name RPAREN')
    def atomic_specifier(self, p: Any):
        typ = p.type_name
        typ.quals.append('_Atomic')
        return typ

    @_('CONST', 'RESTRICT', 'VOLATILE', '_ATOMIC')
    def type_qualifier(self, p: Any):
        return p[0]

    @_('init_declarator { COMMA init_declarator }')
    def init_declarator_list(self, p: Any):
        return [p.init_declarator0, *p.init_declarator1]

    @_('empty', 'init_declarator_list')
    def init_declarator_list_opt(self, p: Any):
        return p[0]

    # Returns a {decl=<declarator> : init=<initializer>} dictionary
    # If there's no initializer, uses None
    #
    @_('declarator [ EQUALS initializer ]')
    def init_declarator(self, p: Any):
        return {"decl": p.declarator, "init": p.initializer}

    @_('id_init_declarator')
    def id_init_declarator_list(self, p: Any):
        return [p.id_init_declarator]

    @_('id_init_declarator_list COMMA init_declarator')
    def id_init_declarator_list(self, p: Any):
        return [*p.id_init_declarator_list, p.init_declarator]

    @_('empty', 'id_init_declarator_list')
    def id_init_declarator_list_opt(self, p: Any):
        return p[0]

    @_('id_declarator [ EQUALS initializer ]')
    def id_init_declarator(self, p: Any) -> dict[str, Any]:
        return {"decl": p.id_declarator, "init": p.initializer}

    # Require at least one type specifier in a specifier-qualifier-list
    #
    @_('specifier_qualifier_list type_specifier_no_typeid')
    def specifier_qualifier_list(self, p: Any):
        return self._add_declaration_specifier(
            p.specifier_qualifier_list,
            p.type_specifier_no_typeid,
            'type',
            append=True,
        )

    @_('specifier_qualifier_list type_qualifier')
    def specifier_qualifier_list(self, p: Any):
        return self._add_declaration_specifier(p.specifier_qualifier_list, p.type_qualifier, 'qual', append=True)

    @_('type_specifier')
    def specifier_qualifier_list(self, p: Any):
        return self._add_declaration_specifier(None, p.type_specifier, 'type')

    @_('type_qualifier_list type_specifier')
    def specifier_qualifier_list(self, p: Any):
        return {
            "qual": p.type_qualifier_list,
            "alignment": [],
            "storage": [],
            "type": [p.type_specifier],
            "function": [],
        }

    @_('alignment_specifier')
    def specifier_qualifier_list(self, p: Any):
        return {"qual": [], "alignment": [p.alignment_specifier], "storage": [], "type": [], "function": []}

    @_('specifier_qualifier_list alignment_specifier')
    def specifier_qualifier_list(self, p: Any):
        return self._add_declaration_specifier(p.specifier_qualifier_list, p.alignment_specifier, 'alignment')

    # TYPEID is allowed here (and in other struct/enum related tag names), because
    # struct/enum tags reside in their own namespace and can be named the same as types
    #
    @_('struct_or_union ID', 'struct_or_union TYPEID')
    def struct_or_union_specifier(self, p: Any):
        klass = self._select_struct_union_class(p.struct_or_union)
        # None means no list of members
        return klass(name=p[1], decls=None, coord=self._token_coord(p, 2))

    @_('struct_or_union brace_open [ struct_declaration_list ] brace_close')
    def struct_or_union_specifier(self, p: Any):
        klass = self._select_struct_union_class(p.struct_or_union)
        # Empty sequence means an empty list of members
        return klass(name=None, decls=p.struct_declaration_list or [], coord=self._token_coord(p, 2))

    @_(
        'struct_or_union ID brace_open [ struct_declaration_list ] brace_close',
        'struct_or_union TYPEID brace_open [ struct_declaration_list ] brace_close',
    )
    def struct_or_union_specifier(self, p: Any):
        klass = self._select_struct_union_class(p.struct_or_union)
        # Empty sequence means an empty list of members
        return klass(name=p[1], decls=p.struct_declaration_list or [], coord=self._token_coord(p, 2))

    @_('STRUCT', 'UNION')
    def struct_or_union(self, p: Any):
        return p[0]

    # Combine all declarations into a single list
    #
    @_('struct_declaration { struct_declaration }')
    def struct_declaration_list(self, p: Any):
        return p.struct_declaration0 + (p.struct_declaration1 or [])

    @_('specifier_qualifier_list struct_declarator_list_opt SEMI')
    def struct_declaration(self, p: Any):
        spec = p.specifier_qualifier_list
        assert 'typedef' not in spec['storage']

        if p.struct_declarator_list_opt is not None:
            decls = self._build_declarations(spec=spec, decls=p.struct_declarator_list_opt)

        elif len(spec['type']) == 1:
            # Anonymous struct/union, gcc extension, C1x feature.
            # Although the standard only allows structs/unions here, I see no
            # reason to disallow other types since some compilers have typedefs
            # here, and pycparser isn't about rejecting all invalid code.
            #
            node = spec['type'][0]
            if isinstance(node, c_ast.Node):
                decl_type = node
            else:
                decl_type = c_ast.IdentifierType(node)

            decls = self._build_declarations(spec=spec, decls=[{"decl": decl_type}])

        else:
            # Structure/union members can have the same names as typedefs.
            # The trouble is that the member's name gets grouped into
            # specifier_qualifier_list; _build_declarations compensates.
            #
            decls = self._build_declarations(spec=spec, decls=[{"decl": None, "init": None}])

        return decls

    @_('SEMI')
    def struct_declaration(self, p: Any):
        return None

    @_('pppragma_directive')
    def struct_declaration(self, p: Any):
        return [p.pppragma_directive]

    @_('struct_declarator { COMMA struct_declarator }')
    def struct_declarator_list(self, p: Any):
        return [p.struct_declarator0, *p.struct_declarator1]

    @_('empty', 'struct_declarator_list')
    def struct_declarator_list_opt(self, p: Any):
        return p[0]

    # struct_declarator passes up a dict with the keys: decl (for
    # the underlying declarator) and bitsize (for the bitsize)
    #
    @_('declarator')
    def struct_declarator(self, p: Any):
        return {'decl': p.declarator, 'bitsize': None}

    @_('declarator COLON constant_expression')
    def struct_declarator(self, p: Any):
        return {'decl': p.declarator, 'bitsize': p.constant_expression}

    @_('COLON constant_expression')
    def struct_declarator(self, p: Any):
        return {'decl': c_ast.TypeDecl(None, None, None, None), 'bitsize': p.constant_expression}

    @_('ENUM ID', 'ENUM TYPEID')
    def enum_specifier(self, p: Any):
        return c_ast.Enum(p[1], None, self._token_coord(p, 1))

    @_('ENUM brace_open enumerator_list brace_close')
    def enum_specifier(self, p: Any):
        return c_ast.Enum(None, p.enumerator_list, self._token_coord(p, 1))

    @_(
        'ENUM ID brace_open enumerator_list brace_close',
        'ENUM TYPEID brace_open enumerator_list brace_close',
    )
    def enum_specifier(self, p: Any):
        return c_ast.Enum(p[1], p.enumerator_list, self._token_coord(p, 1))

    @_('enumerator')
    def enumerator_list(self, p: Any):
        return c_ast.EnumeratorList([p.enumerator], p.enumerator.coord)

    @_('enumerator_list COMMA')
    def enumerator_list(self, p: Any):
        return p.enumerator_list

    @_('enumerator_list COMMA enumerator')
    def enumerator_list(self, p: Any):
        p.enumerator_list.enumerators.append(p.enumerator)
        return p.enumerator_list

    @_(
        '_ALIGNAS LPAREN type_name RPAREN',
        '_ALIGNAS LPAREN constant_expression RPAREN',
    )
    def alignment_specifier(self, p: Any):
        return c_ast.Alignas(p[2], self._token_coord(p, 1))

    @_('ID [ EQUALS constant_expression ]')
    def enumerator(self, p: Any):
        enumerator = c_ast.Enumerator(p.ID, p.constant_expression, self._token_coord(p, 1))
        self._add_identifier(enumerator.name, enumerator.coord)
        return enumerator

    @_('id_declarator', 'typeid_declarator')
    def declarator(self, p: Any):
        return p[0]

    @_('direct_id_declarator')
    def id_declarator(self, p: Any):
        return p.direct_id_declarator

    @_('pointer direct_id_declarator')
    def id_declarator(self, p: Any):
        return self._type_modify_decl(p.direct_id_declarator, p.pointer)

    @_('direct_typeid_declarator')
    def typeid_declarator(self, p: Any):
        return p.direct_typeid_declarator

    @_('pointer direct_typeid_declarator')
    def typeid_declarator(self, p: Any):
        return self._type_modify_decl(p.direct_typeid_declarator, p.pointer)

    @_('direct_typeid_noparen_declarator')
    def typeid_noparen_declarator(self, p: Any):
        return p.direct_typeid_noparen_declarator

    @_('pointer direct_typeid_noparen_declarator')
    def typeid_noparen_declarator(self, p: Any):
        return self._type_modify_decl(p.direct_typeid_noparen_declarator, p.pointer)

    # ===============================================================================================

    @_('ID')
    def direct_id_declarator(self, p: Any):
        return c_ast.TypeDecl(declname=p[0], type=None, quals=None, align=None, coord=self._token_coord(p, 1))

    @_('LPAREN id_declarator RPAREN')
    def direct_id_declarator(self, p: Any):
        return p[1]

    @_('direct_id_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET')
    def direct_id_declarator(self, p: Any):
        quals: list[Any] = (p.type_qualifier_list_opt if len(p) > 4 else []) or []
        # Accept dimension qualifiers
        # Per C99 6.7.5.3 p7
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p[3] if len(p) > 4 else p[2],
            dim_quals=quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @_(
        'direct_id_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET',
        'direct_id_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET',
    )
    def direct_id_declarator(self, p: Any):
        listed_quals: list[list[Any]] = [item if isinstance(item, list) else [item] for item in [p[3:5]]]
        dim_quals = [qual for sublist in listed_quals for qual in sublist if qual is not None]
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p.assignment_expression,
            dim_quals=dim_quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    # Special for VLAs
    #
    @_('direct_id_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET')
    def direct_id_declarator(self, p: Any):
        arr = c_ast.ArrayDecl(
            type=None,
            dim=c_ast.ID(p.TIMES, self._token_coord(p, 4)),
            dim_quals=p.type_qualifier_list_opt if p.type_qualifier_list_opt is not None else [],
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @_(
        'direct_id_declarator LPAREN parameter_type_list RPAREN',
        'direct_id_declarator LPAREN identifier_list_opt RPAREN',
    )
    def direct_id_declarator(self, p: Any):
        func = c_ast.FuncDecl(args=p[2], type=None, coord=p[0].coord)

        # To see why _get_yacc_lookahead_token is needed, consider:
        #   typedef char TT;
        #   void foo(int TT) { TT = 10; }
        # Outside the function, TT is a typedef, but inside (starting and
        # ending with the braces) it's a parameter.  The trouble begins with
        # yacc's lookahead token.  We don't know if we're declaring or
        # defining a function until we see LBRACE, but if we wait for yacc to
        # trigger a rule on that token, then TT will have already been read
        # and incorrectly interpreted as TYPEID.  We need to add the
        # parameters to the scope the moment the lexer sees LBRACE.
        #
        if self._get_yacc_lookahead_token().type == "LBRACE" and func.args is not None:
            for param in func.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    break
                self._add_identifier(param.name, param.coord)

        return self._type_modify_decl(decl=p[0], modifier=func)

    # ===============================================================================================

    @_('TYPEID')
    def direct_typeid_declarator(self, p: Any):
        return c_ast.TypeDecl(declname=p[0], type=None, quals=None, align=None, coord=self._token_coord(p, 1))

    @_('LPAREN typeid_declarator RPAREN')
    def direct_typeid_declarator(self, p: Any):
        return p[1]

    @_('direct_typeid_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET')
    def direct_typeid_declarator(self, p: Any):
        quals: list[Any] = (p.type_qualifier_list_opt if len(p) > 4 else []) or []
        # Accept dimension qualifiers
        # Per C99 6.7.5.3 p7
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p[3] if len(p) > 4 else p[2],
            dim_quals=quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @_(
        'direct_typeid_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET',
        'direct_typeid_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET',
    )
    def direct_typeid_declarator(self, p: Any):
        listed_quals: list[list[Any]] = [item if isinstance(item, list) else [item] for item in [p[3:5]]]
        dim_quals = [qual for sublist in listed_quals for qual in sublist if qual is not None]
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p.assignment_expression,
            dim_quals=dim_quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    # Special for VLAs
    #
    @_('direct_typeid_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET')
    def direct_typeid_declarator(self, p: Any):
        arr = c_ast.ArrayDecl(
            type=None,
            dim=c_ast.ID(p.TIMES, self._token_coord(p, 4)),
            dim_quals=p.type_qualifier_list_opt if p.type_qualifier_list_opt is not None else [],
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @_(
        'direct_typeid_declarator LPAREN parameter_type_list RPAREN',
        'direct_typeid_declarator LPAREN identifier_list_opt RPAREN',
    )
    def direct_typeid_declarator(self, p: Any):
        func = c_ast.FuncDecl(args=p[2], type=None, coord=p[0].coord)

        # To see why _get_yacc_lookahead_token is needed, consider:
        #   typedef char TT;
        #   void foo(int TT) { TT = 10; }
        # Outside the function, TT is a typedef, but inside (starting and
        # ending with the braces) it's a parameter.  The trouble begins with
        # yacc's lookahead token.  We don't know if we're declaring or
        # defining a function until we see LBRACE, but if we wait for yacc to
        # trigger a rule on that token, then TT will have already been read
        # and incorrectly interpreted as TYPEID.  We need to add the
        # parameters to the scope the moment the lexer sees LBRACE.
        #
        if self._get_yacc_lookahead_token().type == "LBRACE" and func.args is not None:
            for param in func.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    break
                self._add_identifier(param.name, param.coord)

        return self._type_modify_decl(decl=p[0], modifier=func)

    # ===============================================================================================

    @_('TYPEID')
    def direct_typeid_noparen_declarator(self, p: Any):
        return c_ast.TypeDecl(declname=p[0], type=None, quals=None, align=None, coord=self._token_coord(p, 1))

    @_('direct_typeid_noparen_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET')
    def direct_typeid_noparen_declarator(self, p: Any):
        quals: list[Any] = (p.type_qualifier_list_opt if len(p) > 4 else []) or []
        # Accept dimension qualifiers
        # Per C99 6.7.5.3 p7
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p[3] if len(p) > 4 else p[2],
            dim_quals=quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @_(
        'direct_typeid_noparen_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET',
        'direct_typeid_noparen_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET',
    )
    def direct_typeid_noparen_declarator(self, p: Any):
        listed_quals: list[list[Any]] = [item if isinstance(item, list) else [item] for item in [p[3:5]]]
        dim_quals = [qual for sublist in listed_quals for qual in sublist if qual is not None]
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p.assignment_expression,
            dim_quals=dim_quals,
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    # Special for VLAs
    #
    @_('direct_typeid_noparen_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET')
    def direct_typeid_noparen_declarator(self, p: Any):
        arr = c_ast.ArrayDecl(
            type=None,
            dim=c_ast.ID(p.TIMES, self._token_coord(p, 4)),
            dim_quals=p.type_qualifier_list_opt if p.type_qualifier_list_opt is not None else [],
            coord=p[0].coord,
        )

        return self._type_modify_decl(decl=p[0], modifier=arr)

    @_(
        'direct_typeid_noparen_declarator LPAREN parameter_type_list RPAREN',
        'direct_typeid_noparen_declarator LPAREN identifier_list_opt RPAREN',
    )
    def direct_typeid_noparen_declarator(self, p: Any):
        func = c_ast.FuncDecl(args=p[2], type=None, coord=p[0].coord)

        # To see why _get_yacc_lookahead_token is needed, consider:
        #   typedef char TT;
        #   void foo(int TT) { TT = 10; }
        # Outside the function, TT is a typedef, but inside (starting and
        # ending with the braces) it's a parameter.  The trouble begins with
        # yacc's lookahead token.  We don't know if we're declaring or
        # defining a function until we see LBRACE, but if we wait for yacc to
        # trigger a rule on that token, then TT will have already been read
        # and incorrectly interpreted as TYPEID.  We need to add the
        # parameters to the scope the moment the lexer sees LBRACE.
        #
        if self._get_yacc_lookahead_token().type == "LBRACE" and func.args is not None:
            for param in func.args.params:
                if isinstance(param, c_ast.EllipsisParam):
                    break
                self._add_identifier(param.name, param.coord)

        return self._type_modify_decl(decl=p[0], modifier=func)

    # ===============================================================================================

    @_(
        'TIMES type_qualifier_list_opt',
        'TIMES type_qualifier_list_opt pointer',
    )
    def pointer(self, p: Any):
        coord = self._token_coord(p, 1)
        # Pointer decls nest from inside out. This is important when different
        # levels have different qualifiers. For example:
        #
        #  char * const * p;
        #
        # Means "pointer to const pointer to char"
        #
        # While:
        #
        #  char ** const p;
        #
        # Means "const pointer to pointer to char"
        #
        # So when we construct PtrDecl nestings, the leftmost pointer goes in
        # as the most nested type.
        nested_type = c_ast.PtrDecl(quals=p[1] or [], type=None, coord=coord)
        if len(p) > 2:
            tail_type = p[2]
            while tail_type.type is not None:
                tail_type = tail_type.type
            tail_type.type = nested_type
            return p[2]
        else:
            return nested_type

    @_('type_qualifier { type_qualifier }')
    def type_qualifier_list(self, p: Any):
        return [*p.type_qualifier0, p.type_qualifier1]

    @_('empty', 'type_qualifier_list')
    def type_qualifier_list_opt(self, p: Any):
        return p[0]

    @_('parameter_list', 'parameter_list COMMA ELLIPSIS')
    def parameter_type_list(self, p: Any):
        if len(p) > 1:
            p.parameter_list.params.append(c_ast.EllipsisParam(self._token_coord(p, 3)))
        return p.parameter_list

    @_('empty', 'parameter_type_list')
    def parameter_type_list_opt(self, p: Any):
        return p[0]

    @_('parameter_declaration { COMMA parameter_declaration }')
    def parameter_list(self, p: Any):
        # single parameter
        return c_ast.ParamList([p.parameter_declaration0, *p.parameter_declaration1], p.parameter_declaration0.coord)

    # From ISO/IEC 9899:TC2, 6.7.5.3.11:
    # "If, in a parameter declaration, an identifier can be treated either
    #  as a typedef name or as a parameter name, it shall be taken as a
    #  typedef name."
    #
    # Inside a parameter declaration, once we've reduced declaration specifiers,
    # if we shift in an LPAREN and see a TYPEID, it could be either an abstract
    # declarator or a declarator nested inside parens. This rule tells us to
    # always treat it as an abstract declarator. Therefore, we only accept
    # `id_declarator`s and `typeid_noparen_declarator`s.
    @_(
        'declaration_specifiers id_declarator',
        'declaration_specifiers typeid_noparen_declarator',
    )
    def parameter_declaration(self, p: Any):
        spec = p.declaration_specifiers
        if not spec['type']:
            spec['type'] = [c_ast.IdentifierType(['int'], coord=self._token_coord(p, 1))]
        return self._build_declarations(spec=spec, decls=[{"decl": p[1]}])[0]

    @_('declaration_specifiers abstract_declarator_opt')
    def parameter_declaration(self, p: Any):
        spec = p.declaration_specifiers
        if not spec['type']:
            spec['type'] = [c_ast.IdentifierType(['int'], coord=self._token_coord(p, 1))]

        # Parameters can have the same names as typedefs.  The trouble is that
        # the parameter's name gets grouped into declaration_specifiers, making
        # it look like an old-style declaration; compensate.
        #
        if (
            len(spec['type']) > 1
            and len(spec['type'][-1].names) == 1
            and self._is_type_in_scope(spec['type'][-1].names[0])
        ):
            decl = self._build_declarations(spec=spec, decls=[{"decl": p.abstract_declarator_opt, "init": None}])[0]

        # This truly is an old-style parameter declaration
        #
        else:
            decl = c_ast.Typename(
                name='',
                quals=spec['qual'],
                align=None,
                type=p.abstract_declarator_opt or c_ast.TypeDecl(None, None, None, None),
                coord=self._token_coord(p, 2),
            )
            typename = spec['type']
            decl = self._fix_decl_name_type(decl, typename)

        return decl

    @_('identifier { COMMA identifier }')
    def identifier_list(self, p: Any):
        return c_ast.ParamList([p.identifier0, *p.identifier1], p.identifier0.coord)

    @_('empty', 'identifier_list')
    def identifier_list_opt(self, p: Any):
        return p[0]

    @_('assignment_expression')
    def initializer(self, p: Any):
        return p.assignment_expression

    @_(
        'brace_open initializer_list_opt brace_close',
        'brace_open initializer_list COMMA brace_close',
    )
    def initializer(self, p: Any):
        if p[1] is None:
            return c_ast.InitList([], self._token_coord(p, 1))
        else:
            return p[1]

    @_('designation_opt initializer { COMMA designation_opt initializer }')
    def initializer_list(self, p: Any):
        init = p.designation_opt if p.initializer is None else c_ast.NamedInitializer(p.designation_opt, p.initializer)
        init_exprs = [init]
        for _, extra_des, extra_init in p[2]:
            extra_named_init = extra_init if extra_des is None else c_ast.NamedInitializer(extra_des, extra_init)
            init_exprs.append(extra_named_init)
        return c_ast.InitList(init_exprs, p.initializer.coord)

    @_('empty', 'initializer_list')
    def initializer_list_opt(self, p: Any):
        return p[0]

    @_('designator_list EQUALS')
    def designation(self, p: Any):
        return p.designator_list

    @_('empty', 'designation')
    def designation_opt(self, p: Any):
        return p[0]

    # Designators are represented as a list of nodes, in the order in which
    # they're written in the code.
    #
    @_('designator { designator }')
    def designator_list(self, p: Any):
        return [p.designator0, *p.designator1]

    @_('LBRACKET constant_expression RBRACKET', 'PERIOD identifier')
    def designator(self, p: Any):
        return p[1]

    @_('specifier_qualifier_list abstract_declarator_opt')
    def type_name(self, p: Any):
        typename = c_ast.Typename(
            name='',
            quals=p.specifier_qualifier_list['qual'][:],
            align=None,
            type=p.abstract_declarator_opt or c_ast.TypeDecl(None, None, None, None),
            coord=self._token_coord(p, 2),
        )

        return self._fix_decl_name_type(typename, p[1]['type'])

    @_('pointer')
    def abstract_declarator(self, p: Any):
        dummytype = c_ast.TypeDecl(None, None, None, None)
        return self._type_modify_decl(decl=dummytype, modifier=p.pointer)

    @_('pointer direct_abstract_declarator')
    def abstract_declarator(self, p: Any):
        return self._type_modify_decl(p.direct_abstract_declarator, p.pointer)

    @_('direct_abstract_declarator')
    def abstract_declarator(self, p: Any):
        return p.direct_abstract_declarator

    @_('empty', 'abstract_declarator')
    def abstract_declarator_opt(self, p: Any):
        return p[0]

    # Creating and using direct_abstract_declarator_opt here
    # instead of listing both direct_abstract_declarator and the
    # lack of it in the beginning of _1 and _2 caused two
    # shift/reduce errors.
    #
    @_('LPAREN abstract_declarator RPAREN')
    def direct_abstract_declarator(self, p: Any):
        return p.abstract_declarator

    @_('direct_abstract_declarator LBRACKET assignment_expression_opt RBRACKET')
    def direct_abstract_declarator(self, p: Any):
        arr = c_ast.ArrayDecl(
            type=None, dim=p.assignment_expression_opt, dim_quals=[], coord=p.direct_abstract_declarator.coord
        )

        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=arr)

    @_('LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET')
    def direct_abstract_declarator(self, p: Any):
        quals: list[Any] = (p[1] if len(p) > 3 else []) or []
        return c_ast.ArrayDecl(
            type=c_ast.TypeDecl(None, None, None, None),
            dim=p[2] if len(p) > 3 else p[1],
            dim_quals=quals,
            coord=self._token_coord(p, 1),
        )

    @_('direct_abstract_declarator LBRACKET TIMES RBRACKET')
    def direct_abstract_declarator(self, p: Any):
        arr = c_ast.ArrayDecl(
            type=None,
            dim=c_ast.ID(p.TIMES, self._token_coord(p, 3)),
            dim_quals=[],
            coord=p.direct_abstract_declarator.coord,
        )

        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=arr)

    @_('LBRACKET TIMES RBRACKET')
    def direct_abstract_declarator(self, p: Any):
        return c_ast.ArrayDecl(
            type=c_ast.TypeDecl(None, None, None, None),
            dim=c_ast.ID(p.RBRACKET, self._token_coord(p, 3)),
            dim_quals=[],
            coord=self._token_coord(p, 1),
        )

    @_('direct_abstract_declarator LPAREN parameter_type_list_opt RPAREN')
    def direct_abstract_declarator(self, p: Any):
        func = c_ast.FuncDecl(args=p.parameter_type_list_opt, type=None, coord=p.direct_abstract_declarator.coord)
        return self._type_modify_decl(decl=p.direct_abstract_declarator, modifier=func)

    @_('LPAREN parameter_type_list_opt RPAREN')
    def direct_abstract_declarator(self, p: Any):
        return c_ast.FuncDecl(
            args=p.parameter_type_list_opt,
            type=c_ast.TypeDecl(None, None, None, None),
            coord=self._token_coord(p, 1),
        )

    # declaration is a list, statement isn't. To make it consistent, block_item
    # will always be a list
    #
    @_('declaration', 'statement')
    def block_item(self, p: Any) -> list[Any]:
        item = p[0]
        return item if isinstance(item, list) else [item]

    # Since we made block_item a list, this just combines lists
    #
    @_('block_item', 'block_item_list block_item')
    def block_item_list(self, p: Any):
        # Empty block items (plain ';') produce [None], so ignore them
        return p[0] if (len(p) == 1 or p[1] == [None]) else (p[0] + p[1])

    @_('empty', 'block_item_list')
    def block_item_list_opt(self, p: Any):
        return p[0]

    @_('brace_open block_item_list_opt brace_close')
    def compound_statement(self, p: Any):
        return c_ast.Compound(block_items=p.block_item_list_opt, coord=self._token_coord(p, 1))

    @_('ID COLON pragmacomp_or_statement')
    def labeled_statement(self, p: Any):
        return c_ast.Label(p.ID, p.pragmacomp_or_statement, self._token_coord(p, 1))

    @_('CASE constant_expression COLON pragmacomp_or_statement')
    def labeled_statement(self, p: Any):
        return c_ast.Case(p.constant_expression, [p.pragmacomp_or_statement], self._token_coord(p, 1))

    @_('DEFAULT COLON pragmacomp_or_statement')
    def labeled_statement(self, p: Any):
        return c_ast.Default([p.pragmacomp_or_statement], self._token_coord(p, 1))

    @_('IF LPAREN expression RPAREN pragmacomp_or_statement')
    def selection_statement(self, p: Any):
        return c_ast.If(p[2], p[4], None, self._token_coord(p, 1))

    @_('IF LPAREN expression RPAREN statement ELSE pragmacomp_or_statement')
    def selection_statement(self, p: Any):
        return c_ast.If(p[2], p[4], p[6], self._token_coord(p, 1))

    @_('SWITCH LPAREN expression RPAREN pragmacomp_or_statement')
    def selection_statement(self, p: Any):
        return fix_switch_cases(c_ast.Switch(p.expression, p.pragmacomp_or_statement, self._token_coord(p, 1)))

    @_('WHILE LPAREN expression RPAREN pragmacomp_or_statement')
    def iteration_statement(self, p: Any):
        return c_ast.While(p.expression, p.pragmacomp_or_statement, self._token_coord(p, 1))

    @_('DO pragmacomp_or_statement WHILE LPAREN expression RPAREN SEMI')
    def iteration_statement(self, p: Any):
        return c_ast.DoWhile(p.expression, p.pragmacomp_or_statement, self._token_coord(p, 1))

    @_('FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement')
    def iteration_statement(self, p: Any):
        return c_ast.For(p[2], p[4], p[6], p[8], self._token_coord(p, 1))

    @_('FOR LPAREN declaration expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement')
    def iteration_statement(self, p: Any):
        return c_ast.For(c_ast.DeclList(p[2], self._token_coord(p, 1)), p[3], p[5], p[7], self._token_coord(p, 1))

    @_('GOTO ID SEMI')
    def jump_statement(self, p: Any):
        return c_ast.Goto(p.ID, self._token_coord(p, 1))

    @_('BREAK SEMI')
    def jump_statement(self, p: Any):
        return c_ast.Break(self._token_coord(p, 1))

    @_('CONTINUE SEMI')
    def jump_statement(self, p: Any):
        return c_ast.Continue(self._token_coord(p, 1))

    @_('RETURN expression SEMI')
    def jump_statement(self, p: Any):
        return c_ast.Return(p.expression, self._token_coord(p, 1))

    @_('RETURN SEMI')
    def jump_statement(self, p: Any):
        return c_ast.Return(None, self._token_coord(p, 1))

    @_('expression_opt SEMI')
    def expression_statement(self, p: Any):
        if p.expression_opt is None:
            return c_ast.EmptyStatement(self._token_coord(p, 2))
        else:
            return p.expression_opt

    @_('assignment_expression')
    def expression(self, p: Any):
        return p.assignment_expression

    @_('expression COMMA assignment_expression')
    def expression(self, p: Any):
        if not isinstance(p.expression, c_ast.ExprList):
            p.expression = c_ast.ExprList([p.expression], p.expression.coord)

        p.expression.exprs.append(p.assignment_expression)
        return p.expression

    @_('empty', 'expression')
    def expression_opt(self, p: Any):
        return p[0]

    @_('TYPEID')
    def typedef_name(self, p: Any):
        return c_ast.IdentifierType([p.TYPEID], coord=self._token_coord(p, 1))

    @_('LPAREN compound_statement RPAREN')
    def assignment_expression(self, p: Any):
        # TODO: Verify that the original name "parenthesized_compound_expression", isn't meaningful.
        return p.compound_statement

    @_('conditional_expression')
    def assignment_expression(self, p: Any):
        return p.conditional_expression

    @_('unary_expression assignment_operator assignment_expression')
    def assignment_expression(self, p: Any):
        return c_ast.Assignment(
            p.assignment_operator,
            p.unary_expression,
            p.assignment_expression,
            p.assignment_operator.coord,
        )

    @_('empty', 'assignment_expression')
    def assignment_expression_opt(self, p: Any):
        return p[0]

    # K&R2 defines these as many separate rules, to encode
    # precedence and associativity. Why work hard ? I'll just use
    # the built in precedence/associativity specification feature
    # of PLY. (see precedence declaration above)
    #
    @_(
        'EQUALS',
        'XOREQUAL',
        'TIMESEQUAL',
        'DIVEQUAL',
        'MODEQUAL',
        'PLUSEQUAL',
        'MINUSEQUAL',
        'LSHIFTEQUAL',
        'RSHIFTEQUAL',
        'ANDEQUAL',
        'OREQUAL',
    )
    def assignment_operator(self, p: Any):
        return p[0]

    @_('conditional_expression')
    def constant_expression(self, p: Any):
        return p.conditional_expression

    @_('binary_expression')
    def conditional_expression(self, p: Any):
        return p.binary_expression

    @_('binary_expression CONDOP expression COLON conditional_expression')
    def conditional_expression(self, p: Any):
        return c_ast.TernaryOp(p.binary_expression, p.expression, p.conditional_expression, p.binary_expression.coord)

    @_('cast_expression')
    def binary_expression(self, p: Any):
        return p.cast_expression

    @_(
        'binary_expression TIMES binary_expression',
        'binary_expression DIVIDE binary_expression',
        'binary_expression MOD binary_expression',
        'binary_expression PLUS binary_expression',
        'binary_expression MINUS binary_expression',
        'binary_expression RSHIFT binary_expression',
        'binary_expression LSHIFT binary_expression',
        'binary_expression LT binary_expression',
        'binary_expression LE binary_expression',
        'binary_expression GE binary_expression',
        'binary_expression GT binary_expression',
        'binary_expression EQ binary_expression',
        'binary_expression NE binary_expression',
        'binary_expression AND binary_expression',
        'binary_expression OR binary_expression',
        'binary_expression XOR binary_expression',
        'binary_expression LAND binary_expression',
        'binary_expression LOR binary_expression',
    )
    def binary_expression(self, p: Any):
        return c_ast.BinaryOp(p[1], p[0], p[2], p[0].coord)

    @_('unary_expression')
    def cast_expression(self, p: Any):
        return p.unary_expression

    @_('LPAREN type_name RPAREN cast_expression')
    def cast_expression(self, p: Any):
        return c_ast.Cast(p.type_name, p.cast_expression, self._token_coord(p, 1))

    @_('postfix_expression')
    def unary_expression(self, p: Any):
        return p.postfix_expression

    @_('PLUSPLUS unary_expression', 'MINUSMINUS unary_expression', 'unary_operator cast_expression')
    def unary_expression(self, p: Any):
        return c_ast.UnaryOp(p[1], p[2], p[2].coord)

    @_('SIZEOF unary_expression')
    def unary_expression(self, p: Any):
        return c_ast.UnaryOp(p[0], p[1], self._token_coord(p, 1))

    @_('SIZEOF LPAREN type_name RPAREN', '_ALIGNOF LPAREN type_name RPAREN')
    def unary_expression(self, p: Any):
        return c_ast.UnaryOp(p[0], p[2], self._token_coord(p, 1))

    @_('AND', 'TIMES', 'PLUS', 'MINUS', 'NOT', 'LNOT')
    def unary_operator(self, p: Any):
        return p[0]

    @_('primary_expression')
    def postfix_expression(self, p: Any):
        return p.primary_expression

    @_('postfix_expression LBRACKET expression RBRACKET')
    def postfix_expression(self, p: Any):
        return c_ast.ArrayRef(p.postfix_expression, p.expression, p.postfix_expression.coord)

    @_('postfix_expression LPAREN [ argument_expression_list ] RPAREN')
    def postfix_expression(self, p: Any):
        return c_ast.FuncCall(p.postfix_expression, p.argument_expression_list, p.postfix_expression.coord)

    @_(
        'postfix_expression PERIOD ID',
        'postfix_expression PERIOD TYPEID',
        'postfix_expression ARROW ID',
        'postfix_expression ARROW TYPEID',
    )
    def postfix_expression(self, p: Any):
        field = c_ast.ID(p[2], self._token_coord(p, 3))
        return c_ast.StructRef(p.postfix_expression, p[1], field, p.postfix_expression.coord)

    @_('postfix_expression PLUSPLUS', 'postfix_expression MINUSMINUS')
    def postfix_expression(self, p: Any):
        return c_ast.UnaryOp('p' + p[1], p.postfix_expression, p[1].coord)

    @_('LPAREN type_name RPAREN brace_open initializer_list [ COMMA ] brace_close')
    def postfix_expression(self, p: Any):
        return c_ast.CompoundLiteral(p.type_name, p.initializer_list)

    @_('identifier', 'constant', 'unified_string_literal', 'unified_wstring_literal')
    def primary_expression(self, p: Any):
        return p[0]

    @_('LPAREN expression RPAREN')
    def primary_expression(self, p: Any):
        return p.expression

    @_('OFFSETOF LPAREN type_name COMMA offsetof_member_designator RPAREN')
    def primary_expression(self, p: Any):
        coord = self._token_coord(p, 1)
        return c_ast.FuncCall(
            c_ast.ID(p.OFFSETOF, coord),
            c_ast.ExprList([p.type_name, p.offsetof_member_designator], coord),
            coord,
        )

    @_('identifier')
    def offsetof_member_designator(self, p: Any):
        return p.identifier

    @_('offsetof_member_designator PERIOD identifier')
    def offsetof_member_designator(self, p: Any):
        return c_ast.StructRef(p.offsetof_member_designator, p.PERIOD, p.identifer, p.offsetof_member_designator.coord)

    @_('offsetof_member_designator LBRACKET expression RBRACKET')
    def offsetof_member_designator(self, p: Any):
        return c_ast.ArrayRef(p.offsetof_member_designator, p.expression, p.offsetof_member_designator.coord)

    @_('assignment_expression { COMMA assignment_expression }')
    def argument_expression_list(self, p: Any):
        return c_ast.ExprList([p.assignment_expression0, *p.assignment_expression1], p.assignment_expression.coord)

    @_('ID')
    def identifier(self, p: Any):
        return c_ast.ID(p.ID, self._token_coord(p, 1))

    @_('INT_CONST_DEC', 'INT_CONST_OCT', 'INT_CONST_HEX', 'INT_CONST_BIN', 'INT_CONST_CHAR')
    def constant(self, p: Any):
        uCount = 0
        lCount = 0
        for x in p[0][-3:]:
            if x in {'l', 'L'}:
                lCount += 1
            elif x in {'u', 'U'}:
                uCount += 1

        if uCount > 1:
            raise ValueError('Constant cannot have more than one u/U suffix.')
        if lCount > 2:
            raise ValueError('Constant cannot have more than two l/L suffix.')
        prefix = 'unsigned ' * uCount + 'long ' * lCount
        return c_ast.Constant(prefix + 'int', p[0], self._token_coord(p, 1))

    @_('FLOAT_CONST', 'HEX_FLOAT_CONST')
    def constant(self, p: Any):
        if 'x' in p[0].lower():
            t = 'float'
        else:
            if p[0][-1] in {'f', 'F'}:
                t = 'float'
            elif p[0][-1] in {'l', 'L'}:
                t = 'long double'
            else:
                t = 'double'

        return c_ast.Constant(t, p[0], self._token_coord(p, 1))

    @_('CHAR_CONST', 'WCHAR_CONST', 'U8CHAR_CONST', 'U16CHAR_CONST', 'U32CHAR_CONST')
    def constant(self, p: Any):
        return c_ast.Constant('char', p[0], self._token_coord(p, 1))

    # The "unified" string and wstring literal rules are for supporting
    # concatenation of adjacent string literals.
    # I.e. "hello " "world" is seen by the C compiler as a single string literal
    # with the value "hello world"
    #
    @_('STRING_LITERAL')
    def unified_string_literal(self, p: Any):
        # single literal
        return c_ast.Constant('string', p.STRING_LITERAL, self._token_coord(p, 1))

    @_('unified_string_literal STRING_LITERAL')
    def unified_string_literal(self, p: Any):
        p.unified_string_literal.value = p.unified_string_literal.value[:-1] + p.STRING_LITERAL[1:]
        return p.unified_string_literal

    @_(
        'WSTRING_LITERAL',
        'U8STRING_LITERAL',
        'U16STRING_LITERAL',
        'U32STRING_LITERAL',
    )
    def unified_wstring_literal(self, p: Any):
        return c_ast.Constant('string', p[0], self._token_coord(p, 1))

    @_(
        'unified_wstring_literal WSTRING_LITERAL',
        'unified_wstring_literal U8STRING_LITERAL',
        'unified_wstring_literal U16STRING_LITERAL',
        'unified_wstring_literal U32STRING_LITERAL',
    )
    def unified_wstring_literal(self, p: Any):
        p.unified_wstring_literal.value = p.unified_wstring_literal.value.rstrip()[:-1] + p[2][2:]
        return p.unified_wstring_literal

    # ===============================================================================================
    # TODO: Experiment with braces (brace_open, brace_close)
    # ===============================================================================================

    @_('LBRACE')
    def brace_open(self, p: Any):
        return p[0]

    @_('RBRACE')
    def brace_close(self, p: Any):
        return p[0]

    @_('')
    def empty(self, p: Any):
        return None
