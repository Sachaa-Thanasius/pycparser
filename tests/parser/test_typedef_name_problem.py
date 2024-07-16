"""Test issues related to the typedef-name problem."""

import pytest
from cparsing import CParsingError, c_ast, parse


@pytest.mark.xfail(reason="TODO")
def test_innerscope_typedef():
    # should succeed since TT is not a type in bar
    test_input = r"""
void foo() {
    typedef char TT;
    TT x;
}
void bar() {
    unsigned TT;
}
"""

    assert isinstance(parse(test_input), c_ast.File)


def test_innerscope_typedef_error():
    # should fail since TT is not a type in bar
    test_input = r"""
void foo() {
    typedef char TT;
    TT x;
}
void bar() {
    TT y;
}
"""

    with pytest.raises(CParsingError):
        parse(test_input)


@pytest.mark.parametrize(
    ("test_input", "expected_inner_param_1", "expected_inner_param_2"),
    [
        pytest.param(
            "typedef char TT;\nint foo(int (aa));\nint bar(int (TT));",
            c_ast.Decl("aa", c_ast.TypeDecl("aa", type=c_ast.IdType(["int"]))),
            c_ast.Typename(
                None,
                quals=[],
                align=None,
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.TypeDecl(type=c_ast.IdType(["TT"])),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                ),
            ),
            id="foo takes an int named aa; bar takes a function taking a TT",
        ),
        pytest.param(
            "typedef char TT;\nint foo(int (aa (char)));\nint bar(int (TT (char)));",
            c_ast.Decl(
                "aa",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.TypeDecl(type=c_ast.IdType(["char"])),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("aa", type=c_ast.IdType(["int"])),
                ),
            ),
            c_ast.Typename(
                None,
                quals=[],
                align=None,
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.FuncDecl(
                                    args=c_ast.ParamList(
                                        [
                                            c_ast.Typename(
                                                None,
                                                quals=[],
                                                align=None,
                                                type=c_ast.TypeDecl(type=c_ast.IdType(["char"])),
                                            )
                                        ]
                                    ),
                                    type=c_ast.TypeDecl(type=c_ast.IdType(["TT"])),
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                ),
            ),
            id="foo takes a function taking a char; bar takes a function taking a function taking a char",
        ),
        pytest.param(
            "typedef char TT;\nint foo(int (aa[]));\nint bar(int (TT[]));",
            c_ast.Decl(
                "aa",
                c_ast.ArrayDecl(type=c_ast.TypeDecl("aa", type=c_ast.IdType(["int"])), dim=None, dim_quals=[]),
            ),
            c_ast.Typename(
                None,
                quals=[],
                align=None,
                type=c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.ArrayDecl(
                                    type=c_ast.TypeDecl(type=c_ast.IdType(["TT"])), dim=None, dim_quals=[]
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                ),
            ),
            id="foo takes an int array named aa; bar takes a function taking a TT array",
        ),
    ],
)
def test_ambiguous_parameters(test_input: str, expected_inner_param_1: c_ast.AST, expected_inner_param_2: c_ast.AST):
    # From ISO/IEC 9899:TC2, 6.7.5.3.11:
    # "If, in a parameter declaration, an identifier can be treated either
    #  as a typedef name or as a parameter name, it shall be taken as a
    #  typedef name."

    tree = parse(test_input)
    assert tree.ext[1].type.args.params[0] == expected_inner_param_1  # pyright: ignore
    assert tree.ext[2].type.args.params[0] == expected_inner_param_2  # pyright: ignore


@pytest.mark.xfail(reason="TODO")
def test_innerscope_reuse_typedef_name():
    # identifiers can be reused in inner scopes; the original should be restored at the end of the block

    test_input_1 = r"""
typedef char TT;
void foo(void) {
    unsigned TT;
    TT = 10;
}
TT x = 5;
"""
    tree1 = parse(test_input_1)

    expected_before_end = c_ast.Decl("TT", c_ast.TypeDecl("TT", type=c_ast.IdType(["unsigned"])))
    expected_after_end = c_ast.Decl(
        "x", c_ast.TypeDecl("x", type=c_ast.IdType(["TT"])), init=c_ast.Constant("int", "5")
    )
    assert tree1.ext[1].body.block_items[0] == expected_before_end  # pyright: ignore
    assert tree1.ext[2] == expected_after_end

    # this should be recognized even with an initializer
    test_input_2 = r"""
typedef char TT;
void foo(void) {
    unsigned TT = 10;
}
"""
    tree2 = parse(test_input_2)

    expected = c_ast.Decl("TT", c_ast.TypeDecl("TT", type=c_ast.IdType(["unsigned"])), init=c_ast.Constant("int", "10"))
    assert tree2.ext[1].body.block_items[0] == expected  # pyright: ignore

    # before the second local variable, TT is a type; after, it's a
    # variable
    test_input_3 = r"""
typedef char TT;
void foo(void) {
    TT tt = sizeof(TT);
    unsigned TT = 10;
}
"""
    tree3 = parse(test_input_3)

    expected_before_end = c_ast.Decl(
        "tt",
        c_ast.TypeDecl("tt", type=c_ast.IdType(["TT"])),
        init=c_ast.UnaryOp(
            op="sizeof",
            expr=c_ast.Typename(None, quals=[], align=None, type=c_ast.TypeDecl(type=c_ast.IdType(["TT"]))),
        ),
    )
    expected_after_end = c_ast.Decl(
        "TT",
        c_ast.TypeDecl("TT", type=c_ast.IdType(names=["unsigned"])),
        init=c_ast.Constant("int", "10"),
    )
    assert tree3.ext[1].body.block_items[0] == expected_before_end  # pyright: ignore
    assert tree3.ext[1].body.block_items[1] == expected_after_end  # pyright: ignore

    # a variable and its type can even share the same name
    test_input_4 = r"""
typedef char TT;
void foo(void) {
    TT TT = sizeof(TT);
    unsigned uu = TT * 2;
}
"""
    tree4 = parse(test_input_4)

    expected_before_end = c_ast.Decl(
        "TT",
        c_ast.TypeDecl("TT", type=c_ast.IdType(["TT"])),
        init=c_ast.UnaryOp(
            op="sizeof",
            expr=c_ast.Typename(None, quals=[], align=None, type=c_ast.TypeDecl(type=c_ast.IdType(["TT"]))),
        ),
    )

    expected_after_end = c_ast.Decl(
        "uu",
        c_ast.TypeDecl("uu", type=c_ast.IdType(["unsigned"])),
        init=c_ast.BinaryOp(op="*", left=c_ast.Id("TT"), right=c_ast.Constant("int", "2")),
    )

    assert tree4.ext[1].body.block_items[0] == expected_before_end  # pyright: ignore
    assert tree4.ext[1].body.block_items[1] == expected_after_end  # pyright: ignore

    # ensure an error is raised if a type, redeclared as a variable, is
    # used as a type
    test_input_5 = r"""
typedef char TT;
void foo(void) {
    unsigned TT = 10;
    TT erroneous = 20;
}
"""
    with pytest.raises(CParsingError):
        parse(test_input_5)

    # reusing a type name should work with multiple declarators
    test_input_6 = r"""
typedef char TT;
void foo(void) {
    unsigned TT, uu;
}
"""
    tree6 = parse(test_input_6)

    expected_before_end = c_ast.Decl("TT", c_ast.TypeDecl("TT", type=c_ast.IdType(["unsigned"])))
    expected_after_end = c_ast.Decl("uu", c_ast.TypeDecl("uu", type=c_ast.IdType(["unsigned"])))

    assert tree6.ext[1].body.block_items[0] == expected_before_end  # pyright: ignore
    assert tree6.ext[1].body.block_items[1] == expected_after_end  # pyright: ignore

    # reusing a type name should work after a pointer
    test_input_7 = r"""
typedef char TT;
void foo(void) {
    unsigned * TT;
}
"""
    tree7 = parse(test_input_7)

    expected = c_ast.Decl("TT", c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("TT", type=c_ast.IdType(["unsigned"]))))
    assert tree7.ext[1].body.block_items[0] == expected  # pyright: ignore

    # redefine a name in the middle of a multi-declarator declaration
    test_input_8 = r"""
typedef char TT;
void foo(void) {
    int tt = sizeof(TT), TT, uu = sizeof(TT);
    int uu = sizeof(tt);
}
"""
    tree8 = parse(test_input_8)

    expected_first = c_ast.Decl(
        "tt",
        c_ast.TypeDecl("tt", type=c_ast.IdType(["int"])),
        init=c_ast.UnaryOp(
            op="sizeof",
            expr=c_ast.Typename(None, quals=[], align=None, type=c_ast.TypeDecl(type=c_ast.IdType(["TT"]))),
        ),
    )
    expected_second = c_ast.Decl("TT", c_ast.TypeDecl("TT", type=c_ast.IdType(["int"])))
    expected_third = c_ast.Decl(
        "uu",
        c_ast.TypeDecl("uu", type=c_ast.IdType(["int"])),
        init=c_ast.UnaryOp(
            op="sizeof",
            expr=c_ast.Typename(None, quals=[], align=None, type=c_ast.TypeDecl(type=c_ast.IdType(["TT"]))),
        ),
    )

    assert tree8.ext[1].body.block_items[0] == expected_first  # pyright: ignore
    assert tree8.ext[1].body.block_items[1] == expected_second  # pyright: ignore
    assert tree8.ext[1].body.block_items[2] == expected_third  # pyright: ignore

    # Don't test this until we have support for it
    # self.assertEqual(expand_init(items[0].init),
    #     ['UnaryOp', 'sizeof', ['Typename', ['TypeDecl', ['IdentifierType', ['TT']]]]])
    # self.assertEqual(expand_init(items[2].init),
    #     ['UnaryOp', 'sizeof', ['ID', 'TT']])


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        # identifiers can be reused as parameter names; parameter name scope
        # begins and ends with the function body; it's important that TT is
        # used immediately before the LBRACE or after the RBRACE, to test
        # a corner case
        pytest.param(
            r"""
typedef char TT;
void foo(unsigned TT, TT bar) {
    TT = 10;
}
TT x = 5;
""",
            c_ast.FuncDef(
                decl=c_ast.Decl(
                    "foo",
                    c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Decl("TT", c_ast.TypeDecl("TT", type=c_ast.IdType(["unsigned"]))),
                                c_ast.Decl("bar", c_ast.TypeDecl("bar", type=c_ast.IdType(["TT"]))),
                            ]
                        ),
                        type=c_ast.TypeDecl("foo", type=c_ast.IdType(["void"])),
                    ),
                ),
                param_decls=None,
                body=c_ast.Compound([c_ast.Assignment(op="=", left=c_ast.Id("TT"), right=c_ast.Constant("int", "10"))]),
            ),
            marks=pytest.mark.xfail(reason="TODO"),
        ),
        # the scope of a parameter name in a function declaration ends at the
        # end of the declaration...so it is effectively never used; it's
        # important that TT is used immediately after the declaration, to
        # test a corner case
        (
            r"""
typedef char TT;
void foo(unsigned TT, TT bar);
TT x = 5;
""",
            c_ast.Decl(
                "foo",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Decl("TT", c_ast.TypeDecl("TT", type=c_ast.IdType(["unsigned"]))),
                            c_ast.Decl("bar", c_ast.TypeDecl("bar", type=c_ast.IdType(["TT"]))),
                        ]
                    ),
                    type=c_ast.TypeDecl("foo", type=c_ast.IdType(["void"])),
                ),
            ),
        ),
    ],
)
def test_parameter_reuse_typedef_name(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    assert tree.ext[1] == expected


@pytest.mark.parametrize(
    "test_input",
    [
        pytest.param(
            r"""
typedef char TT;
void foo(unsigned TT, TT bar) {
    TT erroneous = 20;
}
""",
            id="Ensure error is raised if a type, redeclared as a parameter, is used as a type",
        )
    ],
)
def test_parameter_reuse_typedef_name_error(test_input: str):
    with pytest.raises(CParsingError):
        parse(test_input)


@pytest.mark.xfail(reason="TODO")
def test_nested_function_decls():
    # parameter names of nested function declarations must not escape into the top-level function _definition's_ scope;
    # the following must succeed because TT is still a typedef inside foo's body
    test_input = r"""
typedef char TT;
void foo(unsigned bar(int TT)) {
    TT x = 10;
}
"""
    assert isinstance(parse(test_input), c_ast.File)


@pytest.mark.parametrize(
    "test_input",
    [
        r"""
typedef char TT;
char TT = 5;
""",
        r"""
char TT = 5;
typedef char TT;
""",
    ],
)
def test_samescope_reuse_name(test_input: str):
    """A typedef name cannot be reused as an object name in the same scope, or vice versa."""

    with pytest.raises(CParsingError):
        parse(test_input)
