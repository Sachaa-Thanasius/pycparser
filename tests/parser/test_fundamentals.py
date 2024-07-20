# ruff: noqa: F403, F405
"""Test fundamentals of the parser."""

from typing import Union as TUnion

import pytest
from cparsing import CParsingError, parse
from cparsing.c_ast import *
from cparsing.utils import Coord


@pytest.mark.parametrize(
    ("test_input", "expected_length"),
    [
        pytest.param("int a; char c;", 2, id="nonempty file"),
        pytest.param("", 0, id="empty file"),
    ],
)
def test_ast_File(test_input: str, expected_length: int):
    tree = parse(test_input)
    assert isinstance(tree, File)
    assert len(tree.ext) == expected_length


@pytest.mark.parametrize("test_input", ["int foo;;"])
def test_empty_toplevel_decl(test_input: str):
    tree = parse(test_input)
    assert isinstance(tree, File)
    assert len(tree.ext) == 1

    expected_decl = Decl("foo", TypeDecl("foo", type=IdType(["int"])))
    assert tree.ext[0] == expected_decl


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (";", File([])),
        (";int foo;", File([Decl("foo", TypeDecl("foo", type=IdType(["int"])))])),
    ],
)
def test_initial_semi(test_input: str, expected: AST):
    tree = parse(test_input)
    assert tree == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("int a;", Decl("a", TypeDecl("a", type=IdType(["int"])))),
        ("unsigned int a;", Decl("a", TypeDecl("a", type=IdType(["unsigned", "int"])))),
        ("_Bool a;", Decl("a", TypeDecl("a", type=IdType(["_Bool"])))),
        ("float _Complex fcc;", Decl("fcc", TypeDecl("fcc", type=IdType(["float", "_Complex"])))),
        ("char* string;", Decl("string", PtrDecl([], type=TypeDecl("string", type=IdType(["char"]))))),
        (
            "long ar[15];",
            Decl("ar", ArrayDecl(type=TypeDecl("ar", type=IdType(["long"])), dim=Constant("int", "15"), dim_quals=[])),
        ),
        (
            "long long ar[15];",
            Decl(
                "ar",
                ArrayDecl(
                    type=TypeDecl("ar", type=IdType(["long", "long"])),
                    dim=Constant(type="int", value="15"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "unsigned ar[];",
            Decl("ar", ArrayDecl(type=TypeDecl("ar", type=IdType(["unsigned"])), dim=None, dim_quals=[])),
        ),
        (
            "int strlen(char* s);",
            Decl(
                "strlen",
                FuncDecl(
                    args=ParamList([Decl("s", PtrDecl(quals=[], type=TypeDecl("s", type=IdType(["char"]))))]),
                    type=TypeDecl("strlen", type=IdType(["int"])),
                ),
            ),
        ),
        (
            "int strcmp(char* s1, char* s2);",
            Decl(
                "strcmp",
                FuncDecl(
                    args=ParamList(
                        [
                            Decl("s1", PtrDecl(quals=[], type=TypeDecl("s1", type=IdType(["char"])))),
                            Decl("s2", PtrDecl(quals=[], type=TypeDecl("s2", type=IdType(["char"])))),
                        ]
                    ),
                    type=TypeDecl("strcmp", type=IdType(["int"])),
                ),
            ),
        ),
        pytest.param(
            "extern foobar(foo, bar);",
            Decl(
                "foobar",
                FuncDecl(args=ParamList([Id("foo"), Id("bar")]), type=TypeDecl("foobar", type=IdType(["int"]))),
                storage=["extern"],
            ),
            id="function return values and parameters may not have type information",
        ),
        pytest.param(
            "__int128 a;",
            Decl("a", TypeDecl("a", type=IdType(["__int128"]))),
            id=(
                "__int128: it isn't part of the core C99 or C11 standards, but is mentioned in both documents "
                "under 'Common Extensions'."
            ),
        ),
    ],
)
def test_simple_decls(test_input: str, expected: AST):
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "char** ar2D;",
            Decl("ar2D", PtrDecl(quals=[], type=PtrDecl(quals=[], type=TypeDecl("ar2D", type=IdType(["char"]))))),
        ),
        (
            "int (*a)[1][2];",
            Decl(
                "a",
                PtrDecl(
                    quals=[],
                    type=ArrayDecl(
                        type=ArrayDecl(
                            type=TypeDecl("a", type=IdType(["int"])),
                            dim=Constant("int", "2"),
                            dim_quals=[],
                        ),
                        dim=Constant("int", "1"),
                        dim_quals=[],
                    ),
                ),
            ),
        ),
        (
            "int *a[1][2];",
            Decl(
                "a",
                ArrayDecl(
                    type=ArrayDecl(
                        type=PtrDecl(quals=[], type=TypeDecl("a", type=IdType(["int"]))),
                        dim=Constant("int", "2"),
                        dim_quals=[],
                    ),
                    dim=Constant("int", "1"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char* const* p;",
            Decl("p", PtrDecl(quals=[], type=PtrDecl(quals=["const"], type=TypeDecl("p", type=IdType(["char"]))))),
        ),
        (
            "const char* const* p;",
            Decl(
                "p",
                PtrDecl(
                    quals=[],
                    type=PtrDecl(quals=["const"], type=TypeDecl("p", quals=["const"], type=IdType(["char"]))),
                ),
                quals=["const"],
            ),
        ),
        (
            "char* * const p;",
            Decl(
                "p",
                PtrDecl(quals=["const"], type=PtrDecl(quals=[], type=TypeDecl("p", type=IdType(["char"])))),
            ),
        ),
        (
            "char ***ar3D[40];",
            Decl(
                "ar3D",
                ArrayDecl(
                    type=PtrDecl(
                        quals=[],
                        type=PtrDecl(quals=[], type=PtrDecl(quals=[], type=TypeDecl("ar3D", type=IdType(["char"])))),
                    ),
                    dim=Constant("int", "40"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char (***ar3D)[40];",
            Decl(
                "ar3D",
                PtrDecl(
                    quals=[],
                    type=PtrDecl(
                        quals=[],
                        type=PtrDecl(
                            quals=[],
                            type=ArrayDecl(
                                type=TypeDecl("ar3D", type=IdType(["char"])),
                                dim=Constant("int", "40"),
                                dim_quals=[],
                            ),
                        ),
                    ),
                ),
            ),
        ),
        (
            "int (*const*const x)(char, int);",
            Decl(
                "x",
                PtrDecl(
                    quals=["const"],
                    type=PtrDecl(
                        quals=["const"],
                        type=FuncDecl(
                            args=ParamList(
                                [
                                    Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["char"]))),
                                    Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["int"]))),
                                ]
                            ),
                            type=TypeDecl("x", type=IdType(["int"])),
                        ),
                    ),
                ),
            ),
        ),
        (
            "int (*x[4])(char, int);",
            Decl(
                "x",
                ArrayDecl(
                    type=PtrDecl(
                        quals=[],
                        type=FuncDecl(
                            args=ParamList(
                                [
                                    Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["char"]))),
                                    Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["int"]))),
                                ]
                            ),
                            type=TypeDecl("x", type=IdType(["int"])),
                        ),
                    ),
                    dim=Constant("int", "4"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char *(*(**foo [][8])())[];",
            Decl(
                "foo",
                ArrayDecl(
                    type=ArrayDecl(
                        type=PtrDecl(
                            quals=[],
                            type=PtrDecl(
                                quals=[],
                                type=FuncDecl(
                                    args=None,
                                    type=PtrDecl(
                                        quals=[],
                                        type=ArrayDecl(
                                            type=PtrDecl(quals=[], type=TypeDecl("foo", type=IdType(["char"]))),
                                            dim=None,
                                            dim_quals=[],
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        dim=Constant("int", "8"),
                        dim_quals=[],
                    ),
                    dim=None,
                    dim_quals=[],
                ),
            ),
        ),
        pytest.param(
            "int (*k)(int);",
            Decl(
                "k",
                PtrDecl(
                    quals=[],
                    type=FuncDecl(
                        args=ParamList([Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["int"])))]),
                        type=TypeDecl("k", type=IdType(["int"])),
                    ),
                ),
            ),
            id="unnamed function pointer parameters w/o quals",
        ),
        pytest.param(
            "int (*k)(const int);",
            Decl(
                "k",
                PtrDecl(
                    quals=[],
                    type=FuncDecl(
                        args=ParamList(
                            [
                                Typename(
                                    None,
                                    quals=["const"],
                                    align=None,
                                    type=TypeDecl(quals=["const"], type=IdType(["int"])),
                                )
                            ]
                        ),
                        type=TypeDecl("k", type=IdType(["int"])),
                    ),
                ),
            ),
            id="unnamed function pointer parameters w/ quals",
        ),
        pytest.param(
            "int (*k)(int q);",
            Decl(
                "k",
                PtrDecl(
                    quals=[],
                    type=FuncDecl(
                        args=ParamList([Decl("q", TypeDecl("q", type=IdType(["int"])))]),
                        type=TypeDecl("k", type=IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/o quals",
        ),
        pytest.param(
            "int (*k)(const volatile int q);",
            Decl(
                "k",
                PtrDecl(
                    quals=[],
                    type=FuncDecl(
                        args=ParamList(
                            [
                                Decl(
                                    "q",
                                    TypeDecl("q", ["const", "volatile"], type=IdType(["int"])),
                                    quals=["const", "volatile"],
                                )
                            ]
                        ),
                        type=TypeDecl("k", [], type=IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 1",
        ),
        pytest.param(
            "int (*k)(_Atomic volatile int q);",
            Decl(
                "k",
                PtrDecl(
                    quals=[],
                    type=FuncDecl(
                        args=ParamList(
                            [
                                Decl(
                                    "q",
                                    TypeDecl("q", ["_Atomic", "volatile"], type=IdType(["int"])),
                                    quals=["_Atomic", "volatile"],
                                )
                            ]
                        ),
                        type=TypeDecl("k", type=IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 2",
        ),
        pytest.param(
            "int (*k)(const volatile int* q);",
            Decl(
                "k",
                PtrDecl(
                    quals=[],
                    type=FuncDecl(
                        args=ParamList(
                            [
                                Decl(
                                    "q",
                                    PtrDecl(quals=[], type=TypeDecl("q", ["const", "volatile"], type=IdType(["int"]))),
                                    quals=["const", "volatile"],
                                )
                            ]
                        ),
                        type=TypeDecl("k", type=IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 3",
        ),
        pytest.param(
            "int (*k)(restrict int* q);",
            Decl(
                "k",
                PtrDecl(
                    quals=[],
                    type=FuncDecl(
                        args=ParamList(
                            [
                                Decl(
                                    "q",
                                    PtrDecl(quals=[], type=TypeDecl("q", ["restrict"], type=IdType(["int"]))),
                                    quals=["restrict"],
                                )
                            ]
                        ),
                        type=TypeDecl("k", type=IdType(["int"])),
                    ),
                ),
            ),
            id="restrict qualifier",
        ),
    ],
)
def test_nested_decls(test_input: str, expected: AST):
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            "int zz(int p[static 10]);",
            Decl(
                "zz",
                FuncDecl(
                    args=ParamList(
                        [
                            Decl(
                                "p",
                                ArrayDecl(
                                    type=TypeDecl("p", type=IdType(["int"])),
                                    dim=Constant("int", "10"),
                                    dim_quals=["static"],
                                ),
                            )
                        ]
                    ),
                    type=TypeDecl("zz", type=IdType(["int"])),
                ),
            ),
            id="named function parameter 1",
        ),
        pytest.param(
            "int zz(int p[const 10]);",
            Decl(
                "zz",
                FuncDecl(
                    args=ParamList(
                        [
                            Decl(
                                "p",
                                ArrayDecl(
                                    type=TypeDecl("p", type=IdType(["int"])),
                                    dim=Constant("int", "10"),
                                    dim_quals=["const"],
                                ),
                            )
                        ]
                    ),
                    type=TypeDecl("zz", type=IdType(["int"])),
                ),
            ),
            id="named function parameter 2",
        ),
        pytest.param(
            "int zz(int p[restrict][5]);",
            Decl(
                "zz",
                FuncDecl(
                    args=ParamList(
                        [
                            Decl(
                                "p",
                                ArrayDecl(
                                    type=ArrayDecl(
                                        type=TypeDecl("p", type=IdType(["int"])),
                                        dim=Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=None,
                                    dim_quals=["restrict"],
                                ),
                            )
                        ]
                    ),
                    type=TypeDecl("zz", type=IdType(["int"])),
                ),
            ),
            id="named function parameter 3",
        ),
        pytest.param(
            "int zz(int p[const restrict static 10][5]);",
            Decl(
                "zz",
                FuncDecl(
                    args=ParamList(
                        [
                            Decl(
                                "p",
                                ArrayDecl(
                                    type=ArrayDecl(
                                        type=TypeDecl("p", type=IdType(["int"])),
                                        dim=Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=Constant("int", "10"),
                                    dim_quals=["const", "restrict", "static"],
                                ),
                            )
                        ]
                    ),
                    type=TypeDecl("zz", type=IdType(["int"])),
                ),
            ),
            id="named function parameter 4",
        ),
        pytest.param(
            "int zz(int [const 10]);",
            Decl(
                "zz",
                FuncDecl(
                    args=ParamList(
                        [
                            Typename(
                                None,
                                quals=[],
                                align=None,
                                type=ArrayDecl(
                                    type=TypeDecl(type=IdType(["int"])),
                                    dim=Constant("int", "10"),
                                    dim_quals=["const"],
                                ),
                            )
                        ]
                    ),
                    type=TypeDecl("zz", type=IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 1",
        ),
        pytest.param(
            "int zz(int [restrict][5]);",
            Decl(
                "zz",
                FuncDecl(
                    args=ParamList(
                        [
                            Typename(
                                None,
                                quals=[],
                                align=None,
                                type=ArrayDecl(
                                    type=ArrayDecl(
                                        type=TypeDecl(type=IdType(["int"])),
                                        dim=Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=None,
                                    dim_quals=["restrict"],
                                ),
                            )
                        ]
                    ),
                    type=TypeDecl("zz", type=IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 2",
        ),
        pytest.param(
            "int zz(int [const restrict volatile 10][5]);",
            Decl(
                "zz",
                FuncDecl(
                    args=ParamList(
                        [
                            Typename(
                                None,
                                quals=[],
                                align=None,
                                type=ArrayDecl(
                                    type=ArrayDecl(
                                        type=TypeDecl(type=IdType(["int"])),
                                        dim=Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=Constant("int", "10"),
                                    dim_quals=["const", "restrict", "volatile"],
                                ),
                            )
                        ]
                    ),
                    type=TypeDecl("zz", type=IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 3",
        ),
    ],
)
def test_func_decls_with_array_dim_qualifiers(test_input: str, expected: AST):
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.parametrize(
    ("test_input", "index", "expected_quals", "expected_storage"),
    [
        ("extern int p;", 0, [], ["extern"]),
        ("_Thread_local int p;", 0, [], ["_Thread_local"]),
        ("const long p = 6;", 0, ["const"], []),
        ("_Atomic int p;", 0, ["_Atomic"], []),
        ("_Atomic restrict int* p;", 0, ["_Atomic", "restrict"], []),
        ("static const int p, q, r;", 0, ["const"], ["static"]),
        ("static const int p, q, r;", 1, ["const"], ["static"]),
        ("static const int p, q, r;", 2, ["const"], ["static"]),
        ("static char * const p;", 0, [], ["static"]),
    ],
)
def test_qualifiers_storage_specifiers_1(
    test_input: str,
    index: int,
    expected_quals: list[str],
    expected_storage: list[str],
):
    tree = parse(test_input).ext[index]

    assert isinstance(tree, Decl)
    assert tree.quals == expected_quals
    assert tree.storage == expected_storage


def test_qualifiers_storage_specifiers_2():
    test_input = "static char * const p;"
    tree = parse(test_input)

    assert isinstance(tree.ext[0], Decl)

    pdecl = tree.ext[0].type
    assert isinstance(pdecl, PtrDecl)
    assert pdecl.quals == ["const"]


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        (
            "_Atomic(int) ai;",
            0,
            Decl("ai", TypeDecl("ai", ["_Atomic"], type=IdType(["int"])), quals=["_Atomic"]),
        ),
        (
            "_Atomic(int*) ai;",
            0,
            Decl("ai", PtrDecl(quals=["_Atomic"], type=TypeDecl("ai", type=IdType(["int"])))),
        ),
        (
            "_Atomic(_Atomic(int)*) aai;",
            0,
            Decl(
                "aai",
                PtrDecl(quals=["_Atomic"], type=TypeDecl("aai", ["_Atomic"], type=IdType(["int"]))),
                quals=["_Atomic"],
            ),
        ),
        pytest.param(
            "_Atomic(int) foo, bar;",
            slice(0, 2),
            [
                Decl("foo", TypeDecl("foo", ["_Atomic"], type=IdType(["int"])), quals=["_Atomic"]),
                Decl("bar", TypeDecl("foo", ["_Atomic"], type=IdType(["int"])), quals=["_Atomic"]),
            ],
            id="multiple declarations",
        ),
        pytest.param(
            "typedef _Atomic(int) atomic_int;",
            0,
            Typedef(
                "atomic_int",
                quals=["_Atomic"],
                storage=["typedef"],
                type=TypeDecl("atomic_int", ["_Atomic"], type=IdType(["int"])),
            ),
            id="typedefs with _Atomic specifiers 1",
        ),
        pytest.param(
            "typedef _Atomic(_Atomic(_Atomic(int (*)(void)) *) *) t;",
            0,
            Typedef(
                "t",
                quals=[],
                storage=["typedef"],
                type=PtrDecl(
                    quals=["_Atomic"],
                    type=PtrDecl(
                        quals=["_Atomic"],
                        type=PtrDecl(
                            quals=["_Atomic"],
                            type=FuncDecl(
                                args=ParamList(
                                    [Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["void"])))]
                                ),
                                type=TypeDecl("t", type=IdType(["int"])),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_atomic_specifier(test_input: str, index: TUnion[int, slice], expected: TUnion[AST, list[AST]]):
    tree = parse(test_input)
    decl = tree.ext[index]
    assert compare(decl, expected)


@pytest.mark.parametrize(
    ("test_input", "expected_compound_block_items"),
    [
        (
            """\
void foo()
{
    int a = sizeof k;
    int b = sizeof(int);
    int c = sizeof(int**);;

    char* p = "just to make sure this parses w/o error...";
    int d = sizeof(int());
}
""",
            (
                UnaryOp(op="sizeof", expr=Id("k")),
                UnaryOp(op="sizeof", expr=Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["int"])))),
                UnaryOp(
                    op="sizeof",
                    expr=Typename(
                        None,
                        quals=[],
                        align=None,
                        type=PtrDecl(quals=[], type=PtrDecl(quals=[], type=TypeDecl(type=IdType(["int"])))),
                    ),
                ),
            ),
        )
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_sizeof(test_input: str, expected_compound_block_items: tuple[AST, ...]) -> None:
    tree = parse(test_input)
    compound = tree.ext[0].body  # pyright: ignore
    assert isinstance(compound, Compound)
    assert compound.block_items

    for index, expected in enumerate(expected_compound_block_items):
        found_init = compound.block_items[index].init  # pyright: ignore
        assert isinstance(found_init, UnaryOp)
        assert found_init == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = _Alignof(int);",
            Decl(
                "a",
                TypeDecl("a", type=IdType(["int"])),
                init=UnaryOp(
                    op="_Alignof",
                    expr=Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["int"]))),
                ),
            ),
        ),
        (
            "_Alignas(_Alignof(int)) char a;",
            Decl(
                "a",
                TypeDecl("a", type=IdType(["char"])),
                align=[
                    Alignas(
                        alignment=UnaryOp(
                            op="_Alignof",
                            expr=Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["int"]))),
                        )
                    )
                ],
            ),
        ),
        (
            "_Alignas(4) char a;",
            Decl("a", TypeDecl("a", type=IdType(["char"])), align=[Alignas(Constant("int", "4"))]),
        ),
        (
            "_Alignas(int) char a;",
            Decl(
                "a",
                TypeDecl("a", type=IdType(["char"])),
                align=[Alignas(Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["int"]))))],
            ),
        ),
    ],
)
def test_alignof(test_input: str, expected: AST) -> None:
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.xfail(reason="TODO")
def test_offsetof():
    test_input = """\
void foo() {
    int a = offsetof(struct S, p);
    a.b = offsetof(struct sockaddr, sp) + strlen(bar);
    int a = offsetof(struct S, p.q.r);
    int a = offsetof(struct S, p[5].q[4][5]);
}
"""

    expected_list = [
        Decl(
            "a",
            TypeDecl("a", type=IdType(names=["int"])),
            init=FuncCall(
                Id("offsetof"),
                args=ExprList(
                    [Typename(None, quals=[], align=None, type=TypeDecl(type=Struct("S", decls=None))), Id("p")]
                ),
            ),
        ),
        Assignment(
            op="=",
            left=StructRef(Id("a"), type=".", field=Id("b")),
            right=BinaryOp(
                op="+",
                left=FuncCall(
                    Id("offsetof"),
                    args=ExprList(
                        [
                            Typename(None, quals=[], align=None, type=TypeDecl(type=Struct("sockaddr", decls=None))),
                            Id("sp"),
                        ]
                    ),
                ),
                right=FuncCall(Id("strlen"), args=ExprList([Id("bar")])),
            ),
        ),
        Decl(
            "a",
            TypeDecl("a", type=IdType(["int"])),
            init=FuncCall(
                Id("offsetof"),
                args=ExprList(
                    [
                        Typename(None, quals=[], align=None, type=TypeDecl(type=Struct("S", decls=None))),
                        StructRef(StructRef(Id("p"), type=".", field=Id("q")), type=".", field=Id("r")),
                    ]
                ),
            ),
        ),
        Decl(
            "a",
            TypeDecl("a", type=IdType(["int"])),
            init=FuncCall(
                Id("offsetof"),
                args=ExprList(
                    [
                        Typename(None, quals=[], align=None, type=TypeDecl(type=Struct("S", decls=None))),
                        ArrayRef(
                            name=ArrayRef(
                                name=StructRef(
                                    name=ArrayRef(name=Id("p"), subscript=Constant("int", "5")),
                                    type=".",
                                    field=Id("q"),
                                ),
                                subscript=Constant("int", "4"),
                            ),
                            subscript=Constant("int", "5"),
                        ),
                    ]
                ),
            ),
        ),
    ]

    tree = parse(test_input)
    assert compare(tree.ext[0].body.block_items, expected_list)  # pyright: ignore


# @pytest.mark.xfail(reason="TODO")
def test_compound_statement() -> None:
    test_input = """\
void foo() {
}
"""

    tree = parse(test_input)

    compound = tree.ext[0].body  # pyright: ignore
    assert isinstance(compound, Compound)
    assert compound.coord == Coord(2, 0, filename="<unknown>")


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        pytest.param(
            0,
            CompoundLiteral(
                type=Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(names=["long", "long"]))),
                init=InitList([Id("k")]),
            ),
            id="C99 compound literal feature 1",
        ),
        pytest.param(
            1,
            CompoundLiteral(
                type=Typename(None, quals=[], align=None, type=TypeDecl(type=Struct("jk", decls=None))),
                init=InitList(
                    [
                        NamedInitializer(
                            name=[Id("a")],
                            expr=InitList([Constant("int", "1"), Constant(type="int", value="2")]),
                        ),
                        NamedInitializer(name=[Id("b"), Constant("int", "0")], expr=Id("t")),
                    ]
                ),
            ),
            id="C99 compound literal feature 2",
        ),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_compound_literals(index: int, expected: CompoundLiteral) -> None:
    test_input = r"""
void foo() {
    p = (long long){k};
    tc = (struct jk){.a = {1, 2}, .b[0] = t};
}"""

    tree = parse(test_input)

    compound = tree.ext[0].body.block_items[index].right  # pyright: ignore
    assert isinstance(compound, CompoundLiteral)
    assert compound == expected


@pytest.mark.xfail(reason="TODO")
def test_parenthesized_compounds() -> None:
    test_input = r"""
void foo() {
    int a;
    ({});
    ({ 1; });
    ({ 1; 2; });
    int b = ({ 1; });
    int c, d = ({ int x = 1; x + 2; });
    a = ({ int x = 1; 2 * x; });
}"""

    expected = [
        Decl("a", TypeDecl("a", type=IdType(["int"]))),
        Compound(block_items=None),
        Compound([Constant("int", value="1")]),
        Compound([Constant("int", value="1"), Constant("int", "2")]),
        Decl("b", TypeDecl("b", type=IdType(["int"])), init=Compound([Constant("int", "1")])),
        Decl("c", TypeDecl("c", type=IdType(["int"]))),
        Decl(
            "d",
            TypeDecl("d", type=IdType(["int"])),
            init=Compound(
                [
                    Decl("x", TypeDecl("x", type=IdType(["int"])), init=Constant("int", "1")),
                    BinaryOp(op="+", left=Id("x"), right=Constant("int", "2")),
                ]
            ),
        ),
        Assignment(
            op="=",
            left=Id("a"),
            right=Compound(
                [
                    Decl("x", TypeDecl("x", type=IdType(["int"])), init=Constant("int", "1")),
                    BinaryOp(op="*", left=Constant("int", "2"), right=Id("x")),
                ]
            ),
        ),
    ]

    tree = parse(test_input)
    assert compare(tree.ext[0].body.block_items, expected)  # pyright: ignore


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("enum mycolor op;", Enum("mycolor")),
        (
            "enum mysize {large=20, small, medium} shoes;",
            Enum(
                "mysize",
                EnumeratorList(
                    [
                        Enumerator("large", value=Constant("int", "20")),
                        Enumerator("small"),
                        Enumerator("medium"),
                    ]
                ),
            ),
        ),
        pytest.param(
            "enum\n{\n    red,\n    blue,\n    green,\n} color;",
            Enum(
                None,
                EnumeratorList(
                    [
                        Enumerator("red"),
                        Enumerator("blue"),
                        Enumerator("green"),
                    ]
                ),
            ),
            id="enum with trailing comma (C99 feature)",
        ),
    ],
)
def test_enums(test_input: str, expected: AST) -> None:
    tree = parse(test_input)
    enum_type = tree.ext[0].type.type  # pyright: ignore
    assert isinstance(enum_type, Enum)
    assert enum_type == expected


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        pytest.param(
            "typedef void* node;\nnode k;",
            slice(2),
            [
                Typedef(
                    "node",
                    quals=[],
                    storage=["typedef"],
                    type=PtrDecl(quals=[], type=TypeDecl("node", type=IdType(["void"]))),
                ),
                Decl("k", TypeDecl("k", type=IdType(["node"]))),
            ],
            id="with typedef",
        ),
        (
            "typedef int T;\ntypedef T *pT;\n\npT aa, bb;",
            3,
            Decl("bb", TypeDecl("bb", type=IdType(["pT"]))),
        ),
        (
            "typedef char* __builtin_va_list;\ntypedef __builtin_va_list __gnuc_va_list;",
            1,
            Typedef(
                "__gnuc_va_list",
                quals=[],
                storage=["typedef"],
                type=TypeDecl("__gnuc_va_list", type=IdType(["__builtin_va_list"])),
            ),
        ),
        (
            "typedef struct tagHash Hash;",
            0,
            Typedef("Hash", quals=[], storage=["typedef"], type=TypeDecl("Hash", type=Struct("tagHash", decls=None))),
        ),
        (
            "typedef int (* const * const T)(void);",
            0,
            Typedef(
                "T",
                quals=[],
                storage=["typedef"],
                type=PtrDecl(
                    quals=["const"],
                    type=PtrDecl(
                        quals=["const"],
                        type=FuncDecl(
                            args=ParamList(
                                [Typename(None, quals=[], align=None, type=TypeDecl(type=IdType(["void"])))]
                            ),
                            type=TypeDecl("T", type=IdType(["int"])),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_typedef(test_input: str, index: TUnion[int, slice], expected: TUnion[AST, list[AST]]) -> None:
    tree = parse(test_input)
    assert compare(tree.ext[index], expected)


@pytest.mark.parametrize(
    "test_input",
    [pytest.param("node k;", id="without typedef")],
)
def test_typedef_error(test_input: str):
    with pytest.raises(CParsingError):
        parse(test_input)


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        (
            "struct {\n    int id;\n    char* name;\n} joe;",
            0,
            Decl(
                "joe",
                TypeDecl(
                    "joe",
                    type=Struct(
                        None,
                        decls=[
                            Decl("id", TypeDecl("id", type=IdType(["int"]))),
                            Decl("name", PtrDecl(quals=[], type=TypeDecl("name", type=IdType(["char"])))),
                        ],
                    ),
                ),
            ),
        ),
        (
            "struct node p;",
            0,
            Decl("p", TypeDecl("p", type=Struct("node"))),
        ),
        (
            "union pri ra;",
            0,
            Decl("ra", TypeDecl("ra", type=Union("pri"))),
        ),
        (
            "struct node* p;",
            0,
            Decl("p", PtrDecl(quals=[], type=TypeDecl("p", type=Struct("node")))),
        ),
        (
            "struct node;",
            0,
            Decl(None, type=Struct("node")),
        ),
        (
            "union\n"
            "{\n"
            "    struct\n"
            "    {\n"
            "        int type;\n"
            "    } n;\n"
            "\n"
            "    struct\n"
            "    {\n"
            "        int type;\n"
            "        int intnode;\n"
            "    } ni;\n"
            "} u;",
            0,
            Decl(
                "u",
                TypeDecl(
                    "u",
                    type=Union(
                        None,
                        decls=[
                            Decl(
                                "n",
                                type=TypeDecl(
                                    "n",
                                    type=Struct(None, decls=[Decl("type", TypeDecl("type", type=IdType(["int"])))]),
                                ),
                            ),
                            Decl(
                                "ni",
                                TypeDecl(
                                    "ni",
                                    type=Struct(
                                        None,
                                        decls=[
                                            Decl("type", TypeDecl("type", type=IdType(["int"]))),
                                            Decl("intnode", TypeDecl("intnode", type=IdType(["int"]))),
                                        ],
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
        (
            "typedef struct foo_tag\n" "{\n" "    void* data;\n" "} foo, *pfoo;",
            slice(2),
            [
                Typedef(
                    "foo",
                    quals=[],
                    storage=["typedef"],
                    type=TypeDecl(
                        "foo",
                        type=Struct(
                            "foo_tag",
                            decls=[Decl("data", PtrDecl(quals=[], type=TypeDecl("data", type=IdType(["void"]))))],
                        ),
                    ),
                ),
                Typedef(
                    "pfoo",
                    quals=[],
                    storage=["typedef"],
                    type=PtrDecl(
                        quals=[],
                        type=TypeDecl(
                            "pfoo",
                            type=Struct(
                                "foo_tag",
                                decls=[Decl("data", PtrDecl(quals=[], type=TypeDecl("data", type=IdType(["void"]))))],
                            ),
                        ),
                    ),
                ),
            ],
        ),
        (
            "typedef enum tagReturnCode {SUCCESS, FAIL} ReturnCode;\n"
            "\n"
            "typedef struct tagEntry\n"
            "{\n"
            "    char* key;\n"
            "    char* value;\n"
            "} Entry;\n"
            "\n"
            "\n"
            "typedef struct tagNode\n"
            "{\n"
            "    Entry* entry;\n"
            "\n"
            "    struct tagNode* next;\n"
            "} Node;\n"
            "\n"
            "typedef struct tagHash\n"
            "{\n"
            "    unsigned int table_size;\n"
            "\n"
            "    Node** heads;\n"
            "\n"
            "} Hash;\n",
            3,
            Typedef(
                "Hash",
                quals=[],
                storage=["typedef"],
                type=TypeDecl(
                    declname="Hash",
                    type=Struct(
                        "tagHash",
                        decls=[
                            Decl("table_size", type=TypeDecl("table_size", type=IdType(["unsigned", "int"]))),
                            Decl(
                                "heads",
                                PtrDecl(
                                    quals=[],
                                    type=PtrDecl(quals=[], type=TypeDecl("heads", type=IdType(["Node"]))),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_union(test_input: str, index: TUnion[int, slice], expected: TUnion[AST, list[AST]]) -> None:
    tree = parse(test_input)
    type_ = tree.ext[index]
    assert compare(type_, expected)


@pytest.mark.xfail(reason="TODO")
def test_struct_with_line_pp():
    test_input = r"""
struct _on_exit_args {
    void *  _fnargs[32];
    void *  _dso_handle[32];

    long _fntypes;
    #line 77 "D:\eli\cpp_stuff\libc_include/sys/reent.h"

    long _is_cxa;
};
"""

    tree = parse(test_input, filename="test.c")
    assert tree.ext[0].type.decls[2].coord == Coord(6, 22, filename="test.c")  # pyright: ignore
    assert tree.ext[0].type.decls[3].coord == Coord(78, 22, filename=r"D:\eli\cpp_stuff\libc_include/sys/reent.h")  # pyright: ignore


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct Foo {\n   enum Bar { A = 1 };\n};",
            Decl(
                None,
                Struct("Foo", decls=[Decl(None, Enum("Bar", EnumeratorList([Enumerator("A", Constant("int", "1"))])))]),
            ),
        ),
        (
            "struct Foo {\n    enum Bar { A = 1, B, C } bar;\n    enum Baz { D = A } baz;\n} foo;",
            Decl(
                "foo",
                TypeDecl(
                    "foo",
                    type=Struct(
                        "Foo",
                        decls=[
                            Decl(
                                "bar",
                                TypeDecl(
                                    "bar",
                                    type=Enum(
                                        "Bar",
                                        EnumeratorList(
                                            [
                                                Enumerator("A", Constant("int", "1")),
                                                Enumerator("B"),
                                                Enumerator("C"),
                                            ]
                                        ),
                                    ),
                                ),
                            ),
                            Decl("baz", TypeDecl("baz", type=Enum("Baz", EnumeratorList([Enumerator("D", Id("A"))])))),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_enum(test_input: str, expected: AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct {\n    int a;;\n} foo;",
            Decl("foo", TypeDecl("foo", type=Struct(None, decls=[Decl("a", TypeDecl("a", type=IdType(["int"])))]))),
        ),
        (
            "struct {\n    int a;;;;\n    float b, c;\n    ;;\n    char d;\n} foo;",
            Decl(
                "foo",
                type=TypeDecl(
                    "foo",
                    type=Struct(
                        None,
                        decls=[
                            Decl("a", TypeDecl("a", type=IdType(["int"]))),
                            Decl("b", TypeDecl("b", type=IdType(["float"]))),
                            Decl("c", TypeDecl("c", type=IdType(["float"]))),
                            Decl("d", TypeDecl("d", type=IdType(["char"]))),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_with_extra_semis_inside(test_input: str, expected: AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct {\n    ;int a;\n} foo;",
            Decl("foo", TypeDecl("foo", type=Struct(None, decls=[Decl("a", TypeDecl("a", type=IdType(["int"])))]))),
        )
    ],
)
def test_struct_with_initial_semi(test_input: str, expected: AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            "union\n"
            "{\n"
            "    union\n"
            "    {\n"
            "        int i;\n"
            "        long l;\n"
            "    };\n"
            "\n"
            "    struct\n"
            "    {\n"
            "        int type;\n"
            "        int intnode;\n"
            "    };\n"
            "} u;",
            FuncDef(
                decl=Decl("foo", FuncDecl(args=None, type=TypeDecl("foo", type=IdType(["void"])))),
                param_decls=None,
                body=Compound(
                    [
                        Decl("a", TypeDecl("a", type=IdType(["int"]))),
                        Compound(block_items=None),
                        Compound(block_items=[Constant("int", "1")]),
                        Compound(block_items=[Constant("int", "1"), Constant("int", "2")]),
                        Decl("b", TypeDecl("b", type=IdType(["int"])), init=Compound([Constant("int", "1")])),
                        Decl("c", TypeDecl("c", type=IdType(["int"]))),
                        Decl(
                            "d",
                            TypeDecl("d", type=IdType(["int"])),
                            init=Compound(
                                [
                                    Decl("x", TypeDecl("x", type=IdType(["int"])), init=Constant("int", "1")),
                                    BinaryOp(op="+", left=Id("x"), right=Constant("int", "2")),
                                ]
                            ),
                        ),
                        Assignment(
                            op="=",
                            left=Id("a"),
                            right=Compound(
                                [
                                    Decl("x", TypeDecl("x", type=IdType(["int"])), init=Constant("int", "1")),
                                    BinaryOp(op="*", left=Constant("int", "2"), right=Id("x")),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            marks=pytest.mark.xfail(reason="TODO"),
        ),
        pytest.param(
            "struct v {\n"
            "    union {\n"
            "        struct { int i, j; };\n"
            "        struct { long k, l; } w;\n"
            "    };\n"
            "    int m;\n"
            "} v1;\n",
            Decl(
                "v1",
                TypeDecl(
                    "v1",
                    type=Struct(
                        "v",
                        decls=[
                            Decl(
                                None,
                                Union(
                                    None,
                                    decls=[
                                        Decl(
                                            None,
                                            Struct(
                                                None,
                                                decls=[
                                                    Decl("i", TypeDecl("i", type=IdType(["int"]))),
                                                    Decl("j", TypeDecl("j", type=IdType(["int"]))),
                                                ],
                                            ),
                                        ),
                                        Decl(
                                            "w",
                                            TypeDecl(
                                                "w",
                                                type=Struct(
                                                    None,
                                                    decls=[
                                                        Decl("k", TypeDecl("k", type=IdType(["long"]))),
                                                        Decl("l", TypeDecl("l", type=IdType(["long"]))),
                                                    ],
                                                ),
                                            ),
                                        ),
                                    ],
                                ),
                            ),
                            Decl("m", TypeDecl("m", type=IdType(["int"]))),
                        ],
                    ),
                ),
            ),
            id="ISO/IEC 9899:201x Committee Draft 2010-11-16, N1539, section 6.7.2.1, par. 19, example 1",
        ),
        pytest.param(
            "struct v {\n    int i;\n    float;\n} v2;",
            Decl(
                "v2",
                TypeDecl(
                    "v2",
                    type=Struct(
                        "v",
                        decls=[
                            Decl("i", TypeDecl("i", type=IdType(["int"]))),
                            Decl(None, IdType(["float"])),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_anonymous_struct_union(test_input: str, expected: AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.xfail(reason="TODO")
def test_struct_members_namespace():
    """Tests that structure/union member names reside in a separate namespace and can be named after existing types."""

    test_input = """
typedef int Name;
typedef Name NameArray[10];

struct {
    Name Name;
    Name NameArray[3];
} sye;

void main(void)
{
    sye.Name = 1;
}
        """

    tree = parse(test_input)

    expected2 = Decl(
        "sye",
        TypeDecl(
            "sye",
            type=Struct(
                None,
                decls=[
                    Decl("Name", TypeDecl("Name", type=IdType(["Name"]))),
                    Decl(
                        "NameArray",
                        ArrayDecl(
                            type=TypeDecl("NameArray", type=IdType(["Name"])),
                            dim=Constant("int", "3"),
                            dim_quals=[],
                        ),
                    ),
                ],
            ),
        ),
    )

    assert tree.ext[2] == expected2
    assert tree.ext[3].body.block_items[0].left.field.name == "Name"  # pyright: ignore


def test_struct_bitfields():
    # a struct with two bitfields, one unnamed
    s1 = """\
struct {
    int k:6;
    int :2;
} joe;
"""

    tree = parse(s1)
    parsed_struct = tree.ext[0]

    expected = Decl(
        "joe",
        TypeDecl(
            "joe",
            type=Struct(
                None,
                decls=[
                    Decl("k", TypeDecl("k", type=IdType(["int"])), bitsize=Constant("int", "6")),
                    Decl(None, TypeDecl(type=IdType(["int"])), bitsize=Constant("int", "2")),
                ],
            ),
        ),
    )

    assert parsed_struct == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("struct foo { };", Decl(None, Struct("foo", decls=[]))),
        ("struct { } foo;", Decl("foo", TypeDecl("foo", type=Struct(None, decls=[])))),
        ("union { } foo;", Decl("foo", TypeDecl("foo", type=Union(None, decls=[])))),
    ],
)
def test_struct_empty(test_input: str, expected: AST):
    """Tests that parsing an empty struct works.

    Empty structs do NOT follow C99 (See 6.2.5-20 of the C99 standard).
    This is nevertheless supported by some compilers (clang, gcc), especially when using FORTIFY code.
    Some compilers (visual) will fail to compile with an error.
    """

    tree = parse(test_input)
    empty_struct = tree.ext[0]
    assert empty_struct == expected


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        (
            (
                "typedef int tagEntry;\n"
                "\n"
                "struct tagEntry\n"
                "{\n"
                "    char* key;\n"
                "    char* value;\n"
                "} Entry;"
            ),
            1,
            Decl(
                "Entry",
                TypeDecl(
                    "Entry",
                    type=Struct(
                        "tagEntry",
                        decls=[
                            Decl("key", PtrDecl(quals=[], type=TypeDecl("key", type=IdType(["char"])))),
                            Decl("value", PtrDecl(quals=[], type=TypeDecl("value", type=IdType(["char"])))),
                        ],
                    ),
                ),
            ),
        ),
        (
            "struct tagEntry;\n"
            "\n"
            "typedef struct tagEntry tagEntry;\n"
            "\n"
            "struct tagEntry\n"
            "{\n"
            "    char* key;\n"
            "    char* value;\n"
            "} Entry;",
            2,
            Decl(
                "Entry",
                TypeDecl(
                    "Entry",
                    type=Struct(
                        "tagEntry",
                        decls=[
                            Decl("key", PtrDecl(quals=[], type=TypeDecl("key", type=IdType(["char"])))),
                            Decl("value", PtrDecl(quals=[], type=TypeDecl("value", type=IdType(["char"])))),
                        ],
                    ),
                ),
            ),
        ),
        (
            "typedef int mytag;\n\nenum mytag {ABC, CDE};\nenum mytag joe;\n",
            1,
            Decl(None, type=Enum("mytag", EnumeratorList([Enumerator("ABC", None), Enumerator("CDE", None)]))),
        ),
    ],
)
def test_tags_namespace(test_input: str, index: TUnion[int, slice], expected: TUnion[AST, list[AST]]):
    """Tests that the tags of structs/unions/enums reside in a separate namespace and
    can be named after existing types.
    """

    tree = parse(test_input)
    assert compare(tree.ext[index], expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a, b;",
            File(
                [
                    Decl("a", TypeDecl("a", type=IdType(["int"]))),
                    Decl("b", TypeDecl("b", type=IdType(["int"]))),
                ]
            ),
        ),
        (
            "char* p, notp, ar[4];",
            File(
                [
                    Decl("p", PtrDecl(quals=[], type=TypeDecl("p", type=IdType(["char"])))),
                    Decl("notp", TypeDecl("notp", type=IdType(["char"]))),
                    Decl(
                        "ar",
                        ArrayDecl(type=TypeDecl("ar", type=IdType(["char"])), dim=Constant("int", "4"), dim_quals=[]),
                    ),
                ]
            ),
        ),
    ],
)
def test_multi_decls(test_input: str, expected: AST):
    tree = parse(test_input)
    assert tree == expected


@pytest.mark.parametrize("test_input", ["int enum {ab, cd} fubr;", "enum kid char brbr;"])
def test_invalid_multiple_types_error(test_input: str):
    with pytest.raises(CParsingError):
        parse(test_input)


def test_invalid_typedef_storage_qual_error():
    """Tests that using typedef as a storage qualifier is correctly flagged as an error."""

    test_input = "typedef const int foo(int a) { return 0; }"
    with pytest.raises(CParsingError):
        parse(test_input)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "typedef int numbertype;\ntypedef int numbertype;",
            [
                Typedef("numbertype", quals=[], storage=["typedef"], type=TypeDecl("numbertype", type=IdType(["int"]))),
                Typedef("numbertype", quals=[], storage=["typedef"], type=TypeDecl("numbertype", type=IdType(["int"]))),
            ],
        ),
        (
            "typedef int (*funcptr)(int x);\ntypedef int (*funcptr)(int x);",
            [
                Typedef(
                    "funcptr",
                    quals=[],
                    storage=["typedef"],
                    type=PtrDecl(
                        quals=[],
                        type=FuncDecl(
                            args=ParamList([Decl("x", TypeDecl("x", type=IdType(["int"])))]),
                            type=TypeDecl("funcptr", type=IdType(["int"])),
                        ),
                    ),
                ),
                Typedef(
                    "funcptr",
                    quals=[],
                    storage=["typedef"],
                    type=PtrDecl(
                        quals=[],
                        type=FuncDecl(
                            args=ParamList([Decl("x", TypeDecl("x", type=IdType(["int"])))]),
                            type=TypeDecl("funcptr", type=IdType(["int"])),
                        ),
                    ),
                ),
            ],
        ),
        (
            "typedef int numberarray[5];\ntypedef int numberarray[5];",
            [
                Typedef(
                    "numberarray",
                    quals=[],
                    storage=["typedef"],
                    type=ArrayDecl(
                        type=TypeDecl("numberarray", type=IdType(["int"])),
                        dim=Constant("int", "5"),
                        dim_quals=[],
                    ),
                ),
                Typedef(
                    "numberarray",
                    quals=[],
                    storage=["typedef"],
                    type=ArrayDecl(
                        type=TypeDecl("numberarray", type=IdType(["int"])),
                        dim=Constant("int", "5"),
                        dim_quals=[],
                    ),
                ),
            ],
        ),
    ],
)
def test_duplicate_typedef(test_input: str, expected: list[AST]):
    """Tests that redeclarations of existing types are parsed correctly. This is non-standard, but allowed by many
    compilers.
    """

    tree = parse(test_input)
    assert compare(tree.ext, expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = 16;",
            Decl("a", TypeDecl("a", type=IdType(["int"])), init=Constant("int", "16")),
        ),
        (
            "float f = 0xEF.56p1;",
            Decl("f", TypeDecl("f", type=IdType(["float"])), init=Constant("float", "0xEF.56p1")),
        ),
        (
            "int bitmask = 0b1001010;",
            Decl("bitmask", TypeDecl("bitmask", type=IdType(["int"])), init=Constant("int", "0b1001010")),
        ),
        (
            "long ar[] = {7, 8, 9};",
            Decl(
                "ar",
                ArrayDecl(TypeDecl("ar", type=IdType(["long"])), dim=None, dim_quals=[]),
                init=InitList([Constant("int", "7"), Constant("int", "8"), Constant("int", "9")]),
            ),
        ),
        (
            "long ar[4] = {};",
            Decl(
                "ar",
                ArrayDecl(TypeDecl("ar", type=IdType(["long"])), dim=Constant("int", "4"), dim_quals=[]),
                init=InitList([]),
            ),
        ),
        (
            "char p = j;",
            Decl("p", TypeDecl("p", type=IdType(["char"])), init=Id("j")),
        ),
        (
            "char x = 'c', *p = {0, 1, 2, {4, 5}, 6};",
            [
                Decl("x", TypeDecl("x", type=IdType(["char"])), init=Constant("char", "'c'")),
                Decl(
                    "p",
                    PtrDecl(quals=[], type=TypeDecl("p", type=IdType(["char"]))),
                    init=InitList(
                        [
                            Constant("int", "0"),
                            Constant("int", "1"),
                            Constant("int", "2"),
                            InitList([Constant("int", "4"), Constant("int", "5")]),
                            Constant("int", "6"),
                        ]
                    ),
                ),
            ],
        ),
        (
            "float d = 1.0;",
            Decl("d", TypeDecl("d", type=IdType(["float"])), init=Constant("double", "1.0")),
        ),
        (
            "float ld = 1.0l;",
            Decl("ld", TypeDecl("ld", type=IdType(["float"])), init=Constant("long double", "1.0l")),
        ),
        (
            "float ld = 1.0L;",
            Decl("ld", TypeDecl("ld", type=IdType(["float"])), init=Constant("long double", "1.0L")),
        ),
        (
            "float ld = 1.0f;",
            Decl("ld", TypeDecl("ld", type=IdType(["float"])), init=Constant("float", "1.0f")),
        ),
        (
            "float ld = 1.0F;",
            Decl("ld", TypeDecl("ld", type=IdType(["float"])), init=Constant("float", "1.0F")),
        ),
        (
            "float ld = 0xDE.38p0;",
            Decl("ld", TypeDecl("ld", type=IdType(["float"])), init=Constant("float", "0xDE.38p0")),
        ),
        (
            "int i = 1;",
            Decl("i", TypeDecl("i", type=IdType(["int"])), init=Constant("int", "1")),
        ),
        (
            "long int li = 1l;",
            Decl("li", TypeDecl("li", type=IdType(["long", "int"])), init=Constant("long int", "1l")),
        ),
        (
            "unsigned int ui = 1u;",
            Decl("ui", TypeDecl("ui", type=IdType(["unsigned", "int"])), init=Constant("unsigned int", "1u")),
        ),
        (
            "unsigned long long int ulli = 1LLU;",
            Decl(
                "ulli",
                TypeDecl("ulli", type=IdType(["unsigned", "long", "long", "int"])),
                init=Constant("unsigned long long int", "1LLU"),
            ),
        ),
    ],
)
def test_decl_inits(test_input: str, expected: AST):
    tree = parse(test_input)

    if isinstance(expected, list):
        assert compare(tree.ext, expected)
    else:
        assert tree.ext[0] == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = {.k = 16};",
            InitList([NamedInitializer(name=[Id("k")], expr=Constant("int", "16"))]),
        ),
        (
            "int a = { [0].a = {1}, [1].a[0] = 2 };",
            InitList(
                [
                    NamedInitializer(
                        name=[Constant("int", "0"), Id("a")],
                        expr=InitList([Constant("int", "1")]),
                    ),
                    NamedInitializer(
                        name=[Constant("int", "1"), Id("a"), Constant("int", "0")],
                        expr=Constant("int", "2"),
                    ),
                ]
            ),
        ),
        (
            "int a = { .a = 1, .c = 3, 4, .b = 5};",
            InitList(
                [
                    NamedInitializer(name=[Id("a")], expr=Constant("int", "1")),
                    NamedInitializer(name=[Id("c")], expr=Constant("int", "3")),
                    Constant("int", "4"),
                    NamedInitializer(name=[Id("b")], expr=Constant("int", "5")),
                ]
            ),
        ),
    ],
)
def test_decl_named_inits(test_input: str, expected: AST):
    tree = parse(test_input)
    assert isinstance(tree.ext[0], Decl)

    init = tree.ext[0].init
    assert isinstance(init, InitList)
    assert init == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            "int factorial(int p)\n{\n    return 3;\n}",
            FuncDef(
                decl=Decl(
                    "factorial",
                    FuncDecl(
                        args=ParamList([Decl("p", TypeDecl("p", type=IdType(["int"])))]),
                        type=TypeDecl("factorial", type=IdType(["int"])),
                    ),
                ),
                param_decls=None,
                body=Compound([Return(expr=Constant("int", "3"))]),
            ),
            marks=pytest.mark.xfail(reason="TODO"),
        ),
        pytest.param(
            "char* zzz(int p, char* c)\n"
            "{\n"
            "    int a;\n"
            "    char b;\n"
            "\n"
            "    a = b + 2;\n"
            "    return 3;\n"
            "}",
            FuncDef(
                decl=Decl(
                    "zzz",
                    FuncDecl(
                        args=ParamList(
                            [
                                Decl("p", TypeDecl("p", type=IdType(["int"]))),
                                Decl("c", PtrDecl(quals=[], type=TypeDecl("c", type=IdType(["char"])))),
                            ]
                        ),
                        type=PtrDecl(quals=[], type=TypeDecl("zzz", type=IdType(["char"]))),
                    ),
                ),
                param_decls=None,
                body=Compound(
                    [
                        Decl("a", TypeDecl("a", type=IdType(["int"]))),
                        Decl("b", TypeDecl("b", type=IdType(["char"]))),
                        Assignment(
                            op="=",
                            left=Id("a"),
                            right=BinaryOp(op="+", left=Id("b"), right=Constant("int", "2")),
                        ),
                        Return(expr=Constant("int", "3")),
                    ]
                ),
            ),
            marks=pytest.mark.xfail(reason="TODO"),
        ),
        pytest.param(
            "char* zzz(p, c)\n"
            "long p, *c;\n"
            "{\n"
            "    int a;\n"
            "    char b;\n"
            "\n"
            "    a = b + 2;\n"
            "    return 3;\n"
            "}",
            FuncDef(
                decl=Decl(
                    "zzz",
                    FuncDecl(
                        args=ParamList([Id("p"), Id("c")]),
                        type=PtrDecl(quals=[], type=TypeDecl("zzz", type=IdType(["char"]))),
                    ),
                ),
                param_decls=[
                    Decl("p", TypeDecl("p", type=IdType(["long"]))),
                    Decl("c", PtrDecl(quals=[], type=TypeDecl("c", type=IdType(["long"])))),
                ],
                body=Compound(
                    [
                        Decl("a", TypeDecl("a", type=IdType(["int"]))),
                        Decl("b", TypeDecl("b", type=IdType(["char"]))),
                        Assignment(
                            op="=",
                            left=Id("a"),
                            right=BinaryOp(op="+", left=Id("b"), right=Constant("int", "2")),
                        ),
                        Return(expr=Constant("int", "3")),
                    ]
                ),
            ),
            marks=pytest.mark.xfail(reason="TODO"),
        ),
        pytest.param(
            "que(p)\n{\n    return 3;\n}",
            FuncDef(
                decl=Decl("que", FuncDecl(args=ParamList([Id("p")]), type=TypeDecl("que", type=IdType(["int"])))),
                param_decls=None,
                body=Compound([Return(expr=Constant("int", "3"))]),
            ),
            id="function return values and parameters may not have type information",
        ),
    ],
)
def test_function_definitions(test_input: str, expected: AST):
    tree = parse(test_input)
    assert tree.ext[0] == expected


@pytest.mark.xfail(reason="TODO")
def test_static_assert():
    test_input = """\
_Static_assert(1, "123");
int factorial(int p)
{
    _Static_assert(2, "456");
    _Static_assert(3);
}
"""

    tree = parse(test_input)

    expected_assert_1 = StaticAssert(cond=Constant("int", "1"), message=Constant("string", '"123"'))
    assert tree.ext[0] == expected_assert_1

    expected_assert_2 = StaticAssert(cond=Constant("int", "2"), message=Constant("string", '"456"'))
    assert tree.ext[1].body.block_items[0] == expected_assert_2  # pyright: ignore

    expected_assert_3 = StaticAssert(cond=Constant("int", "3"), message=None)
    assert tree.ext[1].body.block_items[2] == expected_assert_3  # pyright: ignore


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            'char* s = "hello";',
            Constant("string", '"hello"'),
            id="simple string, for reference",
        ),
        (
            'char* s = "hello" " world";',
            Constant("string", '"hello world"'),
        ),
        (
            'char* s = "" "foobar";',
            Constant("string", '"foobar"'),
        ),
        (
            r'char* s = "foo\"" "bar";',
            Constant("string", r'"foo\"bar"'),
        ),
    ],
)
def test_unified_string_literals(test_input: str, expected: AST):
    tree = parse(test_input)
    assert isinstance(tree.ext[0], Decl)

    assert tree.ext[0].init == expected


@pytest.mark.xfail(reason="Not unsupported yet. See pycparser issue 392.")
def test_escapes_in_unified_string_literals():
    # This is not correct based on the the C spec, but testing it here to see the behavior in action.
    # Will have to fix this for https://github.com/eliben/pycparser/issues/392
    #
    # The spec says in section 6.4.5 that "escape sequences are converted
    # into single members of the execution character set just prior to
    # adjacent string literal concatenation".

    test_input = r'char* s = "\1" "23";'

    with pytest.raises(CParsingError):
        tree = parse(test_input)

    expected = Constant("string", r'"\123"')
    assert tree.ext[0].init == expected  # pyright: ignore


@pytest.mark.xfail(reason="TODO")
def test_unified_string_literals_issue_6():
    test_input = r"""
int main() {
    fprintf(stderr,
    "Wrong Params?\n"
    "Usage:\n"
    "%s <binary_file_path>\n",
    argv[0]
    );
}
    """

    tree = parse(test_input)

    assert tree.ext[0].body.block_items[0].args.exprs[1].value == r'"Wrong Params?\nUsage:\n%s <binary_file_path>\n"'  # pyright: ignore


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ('char* s = L"hello" L"world";', Constant("string", 'L"helloworld"')),
        ('char* s = L"hello " L"world" L" and I";', Constant("string", 'L"hello world and I"')),
    ],
)
def test_unified_wstring_literals(test_input: str, expected: AST):
    tree = parse(test_input)

    assert tree.ext[0].init == expected  # pyright: ignore


@pytest.mark.xfail(reason="TODO")
def test_inline_specifier():
    test_input = "static inline void inlinefoo(void);"
    tree = parse(test_input)

    assert tree.ext[0].funcspec == ["inline"]  # pyright: ignore


def test_noreturn_specifier():
    test_input = "static _Noreturn void noreturnfoo(void);"
    tree = parse(test_input)

    assert tree.ext[0].funcspec == ["_Noreturn"]  # pyright: ignore


@pytest.mark.xfail(reason="TODO")
def test_variable_length_array():
    test_input = r"""
int main() {
    int size;
    int var[size = 5];

    int var2[*];
}
"""

    tree = parse(test_input)

    expected_dim_1 = Assignment(op="=", left=Id("size"), right=Constant("int", "5"))
    assert tree.ext[0].body.block_items[1].type.dim == expected_dim_1  # pyright: ignore

    expected_dim_2 = Id("*")
    assert tree.ext[0].body.block_items[2].type.dim == expected_dim_2  # pyright: ignore


@pytest.mark.xfail(reason="TODO")
def test_pragma():
    test_input = r"""
#pragma bar
void main() {
    #pragma foo
    for(;;) {}
    #pragma baz
    {
        int i = 0;
    }
    #pragma
}
struct s {
#pragma baz
} s;
_Pragma("other \"string\"")
"""

    tree = parse(test_input)

    pragma1 = tree.ext[0]
    assert pragma1 == Pragma("bar")
    assert pragma1.coord.line_start == 2  # pyright: ignore

    pragma2 = tree.ext[1].body.block_items[0]  # pyright: ignore
    assert pragma2 == Pragma("foo")
    assert pragma2.coord.line_start == 4  # pyright: ignore

    pragma3 = tree.ext[1].body.block_items[2]  # pyright: ignore
    assert pragma3 == Pragma("baz")
    assert pragma3.coord.line_start == 6  # pyright: ignore

    pragma4 = tree.ext[1].body.block_items[4]  # pyright: ignore
    assert pragma4 == Pragma("")
    assert pragma4.coord.line_start == 10  # pyright: ignore

    pragma5 = tree.ext[2].body.block_items[0]  # pyright: ignore
    assert pragma5 == Pragma("baz")
    assert pragma5.coord.line_start == 13  # pyright: ignore

    pragma6 = tree.ext[3]
    assert pragma6 == Pragma(r'"other \"string\""')
    assert pragma6.coord.line_start == 15  # pyright: ignore


@pytest.mark.xfail(reason="TODO")
def test_pragmacomp_or_statement():
    test_input = r"""
void main() {
    int sum = 0;
    for (int i; i < 3; i++)
        #pragma omp critical
        sum += 1;

    while(sum < 10)
        #pragma omp critical
        sum += 1;

    mylabel:
        #pragma foo
        sum += 10;

    if (sum > 10)
        #pragma bar
        #pragma baz
        sum = 10;

    switch (sum)
    case 10:
        #pragma foo
        sum = 20;
}
"""
    tree = parse(test_input)

    expected_list = [
        Decl("sum", TypeDecl("sum", type=IdType(["int"])), init=Constant("int", "0")),
        For(
            init=DeclList([Decl("i", TypeDecl("i", type=IdType(["int"])))]),
            cond=BinaryOp(op="<", left=Id("i"), right=Constant("int", "3")),
            next=UnaryOp(op="p++", expr=Id("i")),
            stmt=Compound([Pragma("omp critical"), Assignment(op="+=", left=Id("sum"), right=Constant("int", "1"))]),
        ),
        While(
            cond=BinaryOp(op="<", left=Id("sum"), right=Constant("int", "10")),
            stmt=Compound([Pragma("omp critical"), Assignment(op="+=", left=Id("sum"), right=Constant("int", "1"))]),
        ),
        Label(
            "mylabel",
            stmt=Compound([Pragma("foo"), Assignment(op="+=", left=Id("sum"), right=Constant("int", "10"))]),
        ),
        If(
            cond=BinaryOp(op=">", left=Id("sum"), right=Constant("int", "10")),
            iftrue=Compound(
                [Pragma("bar"), Pragma("baz"), Assignment(op="=", left=Id("sum"), right=Constant("int", "10"))]
            ),
            iffalse=None,
        ),
        Switch(
            cond=Id("sum"),
            stmt=Case(
                expr=Constant("int", "10"),
                stmts=[
                    Compound([Pragma(string="foo"), Assignment(op="=", left=Id("sum"), right=Constant("int", "20"))])
                ],
            ),
        ),
    ]

    assert compare(tree.ext[0].body.block_items, expected_list)  # pyright: ignore
