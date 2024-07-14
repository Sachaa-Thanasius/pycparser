"""Test fundamentals of the parser."""

from typing import Union

import pytest
from cparsing import CParsingError, c_ast, parse
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
    assert isinstance(tree, c_ast.File)
    assert len(tree.ext) == expected_length


@pytest.mark.parametrize("test_input", ["int foo;;"])
def test_empty_toplevel_decl(test_input: str):
    tree = parse(test_input)
    assert isinstance(tree, c_ast.File)
    assert len(tree.ext) == 1

    expected_decl = c_ast.Decl("foo", c_ast.TypeDecl("foo", type=c_ast.IdType(["int"])))
    assert tree.ext[0] == expected_decl


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (";", c_ast.File([])),
        (";int foo;", c_ast.File([c_ast.Decl("foo", c_ast.TypeDecl("foo", type=c_ast.IdType(["int"])))])),
    ],
)
def test_initial_semi(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    assert tree == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a;",
            c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"]))),
        ),
        (
            "unsigned int a;",
            c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["unsigned", "int"]))),
        ),
        (
            "_Bool a;",
            c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["_Bool"]))),
        ),
        (
            "float _Complex fcc;",
            c_ast.Decl("fcc", c_ast.TypeDecl("fcc", type=c_ast.IdType(["float", "_Complex"]))),
        ),
        (
            "char* string;",
            c_ast.Decl("string", c_ast.PtrDecl([], type=c_ast.TypeDecl("string", type=c_ast.IdType(["char"])))),
        ),
        (
            "long ar[15];",
            c_ast.Decl(
                "ar",
                c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("ar", type=c_ast.IdType(["long"])),
                    dim=c_ast.Constant("int", "15"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "long long ar[15];",
            c_ast.Decl(
                "ar",
                c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("ar", type=c_ast.IdType(["long", "long"])),
                    dim=c_ast.Constant(type="int", value="15"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "unsigned ar[];",
            c_ast.Decl(
                "ar",
                c_ast.ArrayDecl(
                    type=c_ast.TypeDecl("ar", type=c_ast.IdType(["unsigned"])),
                    dim=None,
                    dim_quals=[],
                ),
            ),
        ),
        (
            "int strlen(char* s);",
            c_ast.Decl(
                "strlen",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Decl(
                                "s",
                                c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("s", type=c_ast.IdType(["char"]))),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("strlen", type=c_ast.IdType(["int"])),
                ),
            ),
        ),
        (
            "int strcmp(char* s1, char* s2);",
            c_ast.Decl(
                "strcmp",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Decl(
                                "s1",
                                c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("s1", type=c_ast.IdType(["char"]))),
                            ),
                            c_ast.Decl(
                                "s2",
                                c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("s2", type=c_ast.IdType(["char"]))),
                            ),
                        ]
                    ),
                    type=c_ast.TypeDecl("strcmp", type=c_ast.IdType(["int"])),
                ),
            ),
        ),
        pytest.param(
            "extern foobar(foo, bar);",
            c_ast.Decl(
                "foobar",
                c_ast.FuncDecl(
                    args=c_ast.ParamList([c_ast.Id("foo"), c_ast.Id("bar")]),
                    type=c_ast.TypeDecl("foobar", type=c_ast.IdType(["int"])),
                ),
                storage=["extern"],
            ),
            id="function return values and parameters may not have type information",
        ),
        pytest.param(
            "__int128 a;",
            c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["__int128"]))),
            id=(
                "__int128: it isn't part of the core C99 or C11 standards, but is mentioned in both documents "
                "under 'Common Extensions'."
            ),
        ),
    ],
)
def test_simple_decls(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "char** ar2D;",
            c_ast.Decl(
                "ar2D",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("ar2D", type=c_ast.IdType(["char"]))),
                ),
            ),
        ),
        (
            "int (*a)[1][2];",
            c_ast.Decl(
                "a",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.ArrayDecl(
                        type=c_ast.ArrayDecl(
                            type=c_ast.TypeDecl("a", type=c_ast.IdType(["int"])),
                            dim=c_ast.Constant("int", "2"),
                            dim_quals=[],
                        ),
                        dim=c_ast.Constant("int", "1"),
                        dim_quals=[],
                    ),
                ),
            ),
        ),
        (
            "int *a[1][2];",
            c_ast.Decl(
                "a",
                c_ast.ArrayDecl(
                    type=c_ast.ArrayDecl(
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("a", type=c_ast.IdType(["int"]))),
                        dim=c_ast.Constant("int", "2"),
                        dim_quals=[],
                    ),
                    dim=c_ast.Constant("int", "1"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char* const* p;",
            c_ast.Decl(
                "p",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.PtrDecl(quals=["const"], type=c_ast.TypeDecl("p", type=c_ast.IdType(["char"]))),
                ),
            ),
        ),
        (
            "const char* const* p;",
            c_ast.Decl(
                "p",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.PtrDecl(
                        quals=["const"],
                        type=c_ast.TypeDecl("p", quals=["const"], type=c_ast.IdType(["char"])),
                    ),
                ),
                quals=["const"],
            ),
        ),
        (
            "char* * const p;",
            c_ast.Decl(
                "p",
                c_ast.PtrDecl(
                    quals=["const"],
                    type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("p", type=c_ast.IdType(["char"]))),
                ),
            ),
        ),
        (
            "char ***ar3D[40];",
            c_ast.Decl(
                "ar3D",
                c_ast.ArrayDecl(
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.PtrDecl(
                            quals=[],
                            type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("ar3D", type=c_ast.IdType(["char"]))),
                        ),
                    ),
                    dim=c_ast.Constant("int", "40"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char (***ar3D)[40];",
            c_ast.Decl(
                "ar3D",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.PtrDecl(
                            quals=[],
                            type=c_ast.ArrayDecl(
                                type=c_ast.TypeDecl("ar3D", type=c_ast.IdType(["char"])),
                                dim=c_ast.Constant("int", "40"),
                                dim_quals=[],
                            ),
                        ),
                    ),
                ),
            ),
        ),
        (
            "int (*const*const x)(char, int);",
            c_ast.Decl(
                "x",
                c_ast.PtrDecl(
                    quals=["const"],
                    type=c_ast.PtrDecl(
                        quals=["const"],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList(
                                [
                                    c_ast.Typename(
                                        None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(type=c_ast.IdType(["char"])),
                                    ),
                                    c_ast.Typename(
                                        None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                                    ),
                                ]
                            ),
                            type=c_ast.TypeDecl("x", type=c_ast.IdType(["int"])),
                        ),
                    ),
                ),
            ),
        ),
        (
            "int (*x[4])(char, int);",
            c_ast.Decl(
                "x",
                c_ast.ArrayDecl(
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList(
                                [
                                    c_ast.Typename(
                                        None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(type=c_ast.IdType(["char"])),
                                    ),
                                    c_ast.Typename(
                                        None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                                    ),
                                ]
                            ),
                            type=c_ast.TypeDecl("x", type=c_ast.IdType(["int"])),
                        ),
                    ),
                    dim=c_ast.Constant("int", "4"),
                    dim_quals=[],
                ),
            ),
        ),
        (
            "char *(*(**foo [][8])())[];",
            c_ast.Decl(
                "foo",
                c_ast.ArrayDecl(
                    type=c_ast.ArrayDecl(
                        type=c_ast.PtrDecl(
                            quals=[],
                            type=c_ast.PtrDecl(
                                quals=[],
                                type=c_ast.FuncDecl(
                                    args=None,
                                    type=c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.ArrayDecl(
                                            type=c_ast.PtrDecl(
                                                quals=[],
                                                type=c_ast.TypeDecl("foo", type=c_ast.IdType(["char"])),
                                            ),
                                            dim=None,
                                            dim_quals=[],
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        dim=c_ast.Constant("int", "8"),
                        dim_quals=[],
                    ),
                    dim=None,
                    dim_quals=[],
                ),
            ),
        ),
        pytest.param(
            "int (*k)(int);",
            c_ast.Decl(
                "k",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Typename(
                                    None,
                                    quals=[],
                                    align=None,
                                    type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", type=c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="unnamed function pointer parameters w/o quals",
        ),
        pytest.param(
            "int (*k)(const int);",
            c_ast.Decl(
                "k",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Typename(
                                    None,
                                    quals=["const"],
                                    align=None,
                                    type=c_ast.TypeDecl(quals=["const"], type=c_ast.IdType(["int"])),
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", type=c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="unnamed function pointer parameters w/ quals",
        ),
        pytest.param(
            "int (*k)(int q);",
            c_ast.Decl(
                "k",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList([c_ast.Decl("q", c_ast.TypeDecl("q", type=c_ast.IdType(["int"])))]),
                        type=c_ast.TypeDecl("k", type=c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/o quals",
        ),
        pytest.param(
            "int (*k)(const volatile int q);",
            c_ast.Decl(
                "k",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Decl(
                                    "q",
                                    c_ast.TypeDecl("q", ["const", "volatile"], type=c_ast.IdType(["int"])),
                                    quals=["const", "volatile"],
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", [], type=c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 1",
        ),
        pytest.param(
            "int (*k)(_Atomic volatile int q);",
            c_ast.Decl(
                "k",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Decl(
                                    "q",
                                    c_ast.TypeDecl("q", ["_Atomic", "volatile"], type=c_ast.IdType(["int"])),
                                    quals=["_Atomic", "volatile"],
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", type=c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 2",
        ),
        pytest.param(
            "int (*k)(const volatile int* q);",
            c_ast.Decl(
                "k",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Decl(
                                    "q",
                                    c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.TypeDecl("q", ["const", "volatile"], type=c_ast.IdType(["int"])),
                                    ),
                                    quals=["const", "volatile"],
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", type=c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="named function pointer parameters w/ quals 3",
        ),
        pytest.param(
            "int (*k)(restrict int* q);",
            c_ast.Decl(
                "k",
                c_ast.PtrDecl(
                    quals=[],
                    type=c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Decl(
                                    "q",
                                    c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.TypeDecl("q", ["restrict"], type=c_ast.IdType(["int"])),
                                    ),
                                    quals=["restrict"],
                                )
                            ]
                        ),
                        type=c_ast.TypeDecl("k", type=c_ast.IdType(["int"])),
                    ),
                ),
            ),
            id="restrict qualifier",
        ),
    ],
)
def test_nested_decls(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    decl = tree.ext[0]
    assert decl == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            "int zz(int p[static 10]);",
            c_ast.Decl(
                "zz",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Decl(
                                "p",
                                c_ast.ArrayDecl(
                                    type=c_ast.TypeDecl("p", type=c_ast.IdType(["int"])),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["static"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", type=c_ast.IdType(["int"])),
                ),
            ),
            id="named function parameter 1",
        ),
        pytest.param(
            "int zz(int p[const 10]);",
            c_ast.Decl(
                "zz",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Decl(
                                "p",
                                c_ast.ArrayDecl(
                                    type=c_ast.TypeDecl("p", type=c_ast.IdType(["int"])),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["const"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", type=c_ast.IdType(["int"])),
                ),
            ),
            id="named function parameter 2",
        ),
        pytest.param(
            "int zz(int p[restrict][5]);",
            c_ast.Decl(
                "zz",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Decl(
                                "p",
                                c_ast.ArrayDecl(
                                    type=c_ast.ArrayDecl(
                                        type=c_ast.TypeDecl("p", type=c_ast.IdType(["int"])),
                                        dim=c_ast.Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=None,
                                    dim_quals=["restrict"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", type=c_ast.IdType(["int"])),
                ),
            ),
            id="named function parameter 3",
        ),
        pytest.param(
            "int zz(int p[const restrict static 10][5]);",
            c_ast.Decl(
                "zz",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Decl(
                                "p",
                                c_ast.ArrayDecl(
                                    type=c_ast.ArrayDecl(
                                        type=c_ast.TypeDecl("p", type=c_ast.IdType(["int"])),
                                        dim=c_ast.Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["const", "restrict", "static"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", type=c_ast.IdType(["int"])),
                ),
            ),
            id="named function parameter 4",
        ),
        pytest.param(
            "int zz(int [const 10]);",
            c_ast.Decl(
                "zz",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.ArrayDecl(
                                    type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["const"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", type=c_ast.IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 1",
        ),
        pytest.param(
            "int zz(int [restrict][5]);",
            c_ast.Decl(
                "zz",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.ArrayDecl(
                                    type=c_ast.ArrayDecl(
                                        type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                                        dim=c_ast.Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=None,
                                    dim_quals=["restrict"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", type=c_ast.IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 2",
        ),
        pytest.param(
            "int zz(int [const restrict volatile 10][5]);",
            c_ast.Decl(
                "zz",
                c_ast.FuncDecl(
                    args=c_ast.ParamList(
                        [
                            c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.ArrayDecl(
                                    type=c_ast.ArrayDecl(
                                        type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                                        dim=c_ast.Constant("int", "5"),
                                        dim_quals=[],
                                    ),
                                    dim=c_ast.Constant("int", "10"),
                                    dim_quals=["const", "restrict", "volatile"],
                                ),
                            )
                        ]
                    ),
                    type=c_ast.TypeDecl("zz", type=c_ast.IdType(["int"])),
                ),
            ),
            id="unnamed function parameter 3",
        ),
    ],
)
def test_func_decls_with_array_dim_qualifiers(test_input: str, expected: c_ast.AST):
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

    assert isinstance(tree, c_ast.Decl)
    assert tree.quals == expected_quals
    assert tree.storage == expected_storage


def test_qualifiers_storage_specifiers_2():
    test_input = "static char * const p;"
    tree = parse(test_input)

    assert isinstance(tree.ext[0], c_ast.Decl)

    pdecl = tree.ext[0].type
    assert isinstance(pdecl, c_ast.PtrDecl)
    assert pdecl.quals == ["const"]


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        (
            "_Atomic(int) ai;",
            0,
            c_ast.Decl("ai", c_ast.TypeDecl("ai", ["_Atomic"], type=c_ast.IdType(["int"])), quals=["_Atomic"]),
        ),
        (
            "_Atomic(int*) ai;",
            0,
            c_ast.Decl("ai", c_ast.PtrDecl(quals=["_Atomic"], type=c_ast.TypeDecl("ai", type=c_ast.IdType(["int"])))),
        ),
        (
            "_Atomic(_Atomic(int)*) aai;",
            0,
            c_ast.Decl(
                "aai",
                c_ast.PtrDecl(
                    quals=["_Atomic"],
                    type=c_ast.TypeDecl("aai", ["_Atomic"], type=c_ast.IdType(["int"])),
                ),
                quals=["_Atomic"],
            ),
        ),
        pytest.param(
            "_Atomic(int) foo, bar;",
            slice(0, 2),
            [
                c_ast.Decl("foo", c_ast.TypeDecl("foo", ["_Atomic"], type=c_ast.IdType(["int"])), quals=["_Atomic"]),
                c_ast.Decl("bar", c_ast.TypeDecl("foo", ["_Atomic"], type=c_ast.IdType(["int"])), quals=["_Atomic"]),
            ],
            id="multiple declarations",
        ),
        pytest.param(
            "typedef _Atomic(int) atomic_int;",
            0,
            c_ast.Typedef(
                "atomic_int",
                quals=["_Atomic"],
                storage=["typedef"],
                type=c_ast.TypeDecl("atomic_int", ["_Atomic"], type=c_ast.IdType(["int"])),
            ),
            id="typedefs with _Atomic specifiers 1",
        ),
        pytest.param(
            "typedef _Atomic(_Atomic(_Atomic(int (*)(void)) *) *) t;",
            0,
            c_ast.Typedef(
                "t",
                quals=[],
                storage=["typedef"],
                type=c_ast.PtrDecl(
                    quals=["_Atomic"],
                    type=c_ast.PtrDecl(
                        quals=["_Atomic"],
                        type=c_ast.PtrDecl(
                            quals=["_Atomic"],
                            type=c_ast.FuncDecl(
                                args=c_ast.ParamList(
                                    [
                                        c_ast.Typename(
                                            None,
                                            quals=[],
                                            align=None,
                                            type=c_ast.TypeDecl(type=c_ast.IdType(["void"])),
                                        )
                                    ]
                                ),
                                type=c_ast.TypeDecl("t", type=c_ast.IdType(["int"])),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_atomic_specifier(test_input: str, index: Union[int, slice], expected: Union[c_ast.AST, list[c_ast.AST]]):
    tree = parse(test_input)
    decl = tree.ext[index]
    assert c_ast.compare(decl, expected)


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
                c_ast.UnaryOp(op="sizeof", expr=c_ast.Id("k")),
                c_ast.UnaryOp(
                    op="sizeof",
                    expr=c_ast.Typename(None, quals=[], align=None, type=c_ast.TypeDecl(type=c_ast.IdType(["int"]))),
                ),
                c_ast.UnaryOp(
                    op="sizeof",
                    expr=c_ast.Typename(
                        None,
                        quals=[],
                        align=None,
                        type=c_ast.PtrDecl(
                            quals=[],
                            type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl(type=c_ast.IdType(["int"]))),
                        ),
                    ),
                ),
            ),
        )
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_sizeof(test_input: str, expected_compound_block_items: tuple[c_ast.AST, ...]) -> None:
    tree = parse(test_input)
    compound = tree.ext[0].body  # type: ignore
    assert isinstance(compound, c_ast.Compound)
    assert compound.block_items

    for index, expected in enumerate(expected_compound_block_items):
        found_init = compound.block_items[index].init  # type: ignore
        assert isinstance(found_init, c_ast.UnaryOp)
        assert found_init == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = _Alignof(int);",
            c_ast.Decl(
                "a",
                c_ast.TypeDecl("a", type=c_ast.IdType(["int"])),
                init=c_ast.UnaryOp(
                    op="_Alignof",
                    expr=c_ast.Typename(None, quals=[], align=None, type=c_ast.TypeDecl(type=c_ast.IdType(["int"]))),
                ),
            ),
        ),
        (
            "_Alignas(_Alignof(int)) char a;",
            c_ast.Decl(
                "a",
                c_ast.TypeDecl("a", type=c_ast.IdType(["char"])),
                align=[
                    c_ast.Alignas(
                        alignment=c_ast.UnaryOp(
                            op="_Alignof",
                            expr=c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                            ),
                        )
                    )
                ],
            ),
        ),
        (
            "_Alignas(4) char a;",
            c_ast.Decl(
                "a",
                c_ast.TypeDecl("a", type=c_ast.IdType(["char"])),
                align=[c_ast.Alignas(c_ast.Constant("int", "4"))],
            ),
        ),
        (
            "_Alignas(int) char a;",
            c_ast.Decl(
                "a",
                c_ast.TypeDecl("a", type=c_ast.IdType(["char"])),
                align=[
                    c_ast.Alignas(
                        c_ast.Typename(
                            None,
                            quals=[],
                            align=None,
                            type=c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                        )
                    )
                ],
            ),
        ),
    ],
)
def test_alignof(test_input: str, expected: c_ast.AST) -> None:
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
        c_ast.Decl(
            "a",
            c_ast.TypeDecl("a", type=c_ast.IdType(names=["int"])),
            init=c_ast.FuncCall(
                c_ast.Id("offsetof"),
                args=c_ast.ExprList(
                    [
                        c_ast.Typename(
                            None,
                            quals=[],
                            align=None,
                            type=c_ast.TypeDecl(type=c_ast.Struct("S", decls=None)),
                        ),
                        c_ast.Id("p"),
                    ]
                ),
            ),
        ),
        c_ast.Assignment(
            op="=",
            left=c_ast.StructRef(c_ast.Id("a"), type=".", field=c_ast.Id("b")),
            right=c_ast.BinaryOp(
                op="+",
                left=c_ast.FuncCall(
                    c_ast.Id("offsetof"),
                    args=c_ast.ExprList(
                        [
                            c_ast.Typename(
                                None,
                                quals=[],
                                align=None,
                                type=c_ast.TypeDecl(type=c_ast.Struct("sockaddr", decls=None)),
                            ),
                            c_ast.Id("sp"),
                        ]
                    ),
                ),
                right=c_ast.FuncCall(
                    c_ast.Id("strlen"),
                    args=c_ast.ExprList([c_ast.Id("bar")]),
                ),
            ),
        ),
        c_ast.Decl(
            "a",
            c_ast.TypeDecl("a", type=c_ast.IdType(["int"])),
            init=c_ast.FuncCall(
                c_ast.Id("offsetof"),
                args=c_ast.ExprList(
                    [
                        c_ast.Typename(
                            None,
                            quals=[],
                            align=None,
                            type=c_ast.TypeDecl(type=c_ast.Struct("S", decls=None)),
                        ),
                        c_ast.StructRef(
                            c_ast.StructRef(c_ast.Id("p"), type=".", field=c_ast.Id("q")),
                            type=".",
                            field=c_ast.Id("r"),
                        ),
                    ]
                ),
            ),
        ),
        c_ast.Decl(
            "a",
            c_ast.TypeDecl("a", type=c_ast.IdType(["int"])),
            init=c_ast.FuncCall(
                c_ast.Id("offsetof"),
                args=c_ast.ExprList(
                    [
                        c_ast.Typename(
                            None,
                            quals=[],
                            align=None,
                            type=c_ast.TypeDecl(type=c_ast.Struct("S", decls=None)),
                        ),
                        c_ast.ArrayRef(
                            name=c_ast.ArrayRef(
                                name=c_ast.StructRef(
                                    name=c_ast.ArrayRef(name=c_ast.Id("p"), subscript=c_ast.Constant("int", "5")),
                                    type=".",
                                    field=c_ast.Id("q"),
                                ),
                                subscript=c_ast.Constant("int", "4"),
                            ),
                            subscript=c_ast.Constant("int", "5"),
                        ),
                    ]
                ),
            ),
        ),
    ]

    tree = parse(test_input)
    assert c_ast.compare(tree.ext[0].body.block_items, expected_list)  # type: ignore


@pytest.mark.xfail(reason="TODO")
def test_compound_statement() -> None:
    test_input = """\
void foo() {
}
"""

    tree = parse(test_input)

    compound = tree.ext[0].body  # type: ignore
    assert isinstance(compound, c_ast.Compound)
    assert compound.coord == Coord(2, 0, filename="<unknown>")


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        pytest.param(
            0,
            c_ast.CompoundLiteral(
                type=c_ast.Typename(
                    None,
                    quals=[],
                    align=None,
                    type=c_ast.TypeDecl(type=c_ast.IdType(names=["long", "long"])),
                ),
                init=c_ast.InitList([c_ast.Id("k")]),
            ),
            id="C99 compound literal feature 1",
        ),
        pytest.param(
            1,
            c_ast.CompoundLiteral(
                type=c_ast.Typename(
                    None,
                    quals=[],
                    align=None,
                    type=c_ast.TypeDecl(type=c_ast.Struct("jk", decls=None)),
                ),
                init=c_ast.InitList(
                    [
                        c_ast.NamedInitializer(
                            name=[c_ast.Id("a")],
                            expr=c_ast.InitList([c_ast.Constant("int", "1"), c_ast.Constant(type="int", value="2")]),
                        ),
                        c_ast.NamedInitializer(name=[c_ast.Id("b"), c_ast.Constant("int", "0")], expr=c_ast.Id("t")),
                    ]
                ),
            ),
            id="C99 compound literal feature 2",
        ),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_compound_literals(index: int, expected: c_ast.CompoundLiteral) -> None:
    test_input = r"""
void foo() {
    p = (long long){k};
    tc = (struct jk){.a = {1, 2}, .b[0] = t};
}"""

    tree = parse(test_input)

    compound = tree.ext[0].body.block_items[index].right  # type: ignore
    assert isinstance(compound, c_ast.CompoundLiteral)
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
        c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"]))),
        c_ast.Compound(block_items=None),
        c_ast.Compound([c_ast.Constant("int", value="1")]),
        c_ast.Compound([c_ast.Constant("int", value="1"), c_ast.Constant("int", "2")]),
        c_ast.Decl(
            "b",
            c_ast.TypeDecl("b", type=c_ast.IdType(["int"])),
            init=c_ast.Compound([c_ast.Constant("int", "1")]),
        ),
        c_ast.Decl("c", c_ast.TypeDecl("c", type=c_ast.IdType(["int"]))),
        c_ast.Decl(
            "d",
            c_ast.TypeDecl("d", type=c_ast.IdType(["int"])),
            init=c_ast.Compound(
                [
                    c_ast.Decl("x", c_ast.TypeDecl("x", type=c_ast.IdType(["int"])), init=c_ast.Constant("int", "1")),
                    c_ast.BinaryOp(op="+", left=c_ast.Id("x"), right=c_ast.Constant("int", "2")),
                ]
            ),
        ),
        c_ast.Assignment(
            op="=",
            left=c_ast.Id("a"),
            right=c_ast.Compound(
                [
                    c_ast.Decl(
                        "x",
                        c_ast.TypeDecl("x", type=c_ast.IdType(["int"])),
                        init=c_ast.Constant("int", "1"),
                    ),
                    c_ast.BinaryOp(op="*", left=c_ast.Constant("int", "2"), right=c_ast.Id("x")),
                ]
            ),
        ),
    ]

    tree = parse(test_input)
    assert c_ast.compare(tree.ext[0].body.block_items, expected)  # type: ignore


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("enum mycolor op;", c_ast.Enum("mycolor")),
        (
            "enum mysize {large=20, small, medium} shoes;",
            c_ast.Enum(
                "mysize",
                c_ast.EnumeratorList(
                    [
                        c_ast.Enumerator("large", value=c_ast.Constant("int", "20")),
                        c_ast.Enumerator("small"),
                        c_ast.Enumerator("medium"),
                    ]
                ),
            ),
        ),
        pytest.param(
            "enum\n{\n    red,\n    blue,\n    green,\n} color;",
            c_ast.Enum(
                None,
                c_ast.EnumeratorList(
                    [
                        c_ast.Enumerator("red"),
                        c_ast.Enumerator("blue"),
                        c_ast.Enumerator("green"),
                    ]
                ),
            ),
            id="enum with trailing comma (C99 feature)",
        ),
    ],
)
def test_enums(test_input: str, expected: c_ast.AST) -> None:
    tree = parse(test_input)
    enum_type = tree.ext[0].type.type  # type: ignore
    assert isinstance(enum_type, c_ast.Enum)
    assert enum_type == expected


@pytest.mark.parametrize(
    ("test_input", "index", "expected"),
    [
        pytest.param(
            "typedef void* node;\nnode k;",
            slice(2),
            [
                c_ast.Typedef(
                    "node",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("node", type=c_ast.IdType(["void"]))),
                ),
                c_ast.Decl("k", c_ast.TypeDecl("k", type=c_ast.IdType(["node"]))),
            ],
            id="with typedef",
        ),
        (
            "typedef int T;\ntypedef T *pT;\n\npT aa, bb;",
            3,
            c_ast.Decl("bb", c_ast.TypeDecl("bb", type=c_ast.IdType(["pT"]))),
        ),
        (
            "typedef char* __builtin_va_list;\ntypedef __builtin_va_list __gnuc_va_list;",
            1,
            c_ast.Typedef(
                "__gnuc_va_list",
                quals=[],
                storage=["typedef"],
                type=c_ast.TypeDecl("__gnuc_va_list", type=c_ast.IdType(["__builtin_va_list"])),
            ),
        ),
        (
            "typedef struct tagHash Hash;",
            0,
            c_ast.Typedef(
                "Hash",
                quals=[],
                storage=["typedef"],
                type=c_ast.TypeDecl("Hash", type=c_ast.Struct("tagHash", decls=None)),
            ),
        ),
        (
            "typedef int (* const * const T)(void);",
            0,
            c_ast.Typedef(
                "T",
                quals=[],
                storage=["typedef"],
                type=c_ast.PtrDecl(
                    quals=["const"],
                    type=c_ast.PtrDecl(
                        quals=["const"],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList(
                                [
                                    c_ast.Typename(
                                        None,
                                        quals=[],
                                        align=None,
                                        type=c_ast.TypeDecl(type=c_ast.IdType(["void"])),
                                    )
                                ]
                            ),
                            type=c_ast.TypeDecl("T", type=c_ast.IdType(["int"])),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_typedef(test_input: str, index: Union[int, slice], expected: Union[c_ast.AST, list[c_ast.AST]]) -> None:
    tree = parse(test_input)
    assert c_ast.compare(tree.ext[index], expected)


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
            c_ast.Decl(
                "joe",
                c_ast.TypeDecl(
                    "joe",
                    type=c_ast.Struct(
                        None,
                        decls=[
                            c_ast.Decl(
                                "id",
                                c_ast.TypeDecl("id", type=c_ast.IdType(["int"])),
                            ),
                            c_ast.Decl(
                                "name",
                                c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.TypeDecl("name", type=c_ast.IdType(["char"])),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
        (
            "struct node p;",
            0,
            c_ast.Decl("p", c_ast.TypeDecl("p", type=c_ast.Struct("node"))),
        ),
        (
            "union pri ra;",
            0,
            c_ast.Decl("ra", c_ast.TypeDecl("ra", type=c_ast.Union("pri"))),
        ),
        (
            "struct node* p;",
            0,
            c_ast.Decl(
                "p",
                c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("p", type=c_ast.Struct("node"))),
            ),
        ),
        (
            "struct node;",
            0,
            c_ast.Decl(None, type=c_ast.Struct("node")),
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
            c_ast.Decl(
                "u",
                c_ast.TypeDecl(
                    "u",
                    type=c_ast.Union(
                        None,
                        decls=[
                            c_ast.Decl(
                                "n",
                                type=c_ast.TypeDecl(
                                    "n",
                                    type=c_ast.Struct(
                                        None,
                                        decls=[c_ast.Decl("type", c_ast.TypeDecl("type", type=c_ast.IdType(["int"])))],
                                    ),
                                ),
                            ),
                            c_ast.Decl(
                                "ni",
                                c_ast.TypeDecl(
                                    "ni",
                                    type=c_ast.Struct(
                                        None,
                                        decls=[
                                            c_ast.Decl("type", c_ast.TypeDecl("type", type=c_ast.IdType(["int"]))),
                                            c_ast.Decl(
                                                "intnode",
                                                c_ast.TypeDecl("intnode", type=c_ast.IdType(["int"])),
                                            ),
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
                c_ast.Typedef(
                    "foo",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.TypeDecl(
                        "foo",
                        type=c_ast.Struct(
                            "foo_tag",
                            decls=[
                                c_ast.Decl(
                                    "data",
                                    c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.TypeDecl("data", type=c_ast.IdType(["void"])),
                                    ),
                                )
                            ],
                        ),
                    ),
                ),
                c_ast.Typedef(
                    "pfoo",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.TypeDecl(
                            "pfoo",
                            type=c_ast.Struct(
                                "foo_tag",
                                decls=[
                                    c_ast.Decl(
                                        "data",
                                        c_ast.PtrDecl(
                                            quals=[],
                                            type=c_ast.TypeDecl(
                                                "data",
                                                type=c_ast.IdType(["void"]),
                                            ),
                                        ),
                                    )
                                ],
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
            c_ast.Typedef(
                "Hash",
                quals=[],
                storage=["typedef"],
                type=c_ast.TypeDecl(
                    declname="Hash",
                    type=c_ast.Struct(
                        "tagHash",
                        decls=[
                            c_ast.Decl(
                                "table_size",
                                type=c_ast.TypeDecl("table_size", type=c_ast.IdType(["unsigned", "int"])),
                            ),
                            c_ast.Decl(
                                "heads",
                                c_ast.PtrDecl(
                                    quals=[],
                                    type=c_ast.PtrDecl(
                                        quals=[],
                                        type=c_ast.TypeDecl("heads", type=c_ast.IdType(["Node"])),
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_union(test_input: str, index: Union[int, slice], expected: Union[c_ast.AST, list[c_ast.AST]]) -> None:
    tree = parse(test_input)
    type_ = tree.ext[index]
    assert c_ast.compare(type_, expected)


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

    s7_ast = parse(test_input, filename="test.c")
    assert s7_ast.ext[0].type.decls[2].coord == Coord(6, 22, filename="test.c")  # type: ignore
    assert s7_ast.ext[0].type.decls[3].coord == Coord(r"D:\eli\cpp_stuff\libc_include/sys/reent.h", 78, 22)  # type: ignore


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct Foo {\n   enum Bar { A = 1 };\n};",
            c_ast.Decl(
                None,
                c_ast.Struct(
                    "Foo",
                    decls=[
                        c_ast.Decl(
                            None,
                            c_ast.Enum(
                                "Bar",
                                c_ast.EnumeratorList([c_ast.Enumerator("A", c_ast.Constant("int", "1"))]),
                            ),
                        )
                    ],
                ),
            ),
        ),
        (
            "struct Foo {\n    enum Bar { A = 1, B, C } bar;\n    enum Baz { D = A } baz;\n} foo;",
            c_ast.Decl(
                "foo",
                c_ast.TypeDecl(
                    "foo",
                    type=c_ast.Struct(
                        "Foo",
                        decls=[
                            c_ast.Decl(
                                "bar",
                                c_ast.TypeDecl(
                                    "bar",
                                    type=c_ast.Enum(
                                        "Bar",
                                        c_ast.EnumeratorList(
                                            [
                                                c_ast.Enumerator("A", c_ast.Constant("int", "1")),
                                                c_ast.Enumerator("B"),
                                                c_ast.Enumerator("C"),
                                            ]
                                        ),
                                    ),
                                ),
                            ),
                            c_ast.Decl(
                                "baz",
                                c_ast.TypeDecl(
                                    "baz",
                                    type=c_ast.Enum(
                                        "Baz",
                                        c_ast.EnumeratorList([c_ast.Enumerator("D", c_ast.Id("A"))]),
                                    ),
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_enum(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct {\n    int a;;\n} foo;",
            c_ast.Decl(
                "foo",
                c_ast.TypeDecl(
                    "foo",
                    type=c_ast.Struct(None, decls=[c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"])))]),
                ),
            ),
        ),
        (
            "struct {\n    int a;;;;\n    float b, c;\n    ;;\n    char d;\n} foo;",
            c_ast.Decl(
                "foo",
                type=c_ast.TypeDecl(
                    "foo",
                    type=c_ast.Struct(
                        None,
                        decls=[
                            c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"]))),
                            c_ast.Decl("b", c_ast.TypeDecl("b", type=c_ast.IdType(["float"]))),
                            c_ast.Decl("c", c_ast.TypeDecl("c", type=c_ast.IdType(["float"]))),
                            c_ast.Decl("d", c_ast.TypeDecl("d", type=c_ast.IdType(["char"]))),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_struct_with_extra_semis_inside(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    type_ = tree.ext[0]
    assert type_ == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "struct {\n    ;int a;\n} foo;",
            c_ast.Decl(
                "foo",
                c_ast.TypeDecl(
                    "foo",
                    type=c_ast.Struct(
                        None,
                        decls=[c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"])))],
                    ),
                ),
            ),
        )
    ],
)
def test_struct_with_initial_semi(test_input: str, expected: c_ast.AST):
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
            c_ast.FuncDef(
                decl=c_ast.Decl(
                    "foo",
                    c_ast.FuncDecl(args=None, type=c_ast.TypeDecl("foo", type=c_ast.IdType(["void"]))),
                ),
                param_decls=None,
                body=c_ast.Compound(
                    [
                        c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"]))),
                        c_ast.Compound(block_items=None),
                        c_ast.Compound(block_items=[c_ast.Constant("int", "1")]),
                        c_ast.Compound(block_items=[c_ast.Constant("int", "1"), c_ast.Constant("int", "2")]),
                        c_ast.Decl(
                            "b",
                            c_ast.TypeDecl("b", type=c_ast.IdType(["int"])),
                            init=c_ast.Compound([c_ast.Constant("int", "1")]),
                        ),
                        c_ast.Decl("c", c_ast.TypeDecl("c", type=c_ast.IdType(["int"]))),
                        c_ast.Decl(
                            "d",
                            c_ast.TypeDecl("d", type=c_ast.IdType(["int"])),
                            init=c_ast.Compound(
                                [
                                    c_ast.Decl(
                                        "x",
                                        c_ast.TypeDecl("x", type=c_ast.IdType(["int"])),
                                        init=c_ast.Constant("int", "1"),
                                    ),
                                    c_ast.BinaryOp(op="+", left=c_ast.Id("x"), right=c_ast.Constant("int", "2")),
                                ]
                            ),
                        ),
                        c_ast.Assignment(
                            op="=",
                            left=c_ast.Id("a"),
                            right=c_ast.Compound(
                                [
                                    c_ast.Decl(
                                        "x",
                                        c_ast.TypeDecl("x", type=c_ast.IdType(["int"])),
                                        init=c_ast.Constant("int", "1"),
                                    ),
                                    c_ast.BinaryOp(op="*", left=c_ast.Constant("int", "2"), right=c_ast.Id("x")),
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
            c_ast.Decl(
                "v1",
                c_ast.TypeDecl(
                    "v1",
                    type=c_ast.Struct(
                        "v",
                        decls=[
                            c_ast.Decl(
                                None,
                                c_ast.Union(
                                    None,
                                    decls=[
                                        c_ast.Decl(
                                            None,
                                            c_ast.Struct(
                                                None,
                                                decls=[
                                                    c_ast.Decl("i", c_ast.TypeDecl("i", type=c_ast.IdType(["int"]))),
                                                    c_ast.Decl("j", c_ast.TypeDecl("j", type=c_ast.IdType(["int"]))),
                                                ],
                                            ),
                                        ),
                                        c_ast.Decl(
                                            "w",
                                            c_ast.TypeDecl(
                                                "w",
                                                type=c_ast.Struct(
                                                    None,
                                                    decls=[
                                                        c_ast.Decl(
                                                            "k",
                                                            c_ast.TypeDecl("k", type=c_ast.IdType(["long"])),
                                                        ),
                                                        c_ast.Decl(
                                                            "l",
                                                            c_ast.TypeDecl("l", type=c_ast.IdType(["long"])),
                                                        ),
                                                    ],
                                                ),
                                            ),
                                        ),
                                    ],
                                ),
                            ),
                            c_ast.Decl("m", c_ast.TypeDecl("m", type=c_ast.IdType(["int"]))),
                        ],
                    ),
                ),
            ),
            id="ISO/IEC 9899:201x Committee Draft 2010-11-16, N1539, section 6.7.2.1, par. 19, example 1",
        ),
        pytest.param(
            "struct v {\n    int i;\n    float;\n} v2;",
            c_ast.Decl(
                "v2",
                c_ast.TypeDecl(
                    "v2",
                    type=c_ast.Struct(
                        "v",
                        decls=[
                            c_ast.Decl("i", c_ast.TypeDecl("i", type=c_ast.IdType(["int"]))),
                            c_ast.Decl(None, c_ast.IdType(["float"])),
                        ],
                    ),
                ),
            ),
        ),
    ],
)
def test_anonymous_struct_union(test_input: str, expected: c_ast.AST):
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

    expected2 = c_ast.Decl(
        "sye",
        c_ast.TypeDecl(
            "sye",
            type=c_ast.Struct(
                None,
                decls=[
                    c_ast.Decl("Name", c_ast.TypeDecl("Name", type=c_ast.IdType(["Name"]))),
                    c_ast.Decl(
                        "NameArray",
                        c_ast.ArrayDecl(
                            type=c_ast.TypeDecl("NameArray", type=c_ast.IdType(["Name"])),
                            dim=c_ast.Constant("int", "3"),
                            dim_quals=[],
                        ),
                    ),
                ],
            ),
        ),
    )

    assert tree.ext[2] == expected2
    assert tree.ext[3].body.block_items[0].left.field.name == "Name"  # type: ignore


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

    expected = c_ast.Decl(
        "joe",
        c_ast.TypeDecl(
            "joe",
            type=c_ast.Struct(
                None,
                decls=[
                    c_ast.Decl(
                        "k",
                        c_ast.TypeDecl("k", type=c_ast.IdType(["int"])),
                        bitsize=c_ast.Constant("int", "6"),
                    ),
                    c_ast.Decl(
                        None,
                        c_ast.TypeDecl(type=c_ast.IdType(["int"])),
                        bitsize=c_ast.Constant("int", "2"),
                    ),
                ],
            ),
        ),
    )

    assert parsed_struct == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("struct foo { };", c_ast.Decl(None, c_ast.Struct("foo", decls=[]))),
        ("struct { } foo;", c_ast.Decl("foo", c_ast.TypeDecl("foo", type=c_ast.Struct(None, decls=[])))),
        ("union { } foo;", c_ast.Decl("foo", c_ast.TypeDecl("foo", type=c_ast.Union(None, decls=[])))),
    ],
)
def test_struct_empty(test_input: str, expected: c_ast.AST):
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
            c_ast.Decl(
                "Entry",
                c_ast.TypeDecl(
                    "Entry",
                    type=c_ast.Struct(
                        "tagEntry",
                        decls=[
                            c_ast.Decl(
                                "key",
                                c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("key", type=c_ast.IdType(["char"]))),
                            ),
                            c_ast.Decl(
                                "value",
                                c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("value", type=c_ast.IdType(["char"]))),
                            ),
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
            c_ast.Decl(
                "Entry",
                c_ast.TypeDecl(
                    "Entry",
                    type=c_ast.Struct(
                        "tagEntry",
                        decls=[
                            c_ast.Decl(
                                "key",
                                c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("key", type=c_ast.IdType(["char"]))),
                            ),
                            c_ast.Decl(
                                "value",
                                c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("value", type=c_ast.IdType(["char"]))),
                            ),
                        ],
                    ),
                ),
            ),
        ),
        (
            "typedef int mytag;\n\nenum mytag {ABC, CDE};\nenum mytag joe;\n",
            1,
            c_ast.Decl(
                None,
                type=c_ast.Enum(
                    "mytag",
                    c_ast.EnumeratorList([c_ast.Enumerator("ABC", None), c_ast.Enumerator("CDE", None)]),
                ),
            ),
        ),
    ],
)
def test_tags_namespace(test_input: str, index: Union[int, slice], expected: Union[c_ast.AST, list[c_ast.AST]]):
    """Tests that the tags of structs/unions/enums reside in a separate namespace and
    can be named after existing types.
    """

    tree = parse(test_input)
    assert c_ast.compare(tree.ext[index], expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a, b;",
            c_ast.File(
                [
                    c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"]))),
                    c_ast.Decl("b", c_ast.TypeDecl("b", type=c_ast.IdType(["int"]))),
                ]
            ),
        ),
        (
            "char* p, notp, ar[4];",
            c_ast.File(
                [
                    c_ast.Decl("p", c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("p", type=c_ast.IdType(["char"])))),
                    c_ast.Decl("notp", c_ast.TypeDecl("notp", type=c_ast.IdType(["char"]))),
                    c_ast.Decl(
                        "ar",
                        c_ast.ArrayDecl(
                            type=c_ast.TypeDecl("ar", type=c_ast.IdType(["char"])),
                            dim=c_ast.Constant("int", "4"),
                            dim_quals=[],
                        ),
                    ),
                ]
            ),
        ),
    ],
)
def test_multi_decls(test_input: str, expected: c_ast.AST):
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
                c_ast.Typedef(
                    "numbertype",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.TypeDecl("numbertype", type=c_ast.IdType(["int"])),
                ),
                c_ast.Typedef(
                    "numbertype",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.TypeDecl("numbertype", type=c_ast.IdType(["int"])),
                ),
            ],
        ),
        (
            "typedef int (*funcptr)(int x);\ntypedef int (*funcptr)(int x);",
            [
                c_ast.Typedef(
                    "funcptr",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList([c_ast.Decl("x", c_ast.TypeDecl("x", type=c_ast.IdType(["int"])))]),
                            type=c_ast.TypeDecl("funcptr", type=c_ast.IdType(["int"])),
                        ),
                    ),
                ),
                c_ast.Typedef(
                    "funcptr",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.PtrDecl(
                        quals=[],
                        type=c_ast.FuncDecl(
                            args=c_ast.ParamList([c_ast.Decl("x", c_ast.TypeDecl("x", type=c_ast.IdType(["int"])))]),
                            type=c_ast.TypeDecl("funcptr", type=c_ast.IdType(["int"])),
                        ),
                    ),
                ),
            ],
        ),
        (
            "typedef int numberarray[5];\ntypedef int numberarray[5];",
            [
                c_ast.Typedef(
                    "numberarray",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.ArrayDecl(
                        type=c_ast.TypeDecl("numberarray", type=c_ast.IdType(["int"])),
                        dim=c_ast.Constant("int", "5"),
                        dim_quals=[],
                    ),
                ),
                c_ast.Typedef(
                    "numberarray",
                    quals=[],
                    storage=["typedef"],
                    type=c_ast.ArrayDecl(
                        type=c_ast.TypeDecl("numberarray", type=c_ast.IdType(["int"])),
                        dim=c_ast.Constant("int", "5"),
                        dim_quals=[],
                    ),
                ),
            ],
        ),
    ],
)
def test_duplicate_typedef(test_input: str, expected: list[c_ast.AST]):
    """Tests that redeclarations of existing types are parsed correctly. This is non-standard, but allowed by many
    compilers.
    """

    tree = parse(test_input)
    assert c_ast.compare(tree.ext, expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = 16;",
            c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"])), init=c_ast.Constant("int", "16")),
        ),
        (
            "float f = 0xEF.56p1;",
            c_ast.Decl(
                "f",
                c_ast.TypeDecl("f", type=c_ast.IdType(["float"])),
                init=c_ast.Constant("float", "0xEF.56p1"),
            ),
        ),
        (
            "int bitmask = 0b1001010;",
            c_ast.Decl(
                "bitmask",
                c_ast.TypeDecl("bitmask", type=c_ast.IdType(["int"])),
                init=c_ast.Constant("int", "0b1001010"),
            ),
        ),
        (
            "long ar[] = {7, 8, 9};",
            c_ast.Decl(
                "ar",
                c_ast.ArrayDecl(c_ast.TypeDecl("ar", type=c_ast.IdType(["long"])), dim=None, dim_quals=[]),
                init=c_ast.InitList(
                    [c_ast.Constant("int", "7"), c_ast.Constant("int", "8"), c_ast.Constant("int", "9")]
                ),
            ),
        ),
        (
            "long ar[4] = {};",
            c_ast.Decl(
                "ar",
                c_ast.ArrayDecl(
                    c_ast.TypeDecl("ar", type=c_ast.IdType(["long"])),
                    dim=c_ast.Constant("int", "4"),
                    dim_quals=[],
                ),
                init=c_ast.InitList([]),
            ),
        ),
        (
            "char p = j;",
            c_ast.Decl("p", c_ast.TypeDecl("p", type=c_ast.IdType(["char"])), init=c_ast.Id("j")),
        ),
        (
            "char x = 'c', *p = {0, 1, 2, {4, 5}, 6};",
            [
                c_ast.Decl("x", c_ast.TypeDecl("x", type=c_ast.IdType(["char"])), init=c_ast.Constant("char", "'c'")),
                c_ast.Decl(
                    "p",
                    c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("p", type=c_ast.IdType(["char"]))),
                    init=c_ast.InitList(
                        [
                            c_ast.Constant("int", "0"),
                            c_ast.Constant("int", "1"),
                            c_ast.Constant("int", "2"),
                            c_ast.InitList([c_ast.Constant("int", "4"), c_ast.Constant("int", "5")]),
                            c_ast.Constant("int", "6"),
                        ]
                    ),
                ),
            ],
        ),
        (
            "float d = 1.0;",
            c_ast.Decl("d", c_ast.TypeDecl("d", type=c_ast.IdType(["float"])), init=c_ast.Constant("double", "1.0")),
        ),
        (
            "float ld = 1.0l;",
            c_ast.Decl(
                "ld",
                c_ast.TypeDecl("ld", type=c_ast.IdType(["float"])),
                init=c_ast.Constant("long double", "1.0l"),
            ),
        ),
        (
            "float ld = 1.0L;",
            c_ast.Decl(
                "ld",
                c_ast.TypeDecl("ld", type=c_ast.IdType(["float"])),
                init=c_ast.Constant("long double", "1.0L"),
            ),
        ),
        (
            "float ld = 1.0f;",
            c_ast.Decl("ld", c_ast.TypeDecl("ld", type=c_ast.IdType(["float"])), init=c_ast.Constant("float", "1.0f")),
        ),
        (
            "float ld = 1.0F;",
            c_ast.Decl("ld", c_ast.TypeDecl("ld", type=c_ast.IdType(["float"])), init=c_ast.Constant("float", "1.0F")),
        ),
        (
            "float ld = 0xDE.38p0;",
            c_ast.Decl(
                "ld",
                c_ast.TypeDecl("ld", type=c_ast.IdType(["float"])),
                init=c_ast.Constant("float", "0xDE.38p0"),
            ),
        ),
        (
            "int i = 1;",
            c_ast.Decl("i", c_ast.TypeDecl("i", type=c_ast.IdType(["int"])), init=c_ast.Constant("int", "1")),
        ),
        (
            "long int li = 1l;",
            c_ast.Decl(
                "li",
                c_ast.TypeDecl("li", type=c_ast.IdType(["long", "int"])),
                init=c_ast.Constant("long int", "1l"),
            ),
        ),
        (
            "unsigned int ui = 1u;",
            c_ast.Decl(
                "ui",
                c_ast.TypeDecl("ui", type=c_ast.IdType(["unsigned", "int"])),
                init=c_ast.Constant("unsigned int", "1u"),
            ),
        ),
        (
            "unsigned long long int ulli = 1LLU;",
            c_ast.Decl(
                "ulli",
                c_ast.TypeDecl("ulli", type=c_ast.IdType(["unsigned", "long", "long", "int"])),
                init=c_ast.Constant("unsigned long long int", "1LLU"),
            ),
        ),
    ],
)
def test_decl_inits(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)

    if isinstance(expected, list):
        assert c_ast.compare(tree.ext, expected)
    else:
        assert tree.ext[0] == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int a = {.k = 16};",
            c_ast.InitList([c_ast.NamedInitializer(name=[c_ast.Id("k")], expr=c_ast.Constant("int", "16"))]),
        ),
        (
            "int a = { [0].a = {1}, [1].a[0] = 2 };",
            c_ast.InitList(
                [
                    c_ast.NamedInitializer(
                        name=[c_ast.Constant("int", "0"), c_ast.Id("a")],
                        expr=c_ast.InitList([c_ast.Constant("int", "1")]),
                    ),
                    c_ast.NamedInitializer(
                        name=[c_ast.Constant("int", "1"), c_ast.Id("a"), c_ast.Constant("int", "0")],
                        expr=c_ast.Constant("int", "2"),
                    ),
                ]
            ),
        ),
        (
            "int a = { .a = 1, .c = 3, 4, .b = 5};",
            c_ast.InitList(
                [
                    c_ast.NamedInitializer(name=[c_ast.Id("a")], expr=c_ast.Constant("int", "1")),
                    c_ast.NamedInitializer(name=[c_ast.Id("c")], expr=c_ast.Constant("int", "3")),
                    c_ast.Constant("int", "4"),
                    c_ast.NamedInitializer(name=[c_ast.Id("b")], expr=c_ast.Constant("int", "5")),
                ]
            ),
        ),
    ],
)
def test_decl_named_inits(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    assert isinstance(tree.ext[0], c_ast.Decl)

    init = tree.ext[0].init
    assert isinstance(init, c_ast.InitList)
    assert init == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int factorial(int p)\n{\n    return 3;\n}",
            c_ast.FuncDef(
                decl=c_ast.Decl(
                    "factorial",
                    c_ast.FuncDecl(
                        args=c_ast.ParamList([c_ast.Decl("p", c_ast.TypeDecl("p", type=c_ast.IdType(["int"])))]),
                        type=c_ast.TypeDecl("factorial", type=c_ast.IdType(["int"])),
                    ),
                ),
                param_decls=None,
                body=c_ast.Compound([c_ast.Return(expr=c_ast.Constant("int", "3"))]),
            ),
        ),
        (
            "char* zzz(int p, char* c)\n"
            "{\n"
            "    int a;\n"
            "    char b;\n"
            "\n"
            "    a = b + 2;\n"
            "    return 3;\n"
            "}",
            c_ast.FuncDef(
                decl=c_ast.Decl(
                    "zzz",
                    c_ast.FuncDecl(
                        args=c_ast.ParamList(
                            [
                                c_ast.Decl("p", c_ast.TypeDecl("p", type=c_ast.IdType(["int"]))),
                                c_ast.Decl(
                                    "c",
                                    c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("c", type=c_ast.IdType(["char"]))),
                                ),
                            ]
                        ),
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("zzz", type=c_ast.IdType(["char"]))),
                    ),
                ),
                param_decls=None,
                body=c_ast.Compound(
                    [
                        c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"]))),
                        c_ast.Decl("b", c_ast.TypeDecl("b", type=c_ast.IdType(["char"]))),
                        c_ast.Assignment(
                            op="=",
                            left=c_ast.Id("a"),
                            right=c_ast.BinaryOp(op="+", left=c_ast.Id("b"), right=c_ast.Constant("int", "2")),
                        ),
                        c_ast.Return(expr=c_ast.Constant("int", "3")),
                    ]
                ),
            ),
        ),
        (
            "char* zzz(p, c)\n"
            "long p, *c;\n"
            "{\n"
            "    int a;\n"
            "    char b;\n"
            "\n"
            "    a = b + 2;\n"
            "    return 3;\n"
            "}",
            c_ast.FuncDef(
                decl=c_ast.Decl(
                    "zzz",
                    c_ast.FuncDecl(
                        args=c_ast.ParamList([c_ast.Id("p"), c_ast.Id("c")]),
                        type=c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("zzz", type=c_ast.IdType(["char"]))),
                    ),
                ),
                param_decls=[
                    c_ast.Decl("p", c_ast.TypeDecl("p", type=c_ast.IdType(["long"]))),
                    c_ast.Decl("c", c_ast.PtrDecl(quals=[], type=c_ast.TypeDecl("c", type=c_ast.IdType(["long"])))),
                ],
                body=c_ast.Compound(
                    [
                        c_ast.Decl("a", c_ast.TypeDecl("a", type=c_ast.IdType(["int"]))),
                        c_ast.Decl("b", c_ast.TypeDecl("b", type=c_ast.IdType(["char"]))),
                        c_ast.Assignment(
                            op="=",
                            left=c_ast.Id("a"),
                            right=c_ast.BinaryOp(op="+", left=c_ast.Id("b"), right=c_ast.Constant("int", "2")),
                        ),
                        c_ast.Return(expr=c_ast.Constant("int", "3")),
                    ]
                ),
            ),
        ),
        pytest.param(
            "que(p)\n{\n    return 3;\n}",
            c_ast.FuncDef(
                decl=c_ast.Decl(
                    "que",
                    c_ast.FuncDecl(
                        args=c_ast.ParamList([c_ast.Id("p")]),
                        type=c_ast.TypeDecl("que", type=c_ast.IdType(["int"])),
                    ),
                ),
                param_decls=None,
                body=c_ast.Compound([c_ast.Return(expr=c_ast.Constant("int", "3"))]),
            ),
            id="function return values and parameters may not have type information",
        ),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_function_definitions(test_input: str, expected: c_ast.AST):
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

    expected_assert_1 = c_ast.StaticAssert(cond=c_ast.Constant("int", "1"), message=c_ast.Constant("string", '"123"'))
    assert tree.ext[0] == expected_assert_1

    expected_assert_2 = c_ast.StaticAssert(cond=c_ast.Constant("int", "2"), message=c_ast.Constant("string", '"456"'))
    assert tree.ext[1].body.block_items[0] == expected_assert_2  # type: ignore

    expected_assert_3 = c_ast.StaticAssert(cond=c_ast.Constant("int", "3"), message=None)
    assert tree.ext[1].body.block_items[2] == expected_assert_3  # type: ignore


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            'char* s = "hello";',
            c_ast.Constant("string", '"hello"'),
            id="simple string, for reference",
        ),
        (
            'char* s = "hello" " world";',
            c_ast.Constant("string", '"hello world"'),
        ),
        (
            'char* s = "" "foobar";',
            c_ast.Constant("string", '"foobar"'),
        ),
        (
            r'char* s = "foo\"" "bar";',
            c_ast.Constant("string", r'"foo\"bar"'),
        ),
    ],
)
def test_unified_string_literals(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)
    assert isinstance(tree.ext[0], c_ast.Decl)

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

    expected = c_ast.Constant("string", r'"\123"')
    assert tree.ext[0].init == expected  # type: ignore


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

    assert tree.ext[0].body.block_items[0].args.exprs[1].value == r'"Wrong Params?\nUsage:\n%s <binary_file_path>\n"'  # type: ignore


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            'char* s = L"hello" L"world";',
            c_ast.Constant("string", 'L"helloworld"'),
        ),
        (
            'char* s = L"hello " L"world" L" and I";',
            c_ast.Constant("string", 'L"hello world and I"'),
        ),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_unified_wstring_literals(test_input: str, expected: c_ast.AST):
    tree = parse(test_input)

    assert tree.ext[0].init == expected  # type: ignore


@pytest.mark.xfail(reason="TODO")
def test_inline_specifier():
    test_input = "static inline void inlinefoo(void);"
    tree = parse(test_input)

    assert tree.ext[0].funcspec == ["inline"]  # type: ignore


def test_noreturn_specifier():
    test_input = "static _Noreturn void noreturnfoo(void);"
    tree = parse(test_input)

    assert tree.ext[0].funcspec == ["_Noreturn"]  # type: ignore


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

    expected_dim_1 = c_ast.Assignment(op="=", left=c_ast.Id("size"), right=c_ast.Constant("int", "5"))
    assert tree.ext[0].body.block_items[1].type.dim == expected_dim_1  # type: ignore

    expected_dim_2 = c_ast.Id("*")
    assert tree.ext[0].body.block_items[2].type.dim == expected_dim_2  # type: ignore


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
    assert pragma1 == c_ast.Pragma("bar")
    assert pragma1.coord.line_start == 2  # type: ignore

    pragma2 = tree.ext[1].body.block_items[0]  # type: ignore
    assert pragma2 == c_ast.Pragma("foo")
    assert pragma2.coord.line_start == 4  # type: ignore

    pragma3 = tree.ext[1].body.block_items[2]  # type: ignore
    assert pragma3 == c_ast.Pragma("baz")
    assert pragma3.coord.line_start == 6  # type: ignore

    pragma4 = tree.ext[1].body.block_items[4]  # type: ignore
    assert pragma4 == c_ast.Pragma("")
    assert pragma4.coord.line_start == 10  # type: ignore

    pragma5 = tree.ext[2].body.block_items[0]  # type: ignore
    assert pragma5 == c_ast.Pragma("baz")
    assert pragma5.coord.line_start == 13  # type: ignore

    pragma6 = tree.ext[3]
    assert pragma6 == c_ast.Pragma(r'"other \"string\""')
    assert pragma6.coord.line_start == 15  # type: ignore


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
        c_ast.Decl(
            "sum",
            c_ast.TypeDecl("sum", type=c_ast.IdType(["int"])),
            init=c_ast.Constant("int", "0"),
        ),
        c_ast.For(
            init=c_ast.DeclList([c_ast.Decl("i", c_ast.TypeDecl("i", type=c_ast.IdType(["int"])))]),
            cond=c_ast.BinaryOp(op="<", left=c_ast.Id("i"), right=c_ast.Constant("int", "3")),
            next=c_ast.UnaryOp(op="p++", expr=c_ast.Id("i")),
            stmt=c_ast.Compound(
                [
                    c_ast.Pragma("omp critical"),
                    c_ast.Assignment(op="+=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "1")),
                ]
            ),
        ),
        c_ast.While(
            cond=c_ast.BinaryOp(op="<", left=c_ast.Id("sum"), right=c_ast.Constant("int", "10")),
            stmt=c_ast.Compound(
                [
                    c_ast.Pragma("omp critical"),
                    c_ast.Assignment(op="+=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "1")),
                ]
            ),
        ),
        c_ast.Label(
            "mylabel",
            stmt=c_ast.Compound(
                [
                    c_ast.Pragma("foo"),
                    c_ast.Assignment(op="+=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "10")),
                ]
            ),
        ),
        c_ast.If(
            cond=c_ast.BinaryOp(op=">", left=c_ast.Id("sum"), right=c_ast.Constant("int", "10")),
            iftrue=c_ast.Compound(
                [
                    c_ast.Pragma("bar"),
                    c_ast.Pragma("baz"),
                    c_ast.Assignment(op="=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "10")),
                ]
            ),
            iffalse=None,
        ),
        c_ast.Switch(
            cond=c_ast.Id("sum"),
            stmt=c_ast.Case(
                expr=c_ast.Constant("int", "10"),
                stmts=[
                    c_ast.Compound(
                        [
                            c_ast.Pragma(string="foo"),
                            c_ast.Assignment(op="=", left=c_ast.Id("sum"), right=c_ast.Constant("int", "20")),
                        ]
                    )
                ],
            ),
        ),
    ]

    assert c_ast.compare(tree.ext[0].body.block_items, expected_list)  # type: ignore
