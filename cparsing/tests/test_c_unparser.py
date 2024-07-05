import platform
from pathlib import Path
from typing import Literal, Tuple

import pytest

from cparsing import c_ast, parse, parse_file

SAMPLE_CFILES_PATH = Path("./tests/c_files").resolve(strict=True)


def cpp_path() -> Literal["gcc", "cpp"]:
    """Path to cpp command."""

    if platform.system() == "Darwin":
        return "gcc"
    return "cpp"


def cpp_args(args: Tuple[str, ...] = ()) -> Tuple[str, ...]:
    """Turn args into a suitable format for passing to cpp."""

    if platform.system() == "Darwin":
        return ("-E", *args)
    return args


def convert_c_to_c(src: str, *, reduce_parentheses: bool = False) -> str:
    return c_ast.unparse(parse(src), reduce_parentheses=reduce_parentheses)


def assert_is_c_to_c_correct(src: str, *, reduce_parentheses: bool = False):
    """Checks that the c2c translation was correct by parsing the code generated by c2c for src and comparing the AST
    with the original AST.
    """

    reparsed_src = convert_c_to_c(src, reduce_parentheses=reduce_parentheses)
    assert c_ast.compare_asts(parse(src), parse(reparsed_src))


# =====================================================================================================================

# ======== Test C to C.


@pytest.mark.parametrize(
    "test_input",
    [
        pytest.param("int a;", id="trivial decls 1"),
        pytest.param("int b, a;", id="trivial decls 2"),
        pytest.param("int c, b, a;", id="trivial decls 3"),
        pytest.param("auto int a;", id="trivial decls 4"),
        pytest.param("register int a;", id="trivial decls 5"),
        pytest.param("_Thread_local int a;", id="trivial decls 6"),
        pytest.param("int a;", id="complex decls 1"),
        pytest.param("int b, a;", id="complex decls 2"),
        pytest.param("int c, b, a;", id="complex decls 3"),
        pytest.param("auto int a;", id="complex decls 4"),
        pytest.param("register int a;", id="complex decls 5"),
        pytest.param("_Thread_local int a;", id="complex decls 6"),
        pytest.param("_Alignas(32) int b;", id="alignment 1"),
        pytest.param("int _Alignas(32) a;", id="alignment 2"),
        pytest.param("_Alignas(32) _Atomic(int) b;", id="alignment 3"),
        pytest.param("_Atomic(int) _Alignas(32) b;", id="alignment 4"),
        pytest.param("_Alignas(long long) int a;", id="alignment 5"),
        pytest.param("int _Alignas(long long) a;", id="alignment 6"),
        pytest.param(
            """\
typedef struct node_t {
    _Alignas(64) void* next;
    int data;
} node;
""",
            id="alignment 7",
        ),
        pytest.param(
            """\
typedef struct node_t {
    void _Alignas(64) * next;
    int data;
} node;
""",
            id="alignment 8",
        ),
        pytest.param(
            """\
int main(void)
{
    int a, b;
    (a == 0) ? (b = 1) : (b = 2);
}""",
            id="ternary",
        ),
        pytest.param(
            """\
int main() {
    int b = (int) f;
    int c = (int*) f;
}""",
            id="casts 1",
        ),
        pytest.param(
            """\
int main() {
    int a = (int) b + 8;
    int t = (int) c;
}
""",
            id="casts 2",
        ),
        pytest.param("int arr[] = {1, 2, 3};", id="initlist"),
        pytest.param(
            """\
int main(void)
{
    int a;
    int b = a++;
    int c = ++a;
    int d = a--;
    int e = --a;
}
""",
            id="exprs",
        ),
        pytest.param(
            """\
int main() {
    int a;
    a = 5;
    ;
    b = - - a;
    return a;
}""",
            id="statements (note two minuses here)",
        ),
        pytest.param(
            """\
typedef struct node_t {
    struct node_t* next;
    int data;
} node;
""",
            id="struct decl",
        ),
        pytest.param(
            """\
int main(argc, argv)
int argc;
char** argv;
{
    return 0;
}
""",
            id="krstyle",
        ),
        pytest.param(
            """\
int main() {
    switch (myvar) {
    case 10:
    {
        k = 10;
        p = k + 1;
        break;
    }
    case 20:
    case 30:
        return 20;
    default:
        break;
    }
}
""",
            id="switchcase",
        ),
        pytest.param(
            """\
int main()
{
    int i[1][1] = { { 1 } };
}""",
            id="nest initializer list",
        ),
        pytest.param(
            """\
struct test
{
    int i;
    struct test_i_t
    {
        int k;
    } test_i;
    int j;
};
struct test test_var = {.i = 0, .test_i = {.k = 1}, .j = 2};
""",
            id="nest named initializer",
        ),
        pytest.param(
            """\
int main()
{
    int i[1] = { (1, 2) };
}""",
            id="expr list in initializer list",
        ),
        pytest.param(
            """\
_Noreturn int x(void) {
    abort();
}
""",
            id="noreturn",
        ),
        pytest.param(
            """\
void x() {
    if (i < j)
        tmp = C[i], C[i] = C[j], C[j] = tmp;
    if (i <= j)
        i++, j--;
}
""",
            id="exprlist with semi",
        ),
        pytest.param(
            """\
void x() {
    (a = b, (b = c, c = a));
}
""",
            id="exprlist with subexprlist",
        ),
        pytest.param(
            """\
void f(int x) { return x; }
int main(void) { f((1, 2)); return 0; }
""",
            id="comma operator funcarg",
        ),
        pytest.param(
            """\
void f() {
    (0, 0) ? (0, 0) : (0, 0);
}
""",
            id="comma op in ternary",
        ),
        pytest.param(
            """
void f() {
    i = (a, b, c);
}
""",
            id="comma op assignment",
        ),
        pytest.param(
            """\
#pragma foo
void f() {
    #pragma bar
    i = (a, b, c);
    if (d)
        #pragma qux
        j = e;
    if (d)
        #pragma qux
        #pragma quux
        j = e;
}
typedef struct s {
#pragma baz
} s;
""",
            id="pragma",
        ),
        pytest.param('char **foo = (char *[]){ "x", "y", "z" };', id="compound literal 1"),
        pytest.param("int i = ++(int){ 1 };", id="compound literal 2"),
        pytest.param("struct foo_s foo = (struct foo_s){ 1, 2 };", id="compound literal 3"),
        pytest.param(
            """\
enum e
{
    a,
    b = 2,
    c = 3
};
""",
            id="enum 1",
        ),
        pytest.param(
            """\
enum f
{
    g = 4,
    h,
    i
};
""",
            id="enum 2",
        ),
        pytest.param("typedef enum EnumName EnumTypedefName;", id="enum typedef"),
        pytest.param("int g(const int a[const 20]){}", id="array decl"),
        pytest.param("const int ** const  x;", id="ptr decl"),
        pytest.param(f"int x = {'sizeof(' * 30}1{')' * 30}", id="nested sizeof"),
        pytest.param('_Static_assert(sizeof(int) == sizeof(int), "123");', id="static assert 1"),
        pytest.param('int main() { _Static_assert(sizeof(int) == sizeof(int), "123"); } ', id="static assert 2"),
        pytest.param("_Static_assert(sizeof(int) == sizeof(int));", id="static assert 3"),
        pytest.param("_Atomic int x;", id="atomic qual 1"),
        pytest.param("_Atomic int* x;", id="atomic qual 2"),
        pytest.param("int* _Atomic x;", id="atomic qual 3"),
        # ==== Issues
        pytest.param("\nint main() {\n}", id="issue 36"),
        pytest.param(
            """\
int main(void)
{
    unsigned size;
    size = sizeof(size);
    return 0;
}""",
            id="issue 37",
        ),
        pytest.param(
            "\nstruct foo;",
            id="issue 66, 1; A non-existing body must not be generated (previous valid behavior, still working)",
        ),
        pytest.param(
            "\nstruct foo {};",
            id="issue 66, 2; An empty body must be generated (added behavior)",
        ),
        pytest.param(
            """\
void x(void) {
    int i = (9, k);
}
""",
            id="issue 83",
        ),
        pytest.param(
            """\
void x(void) {
    for (int i = 0;;)
        i;
}
""",
            id="issue 84",
        ),
        pytest.param("int array[3] = {[0] = 0, [1] = 1, [1+1] = 2};", id="issue 246"),
    ],
)
def test_c_to_c(test_input: str):
    assert_is_c_to_c_correct(test_input)


def test_partial_funcdecl_generation():
    test_input = """\
void noop(void);
void *something(void *thing);
int add(int x, int y);"""
    tree = parse(test_input)
    stubs = [c_ast.unparse(node) for node in c_ast.walk(tree) if isinstance(node, c_ast.FuncDecl)]

    assert len(stubs) == 3
    assert "void noop(void)" in stubs
    assert "void *something(void *thing)" in stubs
    assert "int add(int x, int y)" in stubs


def test_array_decl_subnodes():
    tree = parse("const int a[const 20];")

    assert c_ast.unparse(tree.ext[0].type) == "const int [const 20]"  # type: ignore
    assert c_ast.unparse(tree.ext[0].type.type) == "const int"  # type: ignore


def test_ptr_decl_subnodes():
    tree = parse("const int ** const  x;")

    assert c_ast.unparse(tree.ext[0].type) == "const int ** const"  # type: ignore
    assert c_ast.unparse(tree.ext[0].type.type) == "const int *"  # type: ignore
    assert c_ast.unparse(tree.ext[0].type.type.type) == "const int"  # type: ignore


@pytest.mark.parametrize(
    ("test_input", "expected_reparsed_input"),
    [
        ("_Atomic(int) x;", "_Atomic int x;\n"),
        ("_Atomic(int*) x;", "int * _Atomic x;\n"),
        ("_Atomic(_Atomic(int)*) x;", "_Atomic int * _Atomic x;\n"),
        ("typedef _Atomic(int) atomic_int;", "typedef _Atomic int atomic_int;\n"),
        (
            "typedef _Atomic(_Atomic(_Atomic(int (*)(void)) *) *) t;",
            "typedef int (* _Atomic * _Atomic * _Atomic t)(void);\n",
        ),
        (
            """\
typedef struct node_t {
    _Atomic(void*) a;
    _Atomic(void) *b;
    _Atomic void *c;
} node;
""",
            """\
typedef struct node_t
{
  void * _Atomic a;
  _Atomic void *b;
  _Atomic void *c;
} node;

""",
        ),
    ],
)
def test_atomic_qual(test_input: str, expected_reparsed_input: str):
    # _Atomic specifier gets turned into qualifier.
    assert convert_c_to_c(test_input) == expected_reparsed_input
    assert_is_c_to_c_correct(test_input)

    # TODO: Regeneration with multiple qualifiers is not fully supported.
    # REF: https://github.com/eliben/pycparser/issues/433
    # assert is_c_to_c_correct('auto const _Atomic(int *) a;')


def test_reduce_parentheses_binaryops():
    test_input = "int x = a + b + c + d;"
    assert convert_c_to_c(test_input) == "int x = ((a + b) + c) + d;\n"
    assert convert_c_to_c(test_input, reduce_parentheses=True) == "int x = a + b + c + d;\n"


@pytest.mark.parametrize(
    "test_input",
    [
        "int x = a*b*c*d;",
        "int x = a+b*c*d;",
        "int x = a*b+c*d;",
        "int x = a*b*c+d;",
        "int x = (a+b)*c*d;",
        "int x = (a+b)*(c+d);",
        "int x = (a+b)/(c-d);",
        "int x = a+b-c-d;",
        "int x = a+(b-c)-d;",
    ],
)
def test_minimum_parentheses_binaryops(test_input: str):
    # codes with minimum number of (necessary) parenthesis:
    assert_is_c_to_c_correct(test_input, reduce_parentheses=True)
    reparsed_source = convert_c_to_c(test_input, reduce_parentheses=True)
    assert reparsed_source.count("(") == test_input.count("(")


def test_to_type():
    test_input = "int *x;"
    test_func = c_ast.FuncCall(c_ast.Id("test_fun"), c_ast.ExprList([]))

    tree = parse(test_input)
    assert c_ast.unparse(c_ast.Cast(tree.ext[0].type, test_func)) == "(int *) test_fun()"  # type: ignore
    assert c_ast.unparse(c_ast.Cast(tree.ext[0].type.type, test_func)) == "(int) test_fun()"  # type: ignore


@pytest.mark.skipif(platform.system() != "Linux", reason="cpp only works on Unix")
def test_to_type_with_cpp():
    test_func = c_ast.FuncCall(c_ast.Id("test_fun"), c_ast.ExprList([]))
    memmgr_path = SAMPLE_CFILES_PATH / "memmgr.h"

    tree = parse_file(memmgr_path, use_cpp=True, cpp_path=cpp_path(), cpp_args=cpp_args())
    assert c_ast.unparse(c_ast.Cast(tree.ext[-3].type.type, test_func)) == "(void *) test_fun()"  # type: ignore
    assert c_ast.unparse(c_ast.Cast(tree.ext[-3].type.type.type, test_func)) == "(void) test_fun()"  # type: ignore


@pytest.mark.parametrize(
    ("test_tree", "expected"),
    [
        (c_ast.If(None, None, None), "if ()\n  \n"),
        (c_ast.If(None, None, c_ast.If(None, None, None)), "if ()\n  \nelse\n  if ()\n  \n"),
        (
            c_ast.If(None, None, c_ast.If(None, None, c_ast.If(None, None, None))),
            "if ()\n  \nelse\n  if ()\n  \nelse\n  if ()\n  \n",
        ),
        (
            c_ast.If(
                None,
                c_ast.Compound([]),
                c_ast.If(None, c_ast.Compound([]), c_ast.If(None, c_ast.Compound([]), None)),
            ),
            "if ()\n{\n}\nelse\n  if ()\n{\n}\nelse\n  if ()\n{\n}\n",
        ),
    ],
)
def test_nested_else_if_line_breaks(test_tree: c_ast.AST, expected: str):
    assert c_ast.unparse(test_tree) == expected
