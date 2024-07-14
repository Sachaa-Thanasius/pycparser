"""Test the "coordinates" of parsed elements - file name, line and column numbers, with modification inserted by
#line directives.

TODO: Reexamine Coord system, how they're populated in nodes, the expected results, etc. Compare to pycparser.
"""

import pytest
from cparsing import c_ast, parse
from cparsing.utils import Coord


@pytest.mark.parametrize(("test_input", "expected_coord"), [("int a;", Coord(1, 4, filename="<unknown>"))])
def test_coords_without_filename(test_input: str, expected_coord: Coord):
    tree = parse(test_input)
    assert tree.ext[0].coord == expected_coord


def test_coords_with_filename_1():
    test_input = """\
int a;
int b;


int c;
"""
    filename = "test.c"

    tree = parse(test_input, filename=filename)

    assert tree.ext[0].coord == Coord(1, 4, filename=filename)
    assert tree.ext[1].coord == Coord(2, 11, filename=filename)
    assert tree.ext[2].coord == Coord(5, 20, filename=filename)


# @pytest.mark.xfail(reason="TODO")
def test_coords_with_filename_2():
    test_input = """\
int main() {
    k = p;
    printf("%d", b);
    return 0;
}"""
    filename = "test.c"

    tree = parse(test_input, filename=filename)

    assert tree.ext[0].body.block_items[0].coord == Coord(3, *(13, 0), filename="test.c")  # type: ignore
    assert tree.ext[0].body.block_items[1].coord == Coord(4, *(13, 0), filename="test.c")  # type: ignore


@pytest.mark.xfail(reason="TODO")
def test_coords_on_Cast():
    """Issue 23: Make sure that the Cast has a coord."""

    test_input = """\
    int main () {
        int p = (int) k;
    }"""
    filename = "test.c"

    tree = parse(test_input, filename=filename)

    assert tree.ext[0].body.block_items[0].init.coord == Coord(3, *(21, 0), filename=filename)  # type: ignore


@pytest.mark.xfail(reason="TODO")
@pytest.mark.parametrize(
    ("test_input", "filename", "expected_coords"),
    [
        (
            "#line 99\nint c;",
            "<unknown>",
            [Coord(99, *(13, 0))],
        ),
        (
            'int dsf;\nchar p;\n#line 3000 "in.h"\nchar d;',
            "test.c",
            [
                Coord(2, *(13, 0), filename="test.c"),
                Coord(3, *(14, 0), filename="test.c"),
                Coord(3000, *(14, 0), filename="in.h"),
            ],
        ),
        (
            """\
#line 20 "restore.h"
int maydler(char);

#line 30 "includes/daween.ph"
long j, k;

#line 50000
char* ro;
""",
            "myb.c",
            [
                Coord(20, *(13, 0), filename="restore.h"),
                Coord(30, *(14, 0), filename="includes/daween.ph"),
                Coord(30, *(17, 0), filename="includes/daween.ph"),
                Coord(50000, *(13, 0), filename="includes/daween.ph"),
            ],
        ),
        ("int\n#line 99\nc;", "<unknown>", [Coord(99, *(9, 0))]),
    ],
)
def test_coords_with_line_directive(test_input: str, filename: str, expected_coords: list[Coord]):
    tree = parse(test_input, filename)

    for index, coord in enumerate(expected_coords):
        assert tree.ext[index].coord == coord


@pytest.mark.xfail(reason="TODO")
def test_coord_for_ellipsis():
    test_input = """\
    int foo(int j,
            ...) {
    }"""
    tree = parse(test_input)
    coord = tree.ext[0].decl.type.args.params[1].coord  # type: ignore
    assert coord == Coord(3, *(17, 0))


@pytest.mark.xfail(reason="TODO")
def test_coords_forloop() -> None:
    test_input = """\
void foo() {
    for(int z=0; z<4;
        z++){}
}
"""

    tree = parse(test_input, filename="f.c")

    for_loop = tree.ext[0].body.block_items[0]  # type: ignore
    assert isinstance(for_loop, c_ast.For)

    assert for_loop.init
    assert for_loop.init.coord == Coord(2, 13, filename="f.c")

    assert for_loop.cond
    assert for_loop.cond.coord == Coord(2, 26, filename="f.c")

    assert for_loop.next
    assert for_loop.next.coord == Coord(3, 17, filename="f.c")
