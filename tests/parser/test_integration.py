"""Test parsing whole chunks of code, even some in files.

Notes
-----
Since we don't want to rely on the structure of ASTs too much, most of these tests are implemented with walk().
"""

from pathlib import Path
from typing import Union

import pytest
from cparsing import c_ast, parse


def match_constants(code: Union[str, c_ast.AST], constants: list[str]) -> bool:
    """Check that the list of all Constant values (by 'preorder' appearance) in the chunk of code is as given."""

    tree = parse(code) if isinstance(code, str) else code
    return [node.value for node in c_ast.walk(tree) if isinstance(node, c_ast.Constant)] == constants


def match_number_of_id_refs(code: Union[str, c_ast.AST], name: str, num: int) -> bool:
    """Check that the number of references to the ID with the given name matches the expected number."""

    tree = parse(code) if isinstance(code, str) else code
    return sum(1 for node in c_ast.walk(tree) if isinstance(node, c_ast.Id) and node.name == name) == num


def match_number_of_node_instances(code: Union[str, c_ast.AST], type: type[c_ast.AST], num: int) -> None:  # noqa: A002
    """Check that the amount of klass nodes in the code is the expected number."""

    tree = parse(code) if isinstance(code, str) else code
    assert sum(1 for node in c_ast.walk(tree) if isinstance(node, type)) == num


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        pytest.param(
            "int k = (r + 10.0) >> 6 + 8 << (3 & 0x14);",
            ["10.0", "6", "8", "3", "0x14"],
            marks=pytest.mark.xfail(reason="TODO"),
        ),
        (
            r"""char n = '\n', *prefix = "st_";""",
            [r"'\n'", '"st_"'],
        ),
        pytest.param(
            "int main() {\n"
            "    int i = 5, j = 6, k = 1;\n"
            "    if ((i=j && k == 1) || k > j)\n"
            '        printf("Hello, world\n");\n'
            "    return 0;\n"
            "}",
            ["5", "6", "1", "1", '"Hello, world\\n"', "0"],
            marks=pytest.mark.xfail(reason="TODO"),
        ),
    ],
)
def test_expressions_constants(test_input: str, expected: list[str]) -> None:
    tree = parse(test_input)
    assert match_constants(tree, expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "int main() {\n"
            "    int i = 5, j = 6, k = 1;\n"
            "    if ((i=j && k == 1) || k > j)\n"
            '        printf("Hello, world\n");\n'
            "    return 0;\n"
            "}",
            [("i", 1), ("j", 2)],
        )
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_expressions_id_refs(test_input: str, expected: list[tuple[str, int]]):
    tree = parse(test_input)
    for id_, num in expected:
        assert match_number_of_id_refs(tree, id_, num)


@pytest.mark.parametrize(
    ("test_input", "expected_constants", "expected_id_ref_counts", "expected_node_instance_counts"),
    [
        (
            r"""
void foo(){
if (sp == 1)
    if (optind >= argc ||
        argv[optind][0] != '-' || argv[optind][1] == '\0')
            return -1;
    else if (strcmp(argv[optind], "--") == 0) {
        optind++;
        return -1;
    }
}
""",
            ["1", "0", r"'-'", "1", r"'\0'", "1", r'"--"', "0", "1"],
            [("argv", 3), ("optind", 5)],
            [(c_ast.If, 3), (c_ast.Return, 2), (c_ast.FuncCall, 1), (c_ast.BinaryOp, 7)],  # FuncCall is strcmp
        ),
        pytest.param(
            r"""
typedef int Hash, Node;

void HashDestroy(Hash* hash)
{
    unsigned int i;

    if (hash == NULL)
        return;

    for (i = 0; i < hash->table_size; ++i)
    {
        Node* temp = hash->heads[i];

        while (temp != NULL)
        {
            Node* temp2 = temp;

            free(temp->entry->key);
            free(temp->entry->value);
            free(temp->entry);

            temp = temp->next;

            free(temp2);
        }
    }

    free(hash->heads);
    hash->heads = NULL;

    free(hash);
}
""",
            ["0"],
            [("hash", 6), ("i", 4)],  # declarations don't count
            [(c_ast.FuncCall, 6), (c_ast.FuncDef, 1), (c_ast.For, 1), (c_ast.While, 1), (c_ast.StructRef, 10)],
            id="Hash and Node were defined as int to pacify the parser that sees they're used as types",
        ),
        (
            r"""
void x(void) {
    int a, b;
    if (a < b)
    do {
        a = 0;
    } while (0);
    else if (a == b) {
    a = 1;
    }
}
""",
            ["0", "0", "1"],
            [("a", 4)],
            [(c_ast.DoWhile, 1)],
        ),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_statements(
    test_input: str,
    expected_constants: list[str],
    expected_id_ref_counts: list[tuple[str, int]],
    expected_node_instance_counts: list[tuple[type[c_ast.AST], int]],
):
    tree = parse(test_input)

    assert match_constants(tree, expected_constants)

    for id_, count in expected_id_ref_counts:
        assert match_number_of_id_refs(tree, id_, count)

    for node_type, count in expected_node_instance_counts:
        assert match_number_of_node_instances(tree, node_type, count)


@pytest.mark.xfail(reason="TODO")
def test_empty_statements():
    test_input = r"""
void foo(void){
    ;
    return;;

    ;
}
"""

    tree = parse(test_input)

    assert match_number_of_node_instances(tree, c_ast.EmptyStatement, 3)
    assert match_number_of_node_instances(tree, c_ast.Return, 1)

    assert tree.ext[0].body.block_items[0].coord.line_start == 3  # pyright: ignore
    assert tree.ext[0].body.block_items[1].coord.line_start == 4  # pyright: ignore
    assert tree.ext[0].body.block_items[2].coord.line_start == 4  # pyright: ignore
    assert tree.ext[0].body.block_items[3].coord.line_start == 6  # pyright: ignore


@pytest.mark.xfail(reason="TODO")
def test_switch_statement():
    def is_case_node(node: object, const_value: str) -> bool:
        return isinstance(node, c_ast.Case) and isinstance(node.expr, c_ast.Constant) and node.expr.value == const_value

    test_input_1 = r"""
int foo(void) {
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
    return 0;
}
"""

    tree1 = parse(test_input_1)
    switch = tree1.ext[0].body.block_items[0]  # pyright: ignore

    block = switch.stmt.block_items  # pyright: ignore
    assert len(block) == 4  # pyright: ignore

    assert is_case_node(block[0], "10")  # pyright: ignore
    assert len(block[0].stmts) == 3  # pyright: ignore

    assert is_case_node(block[1], "20")  # pyright: ignore
    assert len(block[1].stmts) == 0  # pyright: ignore

    assert is_case_node(block[2], "30")  # pyright: ignore
    assert len(block[2].stmts) == 1  # pyright: ignore

    assert isinstance(block[3], c_ast.Default)

    test_input_2 = r"""
int foo(void) {
    switch (myvar) {
        default:
            joe = moe;
            return 10;
        case 10:
        case 20:
        case 30:
        case 40:
            break;
    }
    return 0;
}
"""

    tree2 = parse(test_input_2)
    switch = tree2.ext[0].body.block_items[0]  # pyright: ignore

    block = switch.stmt.block_items  # pyright: ignore
    assert len(block) == 5  # pyright: ignore

    assert isinstance(block[0], c_ast.Default)
    assert len(block[0].stmts) == 2  # pyright: ignore

    assert is_case_node(block[1], "10")  # pyright: ignore
    assert len(block[1].stmts) == 0  # pyright: ignore

    assert is_case_node(block[2], "20")  # pyright: ignore
    assert len(block[2].stmts) == 0  # pyright: ignore

    assert is_case_node(block[3], "30")  # pyright: ignore
    assert len(block[3].stmts) == 0  # pyright: ignore

    assert is_case_node(block[4], "40")  # pyright: ignore
    assert len(block[4].stmts) == 1  # pyright: ignore

    test_input_3 = r"""
int foo(void) {
    switch (myvar) {
    }
    return 0;
}
"""

    tree3 = parse(test_input_3)
    switch = tree3.ext[0].body.block_items[0]  # pyright: ignore

    assert switch.stmt.block_items == []  # pyright: ignore


@pytest.mark.parametrize(
    ("test_input", "expected_i_ref_count", "expected_For_instance_count"),
    [
        pytest.param(
            r"""
void x(void)
{
    int i;
    for (i = 0; i < 5; ++i) {
        x = 50;
    }
}
""",
            3,
            1,
            id="3 refs for i since the declaration doesn't count in the visitor.",
        ),
        pytest.param(
            r"""
void x(void)
{
    for (int i = 0; i < 5; ++i) {
        x = 50;
    }
}
""",
            2,
            1,
            id="2 refs for i since the declaration doesn't count in the visitor.",
        ),
        (
            r"""
void x(void) {
    for (int i = 0;;)
        i;
}
""",
            1,
            1,
        ),
    ],
)
@pytest.mark.xfail(reason="TODO")
def test_for_statement(test_input: str, expected_i_ref_count: int, expected_For_instance_count: int):
    tree = parse(test_input)
    assert match_number_of_id_refs(tree, "i", expected_i_ref_count)
    assert match_number_of_node_instances(tree, c_ast.For, expected_For_instance_count)


@pytest.mark.xfail(reason="TODO")
def test_whole_file():
    """See how pycparser handles a whole, real C file."""

    SAMPLE_CFILES_PATH = Path().resolve(strict=True) / "tests" / "c_files"

    code = SAMPLE_CFILES_PATH.joinpath("memmgr_with_h.c").read_text(encoding="utf-8")
    test_input = parse(code)

    assert match_number_of_node_instances(test_input, c_ast.FuncDef, 5)

    # each FuncDef also has a FuncDecl. 4 declarations + 5 definitions, overall 9
    assert match_number_of_node_instances(test_input, c_ast.FuncDecl, 9)

    assert match_number_of_node_instances(test_input, c_ast.Typedef, 4)

    assert test_input.ext[4].coord
    assert test_input.ext[4].coord.line_start == 88
    assert test_input.ext[4].coord.filename == "./memmgr.h"

    assert test_input.ext[6].coord
    assert test_input.ext[6].coord.line_start == 10
    assert test_input.ext[6].coord.filename == "memmgr.c"


@pytest.mark.xfail(reason="TODO")
def test_whole_file_with_stdio():
    """Parse a whole file with stdio.h included by cpp."""

    SAMPLE_CFILES_PATH = Path().resolve(strict=True) / "tests" / "c_files"

    code = SAMPLE_CFILES_PATH.joinpath("cppd_with_stdio_h.c").read_text(encoding="utf-8")
    test_input = parse(code)

    assert isinstance(test_input.ext[0], c_ast.Typedef)
    assert test_input.ext[0].coord
    assert test_input.ext[0].coord.line_start == 213
    assert test_input.ext[0].coord.filename == r"D:\eli\cpp_stuff\libc_include/stddef.h"

    assert isinstance(test_input.ext[-1], c_ast.FuncDef)
    assert test_input.ext[-1].coord
    assert test_input.ext[-1].coord.line_start == 15
    assert test_input.ext[-1].coord.filename == "example_c_file.c"

    assert isinstance(test_input.ext[-8], c_ast.Typedef)
    assert isinstance(test_input.ext[-8].type, c_ast.TypeDecl)
    assert test_input.ext[-8].name == "cookie_io_functions_t"
