# -----------------------------------------------------------------
# pycparser: rewrite_ast.py
#
# Tiny example of rewriting a AST node
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -----------------------------------------------------------------
from cparsing import c_ast, parse


text = r"""
void func(void)
{
    x = 1;
}
"""


def main() -> None:
    tree = parse(text)
    print("Before:")
    print(c_ast.dump(tree, indent=2))

    assign = tree.ext[0].body.block_items[0]
    assign.lvalue.name = "y"
    assign.rvalue.value = 2

    print("After:")
    print(c_ast.dump(tree, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
