# -----------------------------------------------------------------
# pycparser: func_calls.py
#
# Using pycparser for printing out all the calls of some function
# in a C file.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -----------------------------------------------------------------
from cparsing import c_ast, parse_file


class FuncCallVisitor(c_ast.NodeVisitor):
    """A visitor with some state information, namely the funcname it's looking for."""

    def __init__(self, funcname: str) -> None:
        self.funcname = funcname

    def visit_FuncCall(self, node: c_ast.FuncCall) -> None:
        if node.name.name == self.funcname:
            print(f"{self.funcname} called at {node.name.coord}")
        # Visit args in case they contain more func calls.
        if node.args:
            self.generic_visit(node)


def show_func_calls(filename: str, funcname: str) -> None:
    tree = parse_file(filename, use_cpp=True)
    v = FuncCallVisitor(funcname)
    v.visit(tree)


def main() -> None:
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("filename", default="examples/c_files/basic.c")
    argparser.add_argument("func", default="foo")
    args = argparser.parse_args()

    show_func_calls(args.filename, args.func)


if __name__ == "__main__":
    raise SystemExit(main())
