# -----------------------------------------------------------------
# pycparser: func_defs_add_param.py
#
# Example of rewriting AST nodes to add parameters to function
# definitions. Adds an "int _hidden" to every function.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -----------------------------------------------------------------
from cparsing import c_ast, parse


text = r"""
void foo(int a, int b) {
}

void bar() {
}
"""


class ParamAdder(c_ast.NodeVisitor):
    def visit_FuncDecl(self, node: c_ast.FuncDecl) -> None:
        ty = c_ast.TypeDecl(declname="_hidden", quals=[], align=[], type=c_ast.IdType(["int"]))
        newdecl = c_ast.Decl(
            name="_hidden",
            quals=[],
            align=[],
            storage=[],
            funcspec=[],
            type=ty,
            init=None,
            bitsize=None,
            coord=node.coord,
        )
        if node.args:
            node.args.params.append(newdecl)
        else:
            node.args = c_ast.ParamList(params=[newdecl])


def main() -> None:
    tree = parse(text)
    print("AST before change:")
    print(c_ast.dump(tree, indent=2))
    print()

    v = ParamAdder()
    v.visit(tree)

    print("AST after change:")
    print(c_ast.dump(tree, indent=2))
    print()

    print("Code after change:")
    print(c_ast.unparse(tree))


if __name__ == "__main__":
    raise SystemExit(main())
