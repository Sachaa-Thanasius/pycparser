from cparsing import c_ast, parse_file


class MyVisitor(c_ast.NodeVisitor):
    def visit_Typedef(self, node: c_ast.Typedef) -> None:
        print(f"typedef int {node.name};")


def generate_fake_typedefs(filename: str) -> None:
    ast = parse_file(filename, use_cpp=True)
    v = MyVisitor()
    v.visit(ast)


if __name__ == "__main__":
    raise SystemExit(generate_fake_typedefs("example_c_file_pp.c"))
