# -----------------------------------------------------------------
# pycparser: construct_ast_from_scratch.py
#
# Tiny example of writing an AST from scratch to C code.
#
# Andre Ribeiro [https://github.com/Andree37]
# License: BSD
# -----------------------------------------------------------------
from cparsing import c_ast


# target C code:
# int main() {
#     return 0;
# }


def empty_main_function_ast() -> c_ast.FuncDef:
    constant_zero = c_ast.Constant(type="int", value="0")
    return_node = c_ast.Return(expr=constant_zero)
    compound_node = c_ast.Compound(block_items=[return_node])
    type_decl_node = c_ast.TypeDecl(declname="main", quals=[], type=c_ast.IdType(names=["int"]), align=[])
    func_decl_node = c_ast.FuncDecl(args=c_ast.ParamList([]), type=type_decl_node)
    func_def_node = c_ast.Decl(name="main", type=func_decl_node)
    main_func_node = c_ast.FuncDef(decl=func_def_node, param_decls=None, body=compound_node)

    return main_func_node  # noqa: RET504


def generate_c_code(my_ast: c_ast.AST) -> str:
    return c_ast.unparse(my_ast)


def main():
    main_function_ast = empty_main_function_ast()

    print(f'|{"-" * 40}|')
    print(c_ast.dump(main_function_ast, indent=4))
    print(f'|{"-" * 40}|')

    main_c_code = generate_c_code(main_function_ast)
    print(f"C code:\n{main_c_code}")


if __name__ == "__main__":
    raise SystemExit(main())
