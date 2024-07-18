# ------------------------------------------------------------------------------
# pycparser: c-to-c.py
#
# Example of using pycparser.c_generator, serving as a simplistic translator
# from C to AST and back to C.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# ------------------------------------------------------------------------------
from cparsing import c_ast, parse_file


def translate_to_c(filename: str) -> None:
    """Simply use the c_generator module to emit a parsed AST."""

    tree = parse_file(filename, use_cpp=True)
    print(c_ast.unparse(tree))


def main() -> None:
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("filename")
    filename: str = argparser.parse_args().filename
    translate_to_c(filename)


if __name__ == "__main__":
    raise SystemExit(main())
