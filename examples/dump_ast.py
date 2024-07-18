# -----------------------------------------------------------------
# pycparser: dump_ast.py
#
# Basic example of parsing a file and dumping its parsed AST.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -----------------------------------------------------------------
import argparse

from cparsing import c_ast, parse_file


def main() -> None:
    argparser = argparse.ArgumentParser("Dump AST")
    argparser.add_argument("filename", default="examples/c_files/basic.c", nargs="?", help="name of file to parse")
    argparser.add_argument("--coord", help="show coordinates in the dump", action="store_true")
    args = argparser.parse_args()

    filename: str = args.filename
    show_coord: bool = args.coord

    tree = parse_file(filename, use_cpp=False)
    c_ast.dump(tree, include_coords=show_coord)


if __name__ == "__main__":
    raise SystemExit(main())
