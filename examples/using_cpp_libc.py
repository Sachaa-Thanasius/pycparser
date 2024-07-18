# -----------------------------------------------------------------
# pycparser: using_cpp_libc.py
#
# Shows how to use the provided 'cpp' (on Windows, substitute for
# the 'real' cpp if you're on Linux/Unix) and "fake" libc includes
# to parse a file that includes standard C headers.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -----------------------------------------------------------------
from cparsing import c_ast, parse_file


def main() -> None:
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("filename", default="examples/c_files/year.c")
    args = argparser.parse_args()
    filename: str = args.filename

    tree = parse_file(filename, use_cpp=True, cpp_path="cpp", cpp_args=r"-Iutils/fake_libc_include")
    print(c_ast.dump(tree))


if __name__ == "__main__":
    raise SystemExit(main())
