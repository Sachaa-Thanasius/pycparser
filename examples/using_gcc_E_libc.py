# -------------------------------------------------------------------------------
# pycparser: using_gcc_E_libc.py
#
# Similar to the using_cpp_libc.py example, but uses 'gcc -E' instead
# of 'cpp'. The same can be achieved with Clang instead of gcc. If you have
# Clang installed, simply replace 'gcc' with 'clang' here.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -------------------------------------------------------------------------------
from cparsing import c_ast, parse_file


def main() -> None:
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("filename", default="examples/c_files/year.c")
    args = argparser.parse_args()
    filename: str = args.filename

    ast = parse_file(filename, use_cpp=True, cpp_path="gcc", cpp_args=["-E", r"-Iutils/fake_libc_include"])
    print(c_ast.dump(ast))


if __name__ == "__main__":
    raise SystemExit(main())
