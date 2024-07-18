# -----------------------------------------------------------------
# pycparser: serialize_ast.py
#
# Simple example of serializing AST
#
# Hart Chu [https://github.com/CtheSky]
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -----------------------------------------------------------------
import pickle

from cparsing import c_ast, parse


text = r"""
void func(void)
{
  x = 1;
}
"""


def main() -> None:
    tree = parse(text)
    dump_filename = "ast.pickle"

    with open(dump_filename, "wb") as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Deserialize.
    with open(dump_filename, "rb") as f:
        tree: c_ast.File = pickle.load(f)  # noqa: S301
        print(c_ast.dump(tree, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
