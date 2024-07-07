from __future__ import annotations

from collections import ChainMap
from typing import Any

from pycparser.c_parser import CParser as OGParser

from cparsing import c_ast
from cparsing.c_lexer import CLexer
from cparsing.c_parser import CParseError, CParser

source = """\
void foo(int TT){
    TT = 10;
}
"""

# Cases:
# 1.
# char myvar[] = "hello";
#
# 2.
# void foo(int TT){
#     TT = 10;
# }
#
# 3.
# typedef char TT;
# void foo(int TT){
#     TT = 10;
# }


def check_og(source: str) -> Any:
    print("-- Parsing with OG...\n")
    parser = OGParser()
    return parser.parse(source)


def check_new(source: str) -> Any:
    print("-- Parsing with New...\n")
    scope_stack: ChainMap[str, bool] = ChainMap()
    lexer = CLexer(scope_stack)
    parser = CParser(scope_stack)

    result = parser.parse(lexer.tokenize(source))
    return c_ast.dump(result, indent=" " * 4)


def check_diff(first: str, second: str) -> None:
    import difflib

    diff = difflib.Differ()

    first_lines = str(first).splitlines(keepends=True)
    second_lines = str(second).splitlines(keepends=True)

    result = list(diff.compare(first_lines, second_lines))
    print("==== Diff ====")
    print("".join(result))


if __name__ == "__main__":
    print("==== Source ====")
    print(source, "\n")

    og_result = check_og(source)

    try:
        new_result = check_new(source)
    except CParseError:
        print(f"---- og_result ----\n{og_result}")
        raise

    print(new_result)
    check_diff(og_result, new_result)
