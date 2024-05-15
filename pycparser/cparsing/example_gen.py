# ruff: noqa: T201
from __future__ import annotations

from collections import ChainMap

from pycparser.cparsing.ast_c import Case
from pycparser.cparsing.lexer_c import CLexer
from pycparser.cparsing.parser_c import CParser

if __name__ == "__main__":
    _ = Case

    text = """
typedef char TT;
void foo(int TT) {
    TT = 10;
}
"""

    print("==== Source ====")
    print(text, "\n")

    scope_stack: ChainMap[str, bool] = ChainMap()
    lexer = CLexer(scope_stack)
    parser = CParser(scope_stack)

    result = parser.parse(lexer.tokenize(text))

    print(result)
