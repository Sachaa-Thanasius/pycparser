from __future__ import annotations

from collections import ChainMap

from pycparser.sly.cparsing.clexer import CLexer
from pycparser.sly.cparsing.cparser import CParser

if __name__ == "__main__":
    text = """
typedef char TT;
void foo(int TT) { TT = 10; }
"""
    scope_stack: ChainMap[str, bool] = ChainMap()
    lexer = CLexer(scope_stack)
    parser = CParser(scope_stack)

    tokens = list(lexer.tokenize(text))
    for tok in tokens:
        print(f"type={tok.type!r}, value={tok.value!r}")

    result = parser.parse(iter(tokens))

    print(result)
