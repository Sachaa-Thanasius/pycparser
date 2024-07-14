"""Test `c_lexer.CLexer`."""

import pytest
from cparsing.c_context import CContext
from cparsing.c_lexer import CLexer
from sly.lex import Token


@pytest.fixture
def clex() -> CLexer:
    ctx = CContext()
    ctx.scope_stack["mytype"] = True
    return ctx.lexer


def do_lex(lexer: CLexer, inp: str) -> list[str]:
    return [tok.type for tok in lexer.tokenize(inp)]


def match_token(token: Token, **attrs: object) -> bool:
    return all(getattr(token, name) == expected_val for name, expected_val in attrs.items())


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("1", ["INT_CONST_DEC"]),
        ("-", ["MINUS"]),
        ("volatile", ["VOLATILE"]),
        ("...", ["ELLIPSIS"]),
        ("++", ["PLUSPLUS"]),
        ("case int", ["CASE", "INT"]),
        ("caseint", ["ID"]),
        ("$dollar cent$", ["ID", "ID"]),
        ("i ^= 1;", ["ID", "XOREQUAL", "INT_CONST_DEC", ";"]),
    ],
)
def test_trivial_tokens(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("myt", ["ID"]),
        ("mytype", ["TYPEID"]),
        ("mytype var", ["TYPEID", "ID"]),
    ],
)
def test_id_typeid(clex: CLexer, test_input: str, expected: list[str]):
    # NOTE: This test assumes {'mytype': True} is in the scope stack. See the clex fixture.
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("12", ["INT_CONST_DEC"]),
        ("12u", ["INT_CONST_DEC"]),
        ("12l", ["INT_CONST_DEC"]),
        ("199872Ul", ["INT_CONST_DEC"]),
        ("199872lU", ["INT_CONST_DEC"]),
        ("199872LL", ["INT_CONST_DEC"]),
        ("199872ull", ["INT_CONST_DEC"]),
        ("199872llu", ["INT_CONST_DEC"]),
        ("1009843200000uLL", ["INT_CONST_DEC"]),
        ("1009843200000LLu", ["INT_CONST_DEC"]),
        ("077", ["INT_CONST_OCT"]),
        ("0123456L", ["INT_CONST_OCT"]),
        ("0xf7", ["INT_CONST_HEX"]),
        ("0b110", ["INT_CONST_BIN"]),
        ("0x01202AAbbf7Ul", ["INT_CONST_HEX"]),
        ("'12'", ["INT_CONST_CHAR"]),
        ("'123'", ["INT_CONST_CHAR"]),
        ("'1AB4'", ["INT_CONST_CHAR"]),
        (r"'1A\n4'", ["INT_CONST_CHAR"]),
        pytest.param("xf7", ["ID"], id="no 0 before x, so ID catches it"),
        pytest.param("-1", ["MINUS", "INT_CONST_DEC"], id="- is MINUS, the rest a constant"),
    ],
)
def test_integer_constants(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(("test_input", "expected"), [("sizeof offsetof", ["SIZEOF", "OFFSETOF"])])
def test_special_names(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("_Bool", ["BOOL_"]),
        ("_Atomic", ["ATOMIC_"]),
        ("_Alignas _Alignof", ["ALIGNAS_", "ALIGNOF_"]),
    ],
)
def test_new_keywords(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("1.5f", ["FLOAT_CONST"]),
        ("01.5", ["FLOAT_CONST"]),
        (".15L", ["FLOAT_CONST"]),
        ("0.", ["FLOAT_CONST"]),
        pytest.param(".", ["."], id="a period is just a period"),
        ("3.3e-3", ["FLOAT_CONST"]),
        (".7e25L", ["FLOAT_CONST"]),
        ("6.e+125f", ["FLOAT_CONST"]),
        ("666e666", ["FLOAT_CONST"]),
        ("00666e+3", ["FLOAT_CONST"]),
        pytest.param("0x0666e+3", ["INT_CONST_HEX", "PLUS", "INT_CONST_DEC"], id="hex integer + 3"),
    ],
)
def test_floating_constants(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("0xDE.488641p0", ["HEX_FLOAT_CONST"]),
        ("0x.488641p0", ["HEX_FLOAT_CONST"]),
        ("0X12.P0", ["HEX_FLOAT_CONST"]),
    ],
)
def test_hexadecimal_floating_constants(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (r"""'x'""", ["CHAR_CONST"]),
        (r"""L'x'""", ["WCHAR_CONST"]),
        (r"""u8'x'""", ["U8CHAR_CONST"]),
        (r"""u'x'""", ["U16CHAR_CONST"]),
        (r"""U'x'""", ["U32CHAR_CONST"]),
        (r"""'\t'""", ["CHAR_CONST"]),
        (r"""'\''""", ["CHAR_CONST"]),
        (r"""'\?'""", ["CHAR_CONST"]),
        (r"""'\0'""", ["CHAR_CONST"]),
        (r"""'\012'""", ["CHAR_CONST"]),
        (r"""'\x2f'""", ["CHAR_CONST"]),
        (r"""'\x2f12'""", ["CHAR_CONST"]),
        (r"""L'\xaf'""", ["WCHAR_CONST"]),
    ],
)
def test_char_constants(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ('"a string"', ["STRING_LITERAL"]),
        ('L"ing"', ["WSTRING_LITERAL"]),
        ('u8"ing"', ["U8STRING_LITERAL"]),
        ('u"ing"', ["U16STRING_LITERAL"]),
        ('U"ing"', ["U32STRING_LITERAL"]),
        ('"i am a string too \t"', ["STRING_LITERAL"]),
        (r'''"esc\ape \"\'\? \0234 chars \rule"''', ["STRING_LITERAL"]),
        (r'''"hello 'joe' wanna give it a \"go\"?"''', ["STRING_LITERAL"]),
        ('"\123\123\123\123\123\123\123\123\123\123\123\123\123\123\123\123"', ["STRING_LITERAL"]),
        # Note: a-zA-Z and '.-~^_!=&;,' are allowed as escape chars to support #line directives with Windows paths
        # as filenames (..\..\dir\file)
        (r'"\x"', ["STRING_LITERAL"]),
        (
            r'"\a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z\A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z"',
            ["STRING_LITERAL"],
        ),
        (r'"C:\x\fa\x1e\xited"', ["STRING_LITERAL"]),
        pytest.param(r'"jx\9"', ["STRING_LITERAL"], id="permissive lexer allow decimal escapes (not just octal); 1"),
        pytest.param(
            r'"fo\9999999"', ["STRING_LITERAL"], id="permissive lexer allow decimal escapes (not just octal); 1"
        ),
    ],
)
def test_string_literal(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (r"[{}]()", ["[", "{", "}", "]", "(", ")"]),
        (r"()||!C&~Z?J", ["(", ")", "LOR", "LNOT", "ID", "AND", "NOT", "ID", "CONDOP", "ID"]),
        (
            r"+-*/%|||&&&^><>=<===!=",
            [
                "PLUS",
                "MINUS",
                "TIMES",
                "DIVIDE",
                "MOD",
                "LOR",
                "OR",
                "LAND",
                "AND",
                "XOR",
                "GT",
                "LT",
                "GE",
                "LE",
                "EQ",
                "NE",
            ],
        ),
        (r"++--->?.,;:", ["PLUSPLUS", "MINUSMINUS", "ARROW", "CONDOP", ".", ",", ";", ":"]),
    ],
)
def test_mess(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("bb-cc", ["ID", "MINUS", "ID"]),
        ("foo & 0xFF", ["ID", "AND", "INT_CONST_HEX"]),
        (
            "(2+k) * 62",
            ["(", "INT_CONST_DEC", "PLUS", "ID", ")", "TIMES", "INT_CONST_DEC"],
        ),
        ("x | y >> z", ["ID", "OR", "ID", "RSHIFT", "ID"]),
        ("x <<= z << 5", ["ID", "LSHIFTEQUAL", "ID", "LSHIFT", "INT_CONST_DEC"]),
        (
            "x = y > 0 ? y : -6",
            ["ID", "EQUALS", "ID", "GT", "INT_CONST_OCT", "CONDOP", "ID", ":", "MINUS", "INT_CONST_DEC"],
        ),
        ("a+++b", ["ID", "PLUSPLUS", "PLUS", "ID"]),
    ],
)
def test_exprs(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "for (int i = 0; i < n; ++i)",
            [
                "FOR",
                "(",
                "INT",
                "ID",
                "EQUALS",
                "INT_CONST_OCT",
                ";",
                "ID",
                "LT",
                "ID",
                ";",
                "PLUSPLUS",
                "ID",
                ")",
            ],
        ),
        ("self: goto self;", ["ID", ":", "GOTO", "ID", ";"]),
        (
            "switch (typ)\n"
            "{\n"
            "    case TYPE_ID:\n"
            "        m = 5;\n"
            "        break;\n"
            "    default:\n"
            "        m = 8;\n"
            "}\n",
            [
                "SWITCH",
                "(",
                "ID",
                ")",
                "{",
                "CASE",
                "ID",
                ":",
                "ID",
                "EQUALS",
                "INT_CONST_DEC",
                ";",
                "BREAK",
                ";",
                "DEFAULT",
                ":",
                "ID",
                "EQUALS",
                "INT_CONST_DEC",
                ";",
                "}",
            ],
        ),
    ],
)
def test_statements(clex: CLexer, test_input: str, expected: list[str]):
    assert do_lex(clex, test_input) == expected


def test_preprocessor_line(clex: CLexer):
    assert do_lex(clex, "#abracadabra") == ["PP_HASH", "ID"]

    test_input = r"""
546
#line 66 "kwas\df.h"
id 4
dsf
# 9
armo
#line 10 "..\~..\test.h"
tok1
#line 99999 "include/me.h"
tok2
"""

    tokenizer = clex.tokenize(test_input)

    t1 = next(tokenizer)
    assert match_token(t1, type="INT_CONST_DEC", lineno=2)

    t2 = next(tokenizer)
    assert match_token(t2, type="ID", value="id", lineno=66)
    assert clex.ctx.filename == r"kwas\df.h"

    _ = next(tokenizer)
    _ = next(tokenizer)

    t3 = next(tokenizer)
    assert match_token(t3, type="ID", value="armo", lineno=9)
    assert clex.ctx.filename == r"kwas\df.h"

    t4 = next(tokenizer)
    assert match_token(t4, type="ID", value="tok1", lineno=10)
    assert clex.ctx.filename == r"..\~..\test.h"

    t5 = next(tokenizer)
    assert match_token(t5, type="ID", value="tok2", lineno=99999)
    assert clex.ctx.filename == r"include/me.h"


def test_preprocessor_line_funny(clex: CLexer):
    test_input = r"""
#line 10 "..\6\joe.h"
10
"""

    tokenizer = clex.tokenize(test_input)

    t1 = next(tokenizer)
    assert match_token(t1, type="INT_CONST_DEC", lineno=10)
    assert clex.ctx.filename == r"..\6\joe.h"


def test_preprocessor_pragma(clex: CLexer):
    """Test that pragmas are tokenized, including trailing string."""

    test_input = """
42
#pragma
#pragma helo me
#pragma once
# pragma omp parallel private(th_id)
#\tpragma {pack: 2, smack: 3}
#pragma <includeme.h> "nowit.h"
#pragma "string"
#pragma somestring="some_other_string"
#pragma id 124124 and numbers 0235495
_Pragma("something else")
59
"""

    tokenizer = clex.tokenize(test_input)

    t1 = next(tokenizer)
    assert t1.type == "INT_CONST_DEC"

    t2 = next(tokenizer)
    assert t2.type == "PP_PRAGMA"

    t3 = next(tokenizer)
    assert t3.type == "PP_PRAGMA"

    t4 = next(tokenizer)
    assert match_token(t4, type="PP_PRAGMASTR", value="helo me")

    _ = next(tokenizer)
    _ = next(tokenizer)
    _ = next(tokenizer)

    t5 = next(tokenizer)
    assert match_token(t5, type="PP_PRAGMASTR", value="omp parallel private(th_id)")

    for _ in range(5):
        ta = next(tokenizer)
        assert ta.type == "PP_PRAGMA"

        tb = next(tokenizer)
        assert tb.type == "PP_PRAGMASTR"

    t6a = next(tokenizer)
    assert t6a.type == "PRAGMA_"

    t6l = next(tokenizer)
    assert t6l.type == "("

    t6b = next(tokenizer)
    assert match_token(t6b, type="STRING_LITERAL", value='"something else"')

    t6r = next(tokenizer)
    assert t6r.type == ")"

    t7 = next(tokenizer)
    assert match_token(t7, type="INT_CONST_DEC", lineno=13)
