# pyright: reportUndefinedVariable=none
# ruff: noqa: RUF012, F821
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, NoReturn

from pycparser.sly import Lexer
from pycparser.sly.lex import Token

if TYPE_CHECKING:
    from typing import Callable, Protocol, TypeVar, runtime_checkable

    CallableT = TypeVar('CallableT')

    @runtime_checkable
    class _RuleDecorator(Protocol):
        def __call__(self, rule: str, *extras: str) -> Callable[[CallableT], CallableT]: ...

    _ = object()
    assert isinstance(_, _RuleDecorator)


_line_pattern = re.compile(r'([ \t]*line\W)|([ \t]*\d+)')
_pragma_pattern = re.compile(r'[ \t]*pragma\W')

# ==== Regexes for use in tokens

_hex_prefix = '0[xX]'
_hex_digits = '[0-9a-fA-F]+'
_bin_prefix = '0[bB]'
_bin_digits = '[01]+'

# integer constants (K&R2: A.2.5.1)
_integer_suffix_opt = r'(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?'
_decimal_constant = '(0' + _integer_suffix_opt + ')|([1-9][0-9]*' + _integer_suffix_opt + ')'

# character constants (K&R2: A.2.5.2)
# Note: a-zA-Z and '.-~^_!=&;,' are allowed as escape chars to support #line
# directives with Windows paths as filenames (..\..\dir\file)
# For the same reason, decimal_escape allows all digit sequences. We want to
# parse all correct code, even if it means to sometimes parse incorrect
# code.
#
# The original regexes were taken verbatim from the C syntax definition,
# and were later modified to avoid worst-case exponential running time.
#
#   simple_escape = r"""([a-zA-Z._~!=&\^\-\\?'"])"""
#   decimal_escape = r"""(\d+)"""
#   hex_escape = r"""(x[0-9a-fA-F]+)"""
#   bad_escape = r"""([\\][^a-zA-Z._~^!=&\^\-\\?'"x0-7])"""
#
# The following modifications were made to avoid the ambiguity that allowed backtracking:
# (https://github.com/eliben/pycparser/issues/61)
#
# - \x was removed from simple_escape, unless it was not followed by a hex digit, to avoid ambiguity with hex_escape.
# - hex_escape allows one or more hex characters, but requires that the next character(if any) is not hex
# - decimal_escape allows one or more decimal characters, but requires that the next character(if any) is not a decimal
# - bad_escape does not allow any decimals (8-9), to avoid conflicting with the permissive decimal_escape.
#
# Without this change, python's `re` module would recursively try parsing each ambiguous escape sequence in multiple ways.
# e.g. `\123` could be parsed as `\1`+`23`, `\12`+`3`, and `\123`.
_simple_escape = r"""([a-wyzA-Z._~!=&\^\-\\?'"]|x(?![0-9a-fA-F]))"""
_decimal_escape = r"""(\d+)(?!\d)"""
_hex_escape = r"""(x[0-9a-fA-F]+)(?![0-9a-fA-F])"""
_bad_escape = r"""([\\][^a-zA-Z._~^!=&\^\-\\?'"x0-9])"""

_escape_sequence = r"""(\\(""" + _simple_escape + '|' + _decimal_escape + '|' + _hex_escape + '))'

# This complicated regex with lookahead might be slow for strings, so because all of the valid escapes (including \x) allowed
# 0 or more non-escaped characters after the first character, simple_escape+decimal_escape+hex_escape got simplified to
_escape_sequence_start_in_string = r"""(\\[0-9a-zA-Z._~!=&\^\-\\?'"])"""

_string_char = r"""([^"\\\n]|""" + _escape_sequence_start_in_string + ')'
_cconst_char = r"""([^'\\\n]|""" + _escape_sequence + ')'

# floating constants (K&R2: A.2.5.3)
_exponent_part = r"""([eE][-+]?[0-9]+)"""
_fractional_constant = r"""([0-9]*\.[0-9]+)|([0-9]+\.)"""
_binary_exponent_part = r'''([pP][+-]?[0-9]+)'''
_hex_fractional_constant = '(((' + _hex_digits + r""")?\.""" + _hex_digits + ')|(' + _hex_digits + r"""\.))"""


class CLexer(Lexer):
    def __init__(self, type_lookup_func: Callable[[str], object]):
        self.type_lookup_func: Callable[[str], object] = type_lookup_func
        self.filename: str = ''
        self.pp_line: str | None = None
        self.pp_filename: str | None = None

    # ==== Reserved keywords
    # fmt: off
    keywords: set[str] = {
        AUTO, BREAK, CASE, CHAR, CONST,
        CONTINUE, DEFAULT, DO, DOUBLE, ELSE, ENUM, EXTERN,
        FLOAT, FOR, GOTO, IF, INLINE, INT, LONG,
        REGISTER, OFFSETOF,
        RESTRICT, RETURN, SHORT, SIGNED, SIZEOF, STATIC, STRUCT,
        SWITCH, TYPEDEF, UNION, UNSIGNED, VOID,
        VOLATILE, WHILE,
        INT128,
    }

    AUTO        = 'auto'
    BREAK       = 'break'
    CASE        = 'case'
    CHAR        = 'char'
    CONST       = 'const'
    CONTINUE    = 'continue'
    DEFAULT     = 'default'
    DO          = 'do'
    DOUBLE      = 'double'
    ELSE        = 'else'
    ENUM        = 'enum'
    EXTERN      = 'extern'
    FLOAT       = 'float'
    FOR         = 'for'
    GOTO        = 'goto'
    IF          = 'if'
    INLINE      = 'intline'
    INT         = 'int'
    LONG        = 'long'
    REGISTER    = 'register'
    OFFSETOF    = 'offsetof'
    RESTRICT    = 'restrict'
    RETURN      = 'return'
    SHORT       = 'short'
    SIGNED      = 'signed'
    SIZEOF      = 'sizeof'
    STATIC      = 'static'
    STRUCT      = 'struct'
    SWITCH      = 'switch'
    TYPEDEF     = 'typedef'
    UNION       = 'union'
    UNSIGNED    = 'unsigned'
    VOID        = 'void'
    VOLATILE    = 'volatile'
    WHILE       = 'while'
    INT128      = '__int128'

    keywords_new: set[str] = {
        BOOL_, COMPLEX_,
        NORETURN_, THREAD_LOCAL_, STATIC_ASSERT_,
        ATOMIC_, ALIGNOF_, ALIGNAS_,
        PRAGMA_,
    }

    BOOL_           = '_Bool'
    COMPLEX_        = '_Complex'
    NORETURN_       = '_Noreturn'
    THREAD_LOCAL_   = '_Thread_local'
    STATIC_ASSERT_  = '_Static_assert'
    ATOMIC_         = '_Atomic'
    ALIGNOF_        = '_Alignof'
    ALIGNAS_        = '_Alignas'
    PRAGMA_         = '_Pragma'

    keyword_map: dict[str, str] = {str(keyword): keyword for keyword in keywords}
    keyword_map.update({str(keyword): keyword for keyword in keywords_new})

    tokens: set[str] = set(
        keywords | keywords_new | {
            # Identifiers
            ID,

            # Type identifiers (identifiers previously defined as types with typedef)
            TYPEID,

            # constants
            INT_CONST_DEC, INT_CONST_OCT, INT_CONST_HEX, INT_CONST_BIN, INT_CONST_CHAR,
            FLOAT_CONST, HEX_FLOAT_CONST,
            CHAR_CONST,
            WCHAR_CONST,
            U8CHAR_CONST,
            U16CHAR_CONST,
            U32CHAR_CONST,

            # String literals
            STRING_LITERAL,
            WSTRING_LITERAL,
            U8STRING_LITERAL,
            U16STRING_LITERAL,
            U32STRING_LITERAL,

            # Operators
            PLUS, MINUS, TIMES, DIVIDE, MOD,
            OR, AND, NOT, XOR, LSHIFT, RSHIFT,
            LOR, LAND, LNOT,
            LT, LE, GT, GE, EQ, NE,

            # Assignment
            EQUALS, TIMESEQUAL, DIVEQUAL, MODEQUAL,
            PLUSEQUAL, MINUSEQUAL,
            LSHIFTEQUAL,RSHIFTEQUAL, ANDEQUAL, XOREQUAL,
            OREQUAL,

            # Increment/decrement
            PLUSPLUS, MINUSMINUS,

            # Structure dereference (->)
            ARROW,

            # Conditional operator (?)
            CONDOP,

            # Ellipsis (...)
            ELLIPSIS,

            # Delimiters (excluding a few in literals below)
            LPAREN, RPAREN,         # ( )
            LBRACKET, RBRACKET,     # [ ]
            LBRACE, RBRACE,         # { }

            # pre-processor
            PPHASH,       # '#'
            PPPRAGMA,     # 'pragma'
            PPPRAGMASTR,
        }
    )

    # More delimiters. Might be expanded.
    literals = {
        # Operators
        # '=', '!', '<', '>', '+', '-', '*', '/', '%', '|', '&', '~', '^', '?',
        # Delimiters
        # '(', ')', '[', ']',
        ',', '.', ';', ':',
        # Scope delimiters
        # '{', '}',
    }
    # fmt: on

    ignore = ' \t'

    # ==== The rest of the tokens

    @_(r'[ \t]*\#')
    def PPHASH(self, t: Token) -> Token | None:
        if _line_pattern.match(self.text, pos=t.end):
            self.push_state(PPLineLexer)
            self.pp_line = None
            self.pp_filename = None
            return None

        if _pragma_pattern.match(self.text, pos=t.end):
            self.push_state(PPPragmaLexer)
            return None

        t.type = 'PPHASH'
        return t

    STRING_LITERAL = '"' + _string_char + '*"'

    FLOAT_CONST = '((((' + _fractional_constant + ')' + _exponent_part + '?)|([0-9]+' + _exponent_part + '))[FfLl]?)'
    HEX_FLOAT_CONST = (
        '('
        + _hex_prefix
        + '('
        + _hex_digits
        + '|'
        + _hex_fractional_constant
        + ')'
        + _binary_exponent_part
        + '[FfLl]?)'
    )
    INT_CONST_HEX = _hex_prefix + _hex_digits + _integer_suffix_opt
    INT_CONST_BIN = _bin_prefix + _bin_digits + _integer_suffix_opt

    @_('0[0-7]*[89]')
    def BAD_CONST_OCT(self, t: Token) -> None:
        self.error(t, 'Invalid octal constant')

    INT_CONST_OCT = '0[0-7]*' + _integer_suffix_opt
    INT_CONST_DEC = _decimal_constant

    INT_CONST_CHAR = "'" + _cconst_char + "{2,4}'"
    CHAR_CONST = "'" + _cconst_char + "'"
    WCHAR_CONST = 'L' + CHAR_CONST
    U8CHAR_CONST = 'u8' + CHAR_CONST
    U16CHAR_CONST = 'u' + CHAR_CONST
    U32CHAR_CONST = 'U' + CHAR_CONST

    @_("('" + _cconst_char + "*\\n)|('" + _cconst_char + "*$)")
    def UNMATCHED_QUOTE(self, t: Token) -> NoReturn:
        self.error(t, "Unmatched '")

    @_(r"""('""" + _cconst_char + """[^'\n]+')|('')|('""" + _bad_escape + r"""[^'\n]*')""")
    def BAD_CHAR_CONST(self, t: Token) -> NoReturn:
        self.error(t, f'Invalid char constant {t.value}')

    # string literals (K&R2: A.2.6)
    WSTRING_LITERAL = 'L' + STRING_LITERAL
    U8STRING_LITERAL = 'u8' + STRING_LITERAL
    U16STRING_LITERAL = 'u' + STRING_LITERAL
    U32STRING_LITERAL = 'U' + STRING_LITERAL

    @_('"' + _string_char + '*' + _bad_escape + _string_char + '*"')
    def BAD_STRING_LITERAL(self, t: Token) -> NoReturn:
        self.error(t, 'String contains invalid escape code')

    # Increment/decrement
    PLUSPLUS = r'\+\+'
    MINUSMINUS = r'--'

    # ->
    ARROW = r'->'

    # fmt: off
    # Assignment operators
    TIMESEQUAL  = r'\*='
    DIVEQUAL    = r'/='
    MODEQUAL    = r'%='
    PLUSEQUAL   = r'\+='
    MINUSEQUAL  = r'-='
    LSHIFTEQUAL = r'<<='
    RSHIFTEQUAL = r'>>='
    ANDEQUAL    = r'&='
    OREQUAL     = r'\|='
    XOREQUAL    = r'\^='

    # Operators
    LSHIFT      = r'<<'
    RSHIFT      = r'>>'
    LOR         = r'\|\|'
    LAND        = r'&&'

    LE          = r'<='
    GE          = r'>='
    EQ          = r'=='
    NE          = r'!='

    EQUALS      = r'='

    LNOT        = r'!'
    LT          = r'<'
    GT          = r'>'

    PLUS        = r'\+'
    MINUS       = r'-'
    TIMES       = r'\*'
    DIVIDE      = r'/'
    MOD         = r'%'
    OR          = r'\|'
    AND         = r'&'
    NOT         = r'~'
    XOR         = r'\^'

    # ?
    CONDOP      = r'\?'

    # Delimiters
    ELLIPSIS    = r'\.\.\.'
    LPAREN      = r'\('
    RPAREN      = r'\)'
    LBRACKET    = r'\['
    RBRACKET    = r'\]'

    # Scope delimiters
    LBRACE      = r'\{'
    RBRACE      = r'\}'
    # fmt: on

    @_(r'[a-zA-Z_$][0-9a-zA-Z_$]*')
    def ID(self, t: Token) -> Token:
        # valid C identifiers (K&R2: A.2.3), plus '$' (supported by some compilers)
        t.type = self.keyword_map.get(t.value, 'ID')
        if t.type == 'ID' and self.type_lookup_func(t.value):
            t.type = 'TYPEID'
        return t

    @_(r'\n+')
    def ignore_newline(self, t: Token) -> None:
        self.lineno += t.value.count('\n')

    def error(self, t: Token, msg: str | None = None) -> NoReturn:
        last_cr = self.text.rfind('\n', 0, t.index)
        if last_cr < 0:
            last_cr = 0
        column = (t.index - last_cr) + 1
        msg = msg or f'(Line, Column) {self.lineno}, {column}: Bad character {t.value[0]!r}'
        raise RuntimeError(msg, t.value, t.index)


class PPLineLexer(Lexer):
    def __init__(self):
        self.filename: str = ''
        self.pp_line: str | None = None
        self.pp_filename: str | None = None

    tokens: set[str] = {FILENAME, LINE_NUMBER, PPLINE}

    ignore = ' \t'

    @_('"' + _string_char + '*"')  # Same string as STRING_LITERAL.
    def FILENAME(self, t: Token) -> None:
        if self.pp_line is None:
            self.error(t, 'filename before line number in #line')
        else:
            self.pp_filename = t.value.lstrip('"').rstrip('"')

    @_(_decimal_constant)  # Same string as INT_DEC_CONST.
    def LINE_NUMBER(self, t: Token) -> None:
        if self.pp_line is None:
            self.pp_line = t.value
        else:
            # Ignore: GCC's cpp sometimes inserts a numeric flag
            # after the file name
            pass

    @_(r'line')
    def PPLINE(self, t: Token) -> None:
        pass

    @_(r'\n')
    def ignore_newline(self, t: Token) -> None:
        if self.pp_line is None:
            self.error(t, 'line number missing in #line')
        else:
            self.lineno = int(self.pp_line)

            if self.pp_filename is not None:
                self.filename = self.pp_filename

        self.pop_state()

    def error(self, t: Token, msg: str | None = None) -> NoReturn:
        msg = msg or f'invalid #line directive {t.value}'
        raise RuntimeError(msg, t)


class PPPragmaLexer(Lexer):
    def __init__(self):
        self.filename: str = ''
        self.pp_line: str | None = None
        self.pp_filename: str | None = None

    tokens: set[str] = {PPPRAGMA, STR}

    ignore = ' \t'

    PPPRAGMA = 'pragma'

    @_('.+')
    def STR(self, t: Token) -> Token:
        t.type = 'PPPRAGMASTR'
        return t

    @_(r'\n')
    def ignore_newline(self, t: Token) -> None:
        self.lineno += 1
        self.pop_state()

    def error(self, t: Token, msg: str | None = None) -> NoReturn:
        msg = msg or f'invalid #pragma directive {t.value}'
        raise RuntimeError(msg, t)
