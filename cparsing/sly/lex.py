# -----------------------------------------------------------------------------
# sly: lex.py
#
# Copyright (C) 2016 - 2018
# David M. Beazley (Dabeaz LLC)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the David Beazley or Dabeaz LLC may be used to
#   endorse or promote products derived from this software without
#  specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self
else:

    class Self:
        """Placeholder for `typing.Self`."""


_CallableT = TypeVar('_CallableT', bound=Callable[..., Any])

_MISSING: Any = object()

__all__ = ('Lexer', 'LexerStateChange')


class LexError(Exception):
    '''
    Exception raised if an invalid character is encountered and no default
    error handler function is defined.  The .text attribute of the exception
    contains all remaining untokenized text. The .error_index is the index
    location of the error.
    '''

    def __init__(self, message: str, text: str, error_index: int):
        self.args = (message,)
        self.text = text
        self.error_index = error_index


class PatternError(Exception):
    '''
    Exception raised if there's some kind of problem with the specified
    regex patterns in the lexer.
    '''


class LexerBuildError(Exception):
    '''
    Exception raised if there's some sort of problem building the lexer.
    '''


class LexerStateChange(Exception):
    '''
    Exception raised to force a lexing state change
    '''

    def __init__(self, newstate: type[Lexer], tok: Token | None = None):
        self.newstate = newstate
        self.tok = tok


class Token:
    '''
    Representation of a single token.
    '''

    __slots__ = ('type', 'value', 'lineno', 'index', 'end')

    if TYPE_CHECKING:
        type: str
        value: str
        lineno: int
        index: int
        end: int

    def __repr__(self):
        return (
            f'Token(type={self.type!r}, value={self.value!r}, lineno={self.lineno}, index={self.index}, end={self.end})'
        )


class TokenStr(str):
    def __new__(cls, value: object, key: str, remap: dict[tuple[str, Any], Any] | None = None):
        return super().__new__(cls, value)

    def __init__(self, value: object, key: str, remap: dict[tuple[str, Any], Any] | None = None):
        self.key = key
        self.remap = remap

    # Implementation of TOKEN[value] = NEWTOKEN
    def __setitem__(self, key: str, value: str):
        if self.remap is not None:
            self.remap[self.key, key] = value

    # Implementation of del TOKEN[value]
    def __delitem__(self, key: str):
        if self.remap is not None:
            self.remap[self.key, key] = self.key


class _Before:
    def __init__(self, tok: str, pattern: str):
        self.tok = tok
        self.pattern = pattern


class LexerMetaDict(Dict[str, Any]):
    '''
    Special dictionary that prohibits duplicate definitions in lexer specifications.
    '''

    def __init__(self):
        self.before: dict[str, Any] = {}
        self.delete: list[str] = []
        self.remap: dict[tuple[str, Any], Any] = {}

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, str):
            value = TokenStr(value, key, self.remap)

        if isinstance(value, _Before):
            self.before[key] = value.tok
            value = TokenStr(value.pattern, key, self.remap)

        if key in self and not isinstance(value, property):
            prior = self[key]
            if isinstance(prior, str):
                if callable(value):
                    value.pattern = prior
                else:
                    raise AttributeError(f'Name {key} redefined')  # noqa: TRY004

        super().__setitem__(key, value)

    def __delitem__(self, key: str):
        self.delete.append(key)
        if key not in self and key.isupper():
            return None

        return super().__delitem__(key)

    def __missing__(self, key: str) -> TokenStr:
        if key.split('ignore_')[-1].isupper() and key[:1] != '_':
            return TokenStr(key, key, self.remap)
        raise KeyError


class LexerMeta(type):
    '''
    Metaclass for collecting lexing rules
    '''

    if TYPE_CHECKING:
        _rules: list[tuple[str, Any]]

    @classmethod
    def __prepare__(cls, name: str, bases: tuple[type, ...], **kwds: object) -> LexerMetaDict:
        d = LexerMetaDict()

        def _(pattern: str, *extra: str) -> Callable[[_CallableT], _CallableT]:
            patterns = [pattern, *extra]

            def decorate(func: _CallableT) -> _CallableT:
                pattern = '|'.join(f'({pat})' for pat in patterns)
                if hasattr(func, 'pattern'):
                    func.pattern = f'{pattern}|{func.pattern}'  # type: ignore # Runtime attribute access and assignment.
                else:
                    func.pattern = pattern  # type: ignore # Runtime attribute assignment.
                return func

            return decorate

        d['_'] = _
        d['before'] = _Before
        return d

    def __new__(cls, clsname: str, bases: tuple[type, ...], attributes: LexerMetaDict, **kwds: object):
        del attributes['_']
        del attributes['before']

        # Create attributes for use in the actual class body
        cls_attributes = {str(key): str(val) if isinstance(val, TokenStr) else val for key, val in attributes.items()}
        self: type[Lexer] = super().__new__(cls, clsname, bases, cls_attributes, **kwds)

        # Attach various metadata to the class
        self._attributes = dict(attributes)
        self._remap = attributes.remap
        self._before = attributes.before
        self._delete = attributes.delete
        self._build()
        return self


class Lexer(metaclass=LexerMeta):
    # These attributes may be defined in subclasses
    tokens: ClassVar[set[str]] = set()
    literals: ClassVar[set[str]] = set()
    ignore: str = ''
    reflags: int = 0
    regex_module = re

    _token_names: ClassVar[set[str]] = set()
    _token_funcs: ClassVar[dict[str, Callable[[Lexer, Token], Any]]] = {}
    _ignored_tokens: ClassVar[set[str]] = set()
    _remapping: ClassVar[dict[str, dict[str, str]]] = {}
    _delete: ClassVar[list[str]] = []
    _remap: ClassVar[dict[tuple[str, Any], Any]] = {}

    # Internal attributes
    __state_stack: list[type[Lexer]] | None = None
    __set_state: Callable[[type[Lexer]], None] | None = None

    if TYPE_CHECKING:
        _attributes: ClassVar[dict[str, Any]]
        _before: ClassVar[dict[str, str]]

    @classmethod
    def _collect_rules(cls) -> None:
        # Collect all of the rules from class definitions that look like token
        # information.   There are a few things that govern this:
        #
        # 1.  Any definition of the form NAME = str is a token if NAME is
        #     is defined in the tokens set.
        #
        # 2.  Any definition of the form ignore_NAME = str is a rule for an ignored
        #     token.
        #
        # 3.  Any function defined with a 'pattern' attribute is treated as a rule.
        #     Such functions can be created with the @_ decorator or by defining
        #     function with the same name as a previously defined string.
        #
        # This function is responsible for keeping rules in order.

        # Collect all previous rules from base classes
        rules: list[tuple[str, Any]] = []

        for base in cls.__bases__:
            if isinstance(base, LexerMeta):
                rules.extend(base._rules)

        # Dictionary of previous rules
        existing = dict(rules)

        for key, value in cls._attributes.items():
            if (key in cls._token_names) or key.startswith('ignore_') or hasattr(value, 'pattern'):
                if callable(value) and not hasattr(value, 'pattern'):
                    raise LexerBuildError(f"function {value} doesn't have a regex pattern")

                if key in existing:
                    # The definition matches something that already existed in the base class.
                    # We replace it, but keep the original ordering
                    n = rules.index((key, existing[key]))
                    rules[n] = (key, value)
                    existing[key] = value

                elif isinstance(value, TokenStr) and key in cls._before:
                    before = cls._before[key]
                    if before in existing:
                        # Position the token before another specified token
                        n = rules.index((before, existing[before]))
                        rules.insert(n, (key, value))
                    else:
                        # Put at the end of the rule list
                        rules.append((key, value))
                    existing[key] = value
                else:
                    rules.append((key, value))
                    existing[key] = value

            elif isinstance(value, str) and not key.startswith('_') and key not in {'ignore', 'literals'}:
                raise LexerBuildError(f'{key} does not match a name in tokens')

        # Apply deletion rules
        rules = [(key, value) for key, value in rules if key not in cls._delete]
        cls._rules = rules

    @classmethod
    def _build(cls) -> None:
        '''
        Build the lexer object from the collected tokens and regular expressions.
        Validate the rules to make sure they look sane.
        '''

        if 'tokens' not in vars(cls):
            raise LexerBuildError(f'{cls.__qualname__} class does not define a tokens attribute')

        # Pull definitions created for any parent classes
        cls._token_names = cls._token_names | set(cls.tokens)
        cls._ignored_tokens = set(cls._ignored_tokens)
        cls._token_funcs = dict(cls._token_funcs)
        cls._remapping = dict(cls._remapping)

        for (key, val), newtok in cls._remap.items():
            if key not in cls._remapping:
                cls._remapping[key] = {}
            cls._remapping[key][val] = newtok

        remapped_toks: set[str] = set()
        for d in cls._remapping.values():
            remapped_toks.update(d.values())

        undefined = remapped_toks - set(cls._token_names)
        if undefined:
            missing = ', '.join(undefined)
            raise LexerBuildError(f'{missing} not included in token(s)')

        cls._collect_rules()

        parts: list[str] = []
        for tokname, value in cls._rules:
            if tokname.startswith('ignore_'):
                tokname = tokname[7:]  # noqa: PLW2901
                cls._ignored_tokens.add(tokname)

            if isinstance(value, str):
                pattern = value

            elif callable(value):
                cls._token_funcs[tokname] = value
                pattern = value.pattern

            # Form the regular expression component
            part = f'(?P<{tokname}>{pattern})'

            # Make sure the individual regex compiles properly
            try:
                cpat = cls.regex_module.compile(part, cls.reflags)
            except Exception as e:
                raise PatternError(f'Invalid regex for token {tokname}') from e

            # Verify that the pattern doesn't match the empty string
            if cpat.match(''):
                raise PatternError(f'Regex for token {tokname} matches empty input')

            parts.append(part)

        if not parts:
            return

        # Form the master regular expression
        # previous = ('|' + cls._master_re.pattern) if cls._master_re else ''
        # cls._master_re = cls.regex_module.compile('|'.join(parts) + previous, cls.reflags)
        cls._master_re = cls.regex_module.compile('|'.join(parts), cls.reflags)

        # Verify that that ignore and literals specifiers match the input type
        if not isinstance(cls.ignore, str):
            raise LexerBuildError('ignore specifier must be a string')

        if not all(isinstance(lit, str) for lit in cls.literals):
            raise LexerBuildError('literals must be specified as strings')

    def begin(self, cls: type[Lexer]) -> None:
        '''
        Begin a new lexer state
        '''

        if not isinstance(cls, LexerMeta):
            raise TypeError('state must be a subclass of Lexer')

        if self.__set_state:
            self.__set_state(cls)
        self.__class__ = cls

    def push_state(self, cls: type[Lexer]) -> None:
        '''
        Push a new lexer state onto the stack
        '''

        if self.__state_stack is None:
            self.__state_stack = []
        self.__state_stack.append(type(self))
        self.begin(cls)

    def pop_state(self) -> None:
        '''
        Pop a lexer state from the stack
        '''

        assert self.__state_stack
        self.begin(self.__state_stack.pop())

    def tokenize(self, text: str, lineno: int = 1, index: int = 0) -> Generator[Token, Any, None]:
        _ignored_tokens: set[str] = _MISSING
        _master_re: re.Pattern[str] = _MISSING
        _ignore: str = _MISSING
        _token_funcs: dict[str, Callable[[Lexer, Token], Any]] = _MISSING
        _literals: set[str] = _MISSING
        _remapping: dict[str, dict[str, str]] = _MISSING

        # --- Support for state changes
        def _set_state(cls: type[Lexer]) -> None:
            nonlocal _ignored_tokens, _master_re, _ignore, _token_funcs, _literals, _remapping
            _ignored_tokens = cls._ignored_tokens
            _master_re = cls._master_re
            _ignore = cls.ignore
            _token_funcs = cls._token_funcs
            _literals = cls.literals
            _remapping = cls._remapping

        self.__set_state = _set_state
        _set_state(type(self))

        # --- Support for backtracking
        _mark_stack: list[tuple[type[Self], int, int]] = []

        def _mark() -> None:
            _mark_stack.append((type(self), index, lineno))

        self.mark = _mark

        def _accept() -> None:
            _mark_stack.pop()

        self.accept = _accept

        def _reject() -> None:
            nonlocal index, lineno
            cls, index, lineno = _mark_stack[-1]
            _set_state(cls)

        self.reject = _reject

        # --- Main tokenization function
        self.text = text
        try:
            while True:
                try:
                    if text[index] in _ignore:
                        index += 1
                        continue
                except IndexError:
                    return

                tok = Token()
                tok.lineno = lineno
                tok.index = index
                m = _master_re.match(text, index)
                if m:
                    tok.end = index = m.end()
                    tok.value = m.group()
                    tok.type = m.lastgroup

                    if tok.type in _remapping:
                        tok.type = _remapping[tok.type].get(tok.value, tok.type)

                    if tok.type in _token_funcs:
                        self.index = index
                        self.lineno = lineno
                        tok = _token_funcs[tok.type](self, tok)
                        index = self.index
                        lineno = self.lineno
                        if not tok:
                            continue

                    if tok.type in _ignored_tokens:
                        continue

                    yield tok

                else:
                    # No match, see if the character is in literals
                    if text[index] in _literals:
                        tok.value = text[index]
                        tok.end = index + 1
                        tok.type = tok.value
                        index += 1
                        yield tok
                    else:
                        # A lexing error
                        self.index = index
                        self.lineno = lineno
                        tok.type = 'ERROR'
                        tok.value = text[index:]
                        tok = self.error(tok)
                        if tok is not None:
                            tok.end = self.index
                            yield tok

                        index = self.index
                        lineno = self.lineno

        # Set the final state of the lexer before exiting (even if exception)
        finally:
            self.text = text
            self.index = index
            self.lineno = lineno

    # Default implementations of the error handler. May be changed in subclasses
    def error(self, t: Token) -> Any:
        raise LexError(f'Illegal character {t.value[0]!r} at index {self.index}', t.value, self.index)