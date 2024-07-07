"""Shim for typing-related names that may have different sources or not exist at runtime."""

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if sys.version_info >= (3, 9, 2):  # noqa: UP036 # Users might still be on 3.9.0.
    from types import GenericAlias as _GenericAlias
elif TYPE_CHECKING:

    class _GenericAlias:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

else:
    from typing import _GenericAlias


__all__ = ("NotRequired", "Self", "TypeGuard", "TypeAlias")


class _PlaceholderGenericAlias(_GenericAlias):
    def __repr__(self) -> str:
        return f"<placeholder for {super().__repr__()}>"


class _PlaceholderMeta(type):
    def __getitem__(self, item: object) -> _PlaceholderGenericAlias:
        return _PlaceholderGenericAlias(self, item)

    def __repr__(self) -> str:
        return f"<placeholder for {super().__repr__()}>"


if sys.version_info >= (3, 11):
    from typing import NotRequired, Self
elif TYPE_CHECKING:
    from typing_extensions import NotRequired, Self
else:

    class NotRequired(metaclass=_PlaceholderMeta):
        pass

    class Self(metaclass=_PlaceholderMeta):
        pass


if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeGuard
elif TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeGuard
else:

    class TypeGuard(metaclass=_PlaceholderMeta):
        pass

    class TypeAlias(metaclass=_PlaceholderMeta):
        pass


CallableT = TypeVar("CallableT", bound=Callable[..., Any])

if TYPE_CHECKING:
    from typing import Protocol, cast, type_check_only

    @type_check_only
    class _RuleDecorator(Protocol):
        def __call__(self, rule: str, *extras: str) -> Callable[[CallableT], CallableT]: ...

    # Typing hack to account for `_` existing in a sly.Lexer or sly.Parser class's namespace only during class
    # creation. Should only ever be imported within an `if TYPE_CHECKING` block.
    _ = cast(_RuleDecorator, object())
