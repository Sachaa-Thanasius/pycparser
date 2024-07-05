import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 9, 2):
    from types import GenericAlias as _GenericAlias
elif TYPE_CHECKING:

    class _GenericAlias:
        def __init__(self, *args: object, **kwargs: object):
            pass

else:
    from typing import _GenericAlias


__all__ = ("Self", "TypeGuard", "TypeAlias")


class _PlaceholderGenericAlias(_GenericAlias):
    def __repr__(self):
        return f"<placeholder for {super().__repr__()}>"


class _PlaceholderMeta(type):
    def __getitem__(self, item: object) -> _PlaceholderGenericAlias:
        return _PlaceholderGenericAlias(self, item)

    def __repr__(self):
        return f"<placeholder for {super().__repr__()}>"


if sys.version_info >= (3, 11):
    from typing import Self
elif TYPE_CHECKING:
    from typing_extensions import Self
else:

    class Self(metaclass=_PlaceholderMeta):
        pass


if sys.version_info >= (3, 10):
    from typing import TypeAlias, TypeGuard
elif TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeGuard
else:

    class TypeGuard(metaclass=_PlaceholderMeta):
        pass

    class TypeAlias:
        pass
