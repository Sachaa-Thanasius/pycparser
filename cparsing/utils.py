"""Some utilities for interal use.

TODO: Add tests for dataclass functionality. Look at attrs and dataclass for inspiration.
"""

import itertools
import sys
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Generator, Optional, Tuple

from cparsing.sly import Parser

if sys.version_info >= (3, 11):
    from typing import Self, dataclass_transform
elif TYPE_CHECKING:
    from typing_extensions import Self, dataclass_transform
else:
    from typing import Callable, TypeVar
    from typing import Union as TUnion

    _T = TypeVar("_T")

    def dataclass_transform(
        *,
        eq_default: bool = True,
        order_default: bool = False,
        kw_only_default: bool = False,
        frozen_default: bool = False,
        field_specifiers: Tuple[TUnion[type, Callable[..., Any], Any]] = (),
        **kwargs: Any,
    ) -> Callable[[_T], _T]:
        def inner(cls_or_func: _T) -> _T:
            try:
                cls_or_func.__dataclass_transform__ = {
                    "eq_default": eq_default,
                    "order_default": order_default,
                    "kw_only_default": kw_only_default,
                    "frozen_default": frozen_default,
                    "field_specifiers": field_specifiers,
                    "kwargs": kwargs,
                }
            except AttributeError:
                pass
            return cls_or_func

    class Self:
        def __repr__(self):
            return f"<placeholder for {super().__repr__()}>"


__all__ = ("Dataclass", "Coord")


_MISSING = object()


class _FieldInfo:
    """Container for metadata about a dataclass field."""

    __slots__ = ("name", "type", "default", "kw_only")

    def __init__(self, name: str, type: Any, default: Any, kw_only: bool):  # noqa: A002
        self.name = name
        self.type = type
        self.default = default
        self.kw_only = kw_only


def _create_init(config: Dict[str, _FieldInfo]) -> str:
    """Generate the source for an __init__ function based on a given configuration."""

    def create_partial_sig(partial_config: Dict[str, _FieldInfo]) -> str:
        partial_signature_parts: list[str] = []

        for name, field in partial_config.items():
            field_type = field.type

            # Account for string annotations.
            if isinstance(field_type, str):
                actual_ann = repr(field_type)
            else:
                # FIXME: repr is better than __qualname__ for names from typing. Check how attrs and dataclass do it.
                actual_ann = getattr(field_type, "__qualname__", field_type)

            if field.default is _MISSING:
                partial_signature_parts.append(f", {name}: {actual_ann}")
            else:
                partial_signature_parts.append(f", {name}: {actual_ann} = {field.default}")

        return "".join(partial_signature_parts)

    regular_sig_part = create_partial_sig({name: field for name, field in config.items() if not field.kw_only})
    kw_only_sig_part = create_partial_sig({name: field for name, field in config.items() if field.kw_only})

    maybe_star = ", *" if kw_only_sig_part else ""

    init_lines = [f"def __init__(self{regular_sig_part}{maybe_star}{kw_only_sig_part}) -> None:"]
    if not (kw_only_sig_part or regular_sig_part):
        init_lines.append("    pass")
    else:
        init_lines.extend(f"    self.{name} = {name}" for name in config)

    return "\n".join(init_lines)


def _get_slots(cls: type) -> Generator[str, Any, None]:
    """Get the slots from a class.

    Based on code from Lib/dataclasses.py in CPython 3.12.
    """

    slots = cls.__dict__.get("__slots__")

    if slots is None:
        # Supposedly works for pure Python classes and C extension classes in CPython.
        # TODO: Test with PyPy as well.
        if getattr(cls, "__weakrefoffset__", -1) != 0:
            yield "__weakref__"
        if getattr(cls, "__dictrefoffset__", -1) != 0:
            yield "__dict__"
    elif isinstance(slots, str):
        yield slots
    elif not hasattr(slots, "__next__"):
        # Has to be iterable but not an iterator.
        yield from slots
    else:
        msg = f"Slots of {cls.__name__!r} cannot be determined"
        raise TypeError(msg)


@dataclass_transform(eq_default=False)
class _DataclassMeta(type):
    _fields: Tuple[str, ...]
    __pycparser_dataclass_fields__: Dict[str, _FieldInfo]

    def __new__(
        mcls,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        *,
        init: bool = True,
        kw_only: bool = False,
        match_args: bool = True,
        slots: bool = True,
        weakref_slot: bool = False,
    ):
        cls_annotations: dict[str, Any] = namespace.get("__annotations__", {})

        try:
            global_ns = sys.modules[namespace["__module__"]].__dict__
        except (KeyError, AttributeError):
            global_ns = {}

        if slots:
            cls_field_names = tuple(name for name, ann in cls_annotations.items() if ann is not ClassVar)
            unordered_bases = {mro_cls for base in bases for mro_cls in base.mro()}
            inherited_slots = set(itertools.chain.from_iterable(_get_slots(base) for base in unordered_bases))

            namespace["__slots__"] = tuple(
                itertools.filterfalse(
                    inherited_slots.__contains__,
                    itertools.chain(cls_field_names, (("__weakref__",) if weakref_slot else ())),
                )
            )

        field_defaults: dict[str, Any] = {}
        for ann_name in cls_annotations:
            try:
                field_defaults[ann_name] = namespace[ann_name]
            except KeyError:
                field_defaults[ann_name] = _MISSING
            else:
                namespace.pop(ann_name)

        new_cls = super().__new__(mcls, name, bases, namespace)

        dataclass_config: dict[str, _FieldInfo] = {}

        for base in reversed(new_cls.mro()[:-1]):
            base_config = getattr(base, "__pycparser_dataclass_fields__", None)
            if base_config:
                dataclass_config.update(base_config)

        for ann_name, ann in cls_annotations.items():
            try:
                current_field_info = dataclass_config[ann_name]
            except KeyError:
                dataclass_config[ann_name] = _FieldInfo(ann_name, ann, field_defaults[ann_name], kw_only)
            else:
                current_field_info.default = field_defaults[ann_name]
                current_field_info.kw_only = kw_only

        new_cls._fields = tuple(
            field_name
            for field_name, field in dataclass_config.items()
            if (field.type is not ClassVar) and (not field.kw_only)
        )

        if match_args:
            new_cls.__match_args__ = new_cls._fields  # pyright: ignore [reportAttributeAccessIssue] # Runtime attribute assignment.

        new_cls.__pycparser_dataclass_fields__ = dataclass_config

        # TODO: Research if using the old namespace *after* class creation is a problem.
        # If it is, just use an empty local dict.
        if init:
            init_code = _create_init(dataclass_config)
            exec(init_code, global_ns, namespace)  # noqa: S102
            new_cls.__init__ = namespace["__init__"]

        return new_cls


class Dataclass(metaclass=_DataclassMeta):
    """Custom metaclass-based dataclass implementation with fewer features.

    It currently handles kw_only, match_args, and weakref_slot in a similar way to standard dataclasses. It also
    generates a _fields class attribute.

    Notes
    -----
    This is not as featureful as a standard-library dataclass:
        - It can't generate as much.
        - It doesn't provide as many options to change what it can generate.
        - It doesn't handle many edge cases.

    For these reasons and the fact that it's an implementation detail for the AST classes, it isn't part of the public
    API. Still, it was fun to make and helps avoid boilerplate for a simple use case.

    Some current pecularities:
        - __slots__ is always generated, and if slots are already defined, it overrides them without raising an exception.
        - Custom descriptors aren't handled.
        - Interacting with default values sometimes has non-intuitive behavior.
        - Types from typing_extensions aren't accounted for, e.g. ClassVar.
        - __init__ parameter annotations will be missing nested annotation elements, i.e. `val: List[AST]` becomes `val: List`.
    """


class Coord(Dataclass, weakref_slot=True):
    filename: str
    lineno: int
    col_start: int
    col_end: Optional[int] = None

    @classmethod
    def from_literal(cls, p: Any, literal: str, filename: str = "") -> Self:
        return cls(filename, p.lineno, p.index, p.index + len(literal))

    @classmethod
    def from_prod(cls, parser: Parser, p: Any, tokenpos: Optional[int] = None, filename: str = "") -> Self:
        lineno = parser.line_position(p)
        col_start, col_end = parser.index_position(p)
        return cls(filename, lineno, col_start, col_end)

    def __str__(self):
        return f"{self.filename}:{self.lineno}:{(self.col_start, self.col_end)}"

    def __eq__(self, other: object, /):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.filename == other.filename
            and self.lineno == other.lineno
            and self.col_start == other.col_start
            and self.col_end == other.col_end
        )
