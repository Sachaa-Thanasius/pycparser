"""A C parser made with sly. Heavily based on pycparser."""

from .c_context import CContext, parse, parse_file, preprocess_file

__all__ = ("CContext", "parse", "preprocess_file", "parse_file")
