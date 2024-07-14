"""A C parser made with sly. Heavily based on pycparser."""

from .c_context import CContext, CParsingError, parse, parse_file, preprocess_file


__all__ = ("CContext", "CParsingError", "parse", "preprocess_file", "parse_file")
