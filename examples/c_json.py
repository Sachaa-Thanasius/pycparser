# ------------------------------------------------------------------------------
# pycparser: c_json.py
#
# by Michael White (@mypalmike)
#
# This example includes functions to serialize and deserialize an ast
# to and from json format. Serializing involves walking the ast and converting
# each node from a python Node object into a python dict. Deserializing
# involves the opposite conversion, walking the tree formed by the
# dict and converting each dict into the specific Node object it represents.
# The dict itself is serialized and deserialized using the python json module.
#
# The dict representation is a fairly direct transformation of the object
# attributes. Each node in the dict gets one metadata field referring to the
# specific node class name, _nodetype. Each local attribute (i.e. not linking
# to child nodes) has a string value or array of string values. Each child
# attribute is either another dict or an array of dicts, exactly as in the
# Node object representation. The "coord" attribute, representing the
# node's location within the source code, is serialized/deserialized from
# a Coord object into a string of the format "filename:line[:column]".
#
# Example TypeDecl node, with IdentifierType child node, represented as a dict:
#     "type": {
#         "_nodetype": "TypeDecl",
#         "coord": "c_files/funky.c:8",
#         "declname": "o",
#         "quals": [],
#         "type": {
#             "_nodetype": "IdentifierType",
#             "coord": "c_files/funky.c:8",
#             "names": [
#                 "char"
#             ]
#         }
#     }
# ------------------------------------------------------------------------------
import json
import re
from collections.abc import Callable
from typing import Any, Optional

from cparsing import c_ast, parse_file
from cparsing.utils import Coord


RE_CHILD_ARRAY = re.compile(r"(.*)\[(.*)\]")
RE_INTERNAL_ATTR = re.compile("__.*__")


class CJsonError(Exception):
    pass


def memodict(fn: Callable[[c_ast.AST], set[str]]) -> Callable[[c_ast.AST], set[str]]:
    """Fast memoization decorator for a function taking a single argument"""

    class memodict(dict[c_ast.AST, set[str]]):
        def __missing__(self, key: c_ast.AST) -> set[str]:
            ret = self[key] = fn(key)
            return ret

    return memodict().__getitem__


@memodict
def child_attrs_of(klass: c_ast.AST) -> set[str]:
    """Given a Node class, get a set of child attrs. Memoized to avoid highly repetitive string manipulation."""

    non_child_attrs = set(klass._fields)
    all_attrs = {i for i in klass.__slots__ if not RE_INTERNAL_ATTR.match(i)}
    return all_attrs - non_child_attrs


def to_dict(node: c_ast.AST):
    """Recursively convert an ast into dict representation."""

    klass = node.__class__

    result: dict[str, object] = {
        # Metadata
        "_nodetype": klass.__name__,
        # Local node attributes
        **{attr: getattr(node, attr) for attr in klass._fields},
        # Coord object
        "coord": str(node.coord) if node.coord else None,
    }

    # Child attributes
    for child_name, child in node.children():
        # Child strings are either simple (e.g. 'value') or arrays (e.g. 'block_items[1]')
        match = RE_CHILD_ARRAY.match(child_name)
        if match:
            array_name, array_index = match.groups()
            array_index = int(array_index)
            # arrays come in order, so we verify and append.
            result[array_name] = result.get(array_name, [])
            if array_index != len(result[array_name]):
                msg = (
                    f"Internal ast error. Array {array_name} out of order. "
                    f"Expected index {len(result[array_name])}, got {array_index}."
                )
                raise CJsonError(msg)
            result[array_name].append(to_dict(child))
        else:
            result[child_name] = to_dict(child)

    # Any child attributes that were missing need "None" values in the json.
    for child_attr in child_attrs_of(klass):
        if child_attr not in result:
            result[child_attr] = None

    return result


def to_json(node: c_ast.AST, **kwargs: object) -> str:
    """Convert ast node to json string."""

    return json.dumps(to_dict(node), **kwargs)


def file_to_dict(filename: str):
    """Load C file into dict representation of ast."""

    tree = parse_file(filename, use_cpp=True)
    return to_dict(tree)


def file_to_json(filename: str, **kwargs: object) -> str:
    """Load C file into json string representation of ast."""

    tree = parse_file(filename, use_cpp=True)
    return to_json(tree, **kwargs)


def _parse_coord(coord_str: Optional[str]) -> Optional[Coord]:
    """Parse coord string (file:line[:column]) into Coord object."""

    if coord_str is None:
        return None

    vals = coord_str.split(":")
    vals.extend([None] * (3 - len(vals)))
    filename, line, column = vals[:3]
    return Coord(int(line), int(column), filename=filename)


def _convert_to_obj(value: object):
    """Convert an object in the dict representation into an object.

    Notes
    -----
    Mutually recursive with from_dict.
    """

    if isinstance(value, dict):
        return from_dict(value)
    elif isinstance(value, list):
        return [_convert_to_obj(item) for item in value]
    else:
        # String
        return value


def from_dict(node_dict: dict[str, c_ast.AST]) -> Any:
    """Recursively build an ast from dict representation."""

    class_name = node_dict.pop("_nodetype")

    klass = getattr(c_ast, class_name)

    # Create a new dict containing the key-value pairs which we can pass
    # to node constructors.
    objs = {}
    for key, value in node_dict.items():
        if key == "coord":
            objs[key] = _parse_coord(value)
        else:
            objs[key] = _convert_to_obj(value)

    # Use keyword parameters, which works thanks to beautifully consistent
    # ast Node initializers.
    return klass(**objs)


def from_json(ast_json: str) -> c_ast.AST:
    """Build an ast from json string representation."""

    return from_dict(json.loads(ast_json))


# ------------------------------------------------------------------------------
def main() -> None:
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("filename")
    filename: str = argparser.parse_args().filename

    # Some test code...
    # Do trip from C -> ast -> dict -> ast -> json, then print.
    tree = parse_file(filename, use_cpp=True)
    ast_dict = file_to_dict(filename)
    ast = from_dict(ast_dict)
    print(to_json(ast, sort_keys=True, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
