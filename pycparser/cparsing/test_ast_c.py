# ruff: noqa: ANN201
import weakref

from pycparser.cparsing import ast_c, cparser


def test_BinaryOp():
    b1 = ast_c.BinaryOp(op='+', left=ast_c.Constant(type='int', value='6'), right=ast_c.Identifier(name='joe'))

    assert isinstance(b1.left, ast_c.Constant)

    assert b1.left.type == 'int'
    assert b1.left.value == '6'

    assert isinstance(b1.right, ast_c.Identifier)
    assert b1.right.name == 'joe'


def test_weakref_works_on_nodes():
    c1 = ast_c.Constant(type='float', value='3.14')
    wr = weakref.ref(c1)
    cref = wr()

    assert cref
    assert cref.type == 'float'
    assert weakref.getweakrefcount(c1) == 1


def test_weakref_works_on_coord():
    coord = cparser.Coord(filename='a', lineno=2, col_span=(0, 0))
    wr = weakref.ref(coord)
    cref = wr()

    assert cref
    assert cref.lineno == 2
    assert weakref.getweakrefcount(coord) == 1


class ConstantVisitor(ast_c.NodeVisitor):
    def __init__(self):
        self.values: list[str] = []

    def visit_Constant(self, node: ast_c.Constant) -> None:
        self.values.append(node.value)


def test_scalar_children():
    b1 = ast_c.BinaryOp(op='+', left=ast_c.Constant(type='int', value='6'), right=ast_c.Identifier(name='joe'))

    cv = ConstantVisitor()
    cv.visit(b1)

    assert cv.values == ['6']

    b2 = ast_c.BinaryOp(op='*', left=ast_c.Constant(type='int', value='111'), right=b1)
    b3 = ast_c.BinaryOp(op='^', left=b2, right=b1)

    cv = ConstantVisitor()
    cv.visit(b3)

    assert cv.values == ['111', '6', '6']


def tests_list_children():
    c1 = ast_c.Constant(type='float', value='5.6')
    c2 = ast_c.Constant(type='char', value='t')

    b1 = ast_c.BinaryOp(op='+', left=c1, right=c2)
    b2 = ast_c.BinaryOp(op='-', left=b1, right=c2)

    comp = ast_c.Compound(block_items=[b1, b2, c1, c2])

    cv = ConstantVisitor()
    cv.visit(comp)

    assert cv.values == ['5.6', 't', '5.6', 't', 't', '5.6', 't']


def test_dump():
    c1 = ast_c.Constant(type='float', value='5.6')
    c2 = ast_c.Constant(type='char', value='t')

    b1 = ast_c.BinaryOp(op='+', left=c1, right=c2)
    b2 = ast_c.BinaryOp(op='-', left=b1, right=c2)

    comp = ast_c.Compound(block_items=[b1, b2, c1, c2])

    expected = """\
Compound(
    block_items=[
        BinaryOp(
            op='+',
            left=Constant(
                type='float',
                value='5.6'),
            right=Constant(
                type='char',
                value='t')),
        BinaryOp(
            op='-',
            left=BinaryOp(
                op='+',
                left=Constant(
                    type='float',
                    value='5.6'),
                right=Constant(
                    type='char',
                    value='t')),
            right=Constant(
                type='char',
                value='t')),
        Constant(
            type='float',
            value='5.6'),
        Constant(
            type='char',
            value='t')])\
"""

    assert ast_c.dump(comp, indent=" " * 4) == expected
