# -----------------------------------------------------------------
# pycparser: explore_ast.py
#
# This example demonstrates how to "explore" the AST created by
# pycparser to understand its structure. The AST is a n-nary tree
# of nodes, each node having several children, each with a name.
# Just read the code, and let the comments guide you. The lines
# beginning with #~ can be uncommented to print out useful
# information from the AST.
# It helps to have the pycparser/_c_ast.cfg file in front of you.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
# -----------------------------------------------------------------
from cparsing import c_ast, parse


# This is some C source to parse. Note that pycparser must begin
# at the top level of the C file, i.e. with either declarations
# or function definitions (this is called "external declarations"
# in C grammar lingo)
#
# Also, a C parser must have all the types declared in order to
# build the correct AST. It doesn't matter what they're declared
# to, so I've inserted the dummy typedef in the code to let the
# parser know Hash and Node are types. You don't need to do it
# when parsing real, correct C code.

text = r"""
    typedef int Node, Hash;

    void HashPrint(Hash* hash, void (*PrintFunc)(char*, char*))
    {
        unsigned int i;

        if (hash == NULL || hash->heads == NULL)
            return;

        for (i = 0; i < hash->table_size; ++i)
        {
            Node* temp = hash->heads[i];

            while (temp != NULL)
            {
                PrintFunc(temp->entry->key, temp->entry->value);
                temp = temp->next;
            }
        }
    }
"""

# Create the parser and ask to parse the text. parse() will throw
# a ParseError if there's an error in the code
#
tree = parse(text, filename="<none>")
assert tree

# Uncomment the following line to see the AST in a nice, human
# readable way. show() is the most useful tool in exploring ASTs
# created by pycparser. See the c_ast.py file for the options you
# can pass it.

print(c_ast.dump(tree, indent=4, include_coords=True))

# OK, we've seen that the top node is FileAST. This is always the
# top node of the AST. Its children are "external declarations",
# and are stored in a list called ext[] (see _c_ast.cfg for the
# names and types of Nodes and their children).
# As you see from the printout, our AST has two Typedef children
# and one FuncDef child.
# Let's explore FuncDef more closely. As I've mentioned, the list
# ext[] holds the children of FileAST. Since the function
# definition is the third child, it's ext[2]. Uncomment the
# following line to show it:

print(c_ast.dump(tree.ext[2], indent=4))
assert isinstance(tree.ext[2], c_ast.FuncDef)

# A FuncDef consists of a declaration, a list of parameter
# declarations (for K&R style function definitions), and a body.
# First, let's examine the declaration.

function_decl = tree.ext[2].decl
assert isinstance(function_decl, c_ast.Decl)

# function_decl, like any other declaration, is a Decl. Its type child
# is a FuncDecl, which has a return type and arguments stored in a
# ParamList node

print(c_ast.dump(function_decl.type, indent=4))
assert isinstance(function_decl.type, c_ast.FuncDecl)
assert function_decl.type.args is not None

print(c_ast.dump(function_decl.type.args, indent=4))

# The following displays the name and type of each argument:

for param_decl in function_decl.type.args.params:
    print(f"Arg name: {param_decl.name}")
    print("Type:")
    print(c_ast.dump(param_decl.type))

# The body is of FuncDef is a Compound, which is a placeholder for a block
# surrounded by {} (You should be reading _c_ast.cfg parallel to this
# explanation and seeing these things with your own eyes).
# Let's see the block's declarations:

function_body = tree.ext[2].body

# The following displays the declarations and statements in the function
# body

# for decl in function_body.block_items:
#     decl.show()

# We can see a single variable declaration, i, declared to be a simple type
# declaration of type 'unsigned int', followed by statements.

# block_items is a list, so the third element is the For statement:

assert function_body.block_items is not None
for_stmt = function_body.block_items[2]
assert isinstance(for_stmt, c_ast.For)
print(c_ast.dump(for_stmt))

# As you can see in _c_ast.cfg, For's children are 'init, cond,
# next' for the respective parts of the 'for' loop specifier,
# and stmt, which is either a single stmt or a Compound if there's
# a block.
#
# Let's dig deeper, to the while statement inside the for loop:

assert isinstance(for_stmt.stmt, c_ast.Compound)
assert for_stmt.stmt.block_items is not None
while_stmt = for_stmt.stmt.block_items[1]
assert isinstance(while_stmt, c_ast.While)
print(c_ast.dump(while_stmt))

# While is simpler, it only has a condition node and a stmt node.
# The condition:

while_cond = while_stmt.cond
print(c_ast.dump(while_cond))

# Note that it's a BinaryOp node - the basic constituent of
# expressions in our AST. BinaryOp is the expression tree, with
# left and right nodes as children. It also has the op attribute,
# which is just the string representation of the operator.

assert isinstance(while_cond, c_ast.BinaryOp)
print(while_cond.op)
print(c_ast.dump(while_cond.left))
print(c_ast.dump(while_cond.right))


# That's it for the example. I hope you now see how easy it is to explore the
# AST created by pycparser. Although on the surface it is quite complex and has
# a lot of node types, this is the inherent complexity of the C language every
# parser/compiler designer has to cope with.
# Using the tools provided by the c_ast package it's easy to explore the
# structure of AST nodes and write code that processes them.
# Specifically, see the cdecl.py example for a non-trivial demonstration of what
# you can do by recursively going through the AST.
