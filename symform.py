import sympy as sm
import numpy as np
import itertools as it

"""
leaves[i] = (type, symbol): type in "VS" for vector or scalar symbol
operators[i] = (signature, func): signature = (out_type, in_types), also strings
depth: max depth of expressions
see main block below for example
"""
def all_formulas(leaves, operators, depth):

    # initialize tree dictionary, organized by VS type
    # trees[type] = {expr: node, ...}
    # expr is a sympy expression, node is (func, child nodes) or a leaf symbol
    trees = {t: {} for t in "VS"}
    for (t, s) in leaves: trees[t][s] = s # leaves are the symbols themselves

    for D in range(depth):

        # copy trees so far for next depth
        next_trees = {t: dict(trees[t]) for t in trees}

        # try applying each operator to all trees so far
        for (out_type, in_types), func in operators:
            # get tree dictionaries for input/output types
            out_trees = next_trees[out_type]
            in_trees = tuple(trees[t].items() for t in in_types)

            # take every combination of valid inputs from existing trees
            for args in it.product(*in_trees):
                exprs, nodes = zip(*args)
                # get new expression by applying operator 
                expr = func(*exprs)
                # skip if expression simplifies to existing tree
                if expr in out_trees: continue
                # otherwise save the new tree
                out_trees[expr] = (func, nodes)
                # and yield it
                yield out_type, expr, (func, nodes)

        # overwrite trees for next iteration
        trees = next_trees

"""
tree: (func, child nodes) or symbol
inputs[symbol]: value
"""
def evaluate(tree, inputs):
    if type(tree) == tuple:
        func, nodes = tree
        return func(*[evaluate(n, inputs) for n in nodes])
    else:
        return inputs[tree]


if __name__ == "__main__":

    w = sm.MatrixSymbol('w', 4, 1)
    x = sm.MatrixSymbol('x', 4, 1)
    y = sm.Symbol('y')
    N = sm.Symbol('N')
    _1 = sm.Integer(1)

    inputs = {
        w: np.ones((4,1)),
        x: np.arange(4).reshape(4,1),
        y: -1,
        N: 4,
        _1: 1,
    }

    leaves = [
        ("V", w),
        ("V", x),
        ("S", y),
        ("S", N),
        ("S", _1),
    ]

    operators = [
        # scalar arithmetic
        (("S", "S"), lambda a: -a),
        (("S", "S"), lambda a: sm.zoo if a == 0 else 1/a),
        (("S", "S"), lambda a: 0 if a == 0 else a/abs(a)), # sign
        (("S", "SS"), lambda a, b: a + b),
        (("S", "SS"), lambda a, b: a * b),
        (("S", "SS"), lambda a, b: a ** b),
        # vector arithmetic
        (("V", "SV"), lambda a, b: a * b),
        (("V", "VV"), lambda a, b: a + b),
        # inner product
        (("S", "VV"), lambda a, b: (a.T @ b)[0]),
    ]

    max_print = 100
    print("Inputs:", inputs)
    print(f"First {max_print} scalar expressions evaluated on inputs:")
    type_counts = {"V": 0, "S": 0}

    for (out_type, expr, tree) in all_formulas(leaves, operators, depth=1):
    
        type_counts[out_type] += 1

        if out_type == "S" and type_counts[out_type] <= max_print:
            chars = len(str(expr))
            print(type_counts[out_type], expr, " "*(80 - chars), " --> ", evaluate(tree, inputs))
            if type_counts[out_type] == max_print: print("Doing the rest of them...")

    print(f"{sum(type_counts.values())} formulas in total")

