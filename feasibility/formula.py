import itertools as it
import numpy as np
import sympy as sm

class Formula:

    """
    A formula consists of an operation applied to some sub-formulae
    operation should be a function handle that works on both numpy and sympy types
    sub_formulae[i] should be the ith input to the operation
    it can have type Formula, sympy.Symbol, numpy.array, float or int.
    if sympy.Symbol, it represents an input to the formula
    """
    def __init__(self, operation, sub_formulae):
        self.op = operation
        self.subs = sub_formulae
        self.expr = operation(*tuple(
            sub.expr if type(sub) == Formula else sub # symbols do not have expr attribute
            for sub in sub_formulae))

    """
    A Formula can be called on inputs just like any function
    inputs should be a dictionary mapping sympy.Symbols to values
    inputs[v] is the numeric value assigned to formula input v
    """
    def __call__(self, inputs):

        # evaluate the sub-formulae
        args = list(self.subs)
        for i, sub in enumerate(self.subs):
            # replace variables with their assigned value
            if type(sub) == sm.Symbol: args[i] = inputs[sub]
            # call sub-formulae on the inputs to get the resulting value
            if type(sub) == Formula: args[i] = sub(inputs)

        # apply this formula's operation to the values of its sub-formulae
        return self.op(*args)

    """
    Uses sympy expression string representation
    """
    def __str__(self):
        return str(self.expr)

    """
    Checks whether self and other represent mathematically equivalent formulae
    Compares their simplified sympy expressions, so may return False negatives    
    """
    def __eq__(self, other):
        if type(other) != Formula: return False
        return sm.simplify(self.expr) == sm.simplify(other.expr)

    # Hash formulas by their sympy expression
    def __hash__(self):
        return self.expr.__hash__()

"""
Recursively generate all formulas up to a given depth
operations[n] = (function handle, arity)
"""
def all_formulas(depth, operations, variables, constants):
    # store all formulas in this list
    result = []

    # root could be any operation
    for op, arity in operations:

        # sub-formulae could be variables or constants
        all_subs = variables + constants

        # for deeper trees, sub-formulae could also be less deep formulae
        if depth > 0: all_subs += all_formulas(depth - 1, operations, variables, constants)

        # all combinations of sub-formulae, assuming binary ops are commutative
        for subs in it.combinations_with_replacement(all_subs, arity):
            result.append(Formula(op, subs))

    return result

if __name__ == "__main__":

    # initialize formula inputs x and y as sympy.Symbols
    x, y = sm.symbols('x y')

    # assign the value 2 to variable x and value 3 to variable y
    inputs = {x: 2, y: 3}

    # define operation function handles that work on both numpy and sympy types
    add = lambda a, b: a + b
    mul = lambda a, b: a * b

    # build up a simple formula that depends on variable x
    _3x = Formula(mul, (x, 3)) # represents 3*x
    f = Formula(add, (x, _3x)) # represents x + 3*x

    # confirm correct string representation and value
    print("      example:\n")
    print("inputs:", inputs)
    print("formula:", f)
    print("evaluated:", f(inputs)) # when x = 2, x + 3*x = 7

    print("\n    first 50 formulas to depth 2\n")

    # define operators with their arities
    OPS = [
        # inverses for subtraction and division
        (lambda a: -a, 1),
        (lambda a: 1/a, 1),
        # binary operators defined above, should be commutative
        (add, 2),
        (mul, 2),
        # # dot product for vectors
        # (lambda a, b: sum([a[i]*b[i] for i in range(len(a))]),
    ]

    # variables and constants
    VARS = list(sm.symbols('w x y N'))
    CONSTS = [1, 2]

    # enumeration (without duplicates)
    uniques = {} # will map sympy expressions to formulas
    
    for k, f in enumerate(all_formulas(1, OPS, VARS, CONSTS)):
        if f not in uniques:
            uniques[f.expr] = f
            print(k, f)
        if len(uniques) == 50: break

