import itertools as it
import numpy as np

class Formula:

    """
    A formula consists of an operation applied to some sub-formulae
    operation should be a function handle
    sub_formulae[i] should be the ith input to the operation
    it can have type Formula, str, or be a constant numeric value
    if str, it should be a variable name
    """
    def __init__(self, operation, sub_formulae):
        self.op = operation
        self.subs = sub_formulae

    """
    A Formula can be called on inputs just like any function
    inputs should be a dictionary
    inputs[v] is the numeric value assigned to variable name v
    """
    def __call__(self, inputs):

        # evaluate the sub-formulae
        args = list(self.subs)
        for i, sub in enumerate(self.subs):
            # replace string variable names with their assigned value
            if type(sub) == str: args[i] = inputs[sub]
            # call sub-formulae on the inputs to get the resulting value
            if type(sub) == Formula: args[i] = sub(inputs)

        # apply this formula's operation to the values of its sub-formulae
        return self.op(*args)

    """
    Recursively generates a human-readable string of the formula
    """
    def __str__(self):
        sub_strs = map(str, self.subs)
        return f"{self.op.__name__}({', '.join(sub_strs)})"

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

    # assign the value 2 to variable x and value 3 to variable y
    inputs = {'x': 2, 'y': 3}

    # build up a simple formula that depends on variable x
    _x = 'x'
    _3x = Formula(np.multiply, ('x', 3)) # represents 3*x
    f = Formula(np.add, (_x, _3x)) # represents x + 3*x

    # confirm correct string representation and value
    print("      example:\n")
    print("inputs:", inputs)
    print("formula:", f)
    print("evaluated:", f(inputs)) # when x = 2, x + 3*x = 7

    print("\n    first 20 formulas to depth 2\n")

    OPS = [(np.negative, 1), (np.add, 2)]
    VARS = list("wxyN")
    CONSTS = [1]
    for k, f in enumerate(all_formulas(1, OPS, VARS, CONSTS)):
        print(k, f)
        if k == 20: break

