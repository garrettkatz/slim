import numpy as np

# types
class Output: pass
class Scalar(Output): pass
class Vector(Output): pass

class Node:
    def __init__(self, out_type=Output):
        self.out_type = out_type
        self.args = ()

    def __repr__(self):
        return str(self)

    def match(self, pattern):

        # output types must be compatible for match
        if not issubclass(self.out_type, pattern.out_type): return False, {}

        # parameters with same output type are always matched
        if type(pattern) == Parameter: return True, {pattern.num: self}

        # otherwise, node types must match
        if type(self) != type(pattern): return False, {}

        # recursive case, collect matches
        matches = {}
        for arg, pat in zip(self.args, pattern.args):
            status, sub_matches = arg.match(pat)
            if status == False: return False, {}
            matches.update(sub_matches)
        return True, matches

    def substitute(self, matches):
        # for patterns, replaces parameters with arguments

        # base case: replace parameter with its match
        if type(self) == Parameter:
            return matches[self.num]

        # recursive case: substitute args and return a copy
        args = tuple(arg.substitute(matches) for arg in self.args)
        return type(self)(args, self.out_type)

    def neighbors(self):

        # recursively collect neighbors of sub-trees
        result = []
        for a, arg in enumerate(self.args):
            for sub_neighbor in arg.neighbors():
                args = self.args[:a] + (sub_neighbor,) + self.args[a+1:]
                neighbor = type(self)(args, self.out_type)
                result.append(neighbor)

        # match patterns for any additional neighbors at self
        for patterns in neighbor_patterns:

            # find matching pattern if any
            for match_idx, pattern in enumerate(patterns):
                success, matches = self.match(pattern)
                if success: break
            if not success: continue

            # if this code is reached there was a match at match_idx
            candidates = patterns[:match_idx] + patterns[match_idx+1:]
            for candidate in candidates:
                neighbor = candidate.substitute(matches)
                result.append(neighbor)

        return result

class Terminal(Node): pass

class Constant(Terminal):
    def __init__(self, value, out_type=Output):
        super().__init__(out_type)
        self.value = value
    def __call__(self, inputs):
        return self.value
    def __str__(self):
        return str(self.value)
    def substitute(self, matches):
        return Constant(self.value, self.out_type)

class Variable(Terminal):
    def __init__(self, name, out_type=Output):
        super().__init__(out_type)
        self.name = name
    def __call__(self, inputs):
        return inputs[self.name]
    def __str__(self):
        return self.name
    def substitute(self, matches):
        return Variable(self.name, self.out_type)

# base class for non-terminals in expression trees
class Operator(Node):
    def __init__(self, args=(), out_type=Output):
        super().__init__(out_type)
        self.args = args

    def random_arg_classes(self):
        raise NotImplementedError("subclasses should implement this method")

    def sprout(self, term_prob=.5, arg_classes=None):
        # creates copy of self with randomly replaced arguments
        args = ()

        # randomize arg classes if not specified
        if arg_classes is None:
            arg_classes = self.random_arg_classes()

        # assign argument classes
        for out_type in arg_classes:

            # might be a random terminal, base case
            if np.random.rand() < term_prob:
                args += (random_terminal(out_type),)

            # or another operator that sprouts, recursive case
            else:
                op = np.random.choice(operators)
                args += (op(out_type=out_type).sprout(term_prob),)

        # return a copy
        return type(self)(args, self.out_type)

# parameter placeholder in patterns
class Parameter(Node):
    def __init__(self, num, out_type=Output):
        super().__init__(out_type)
        self.num = num
    def __str__(self):
        return f"Param({self.num})"

class Binary(Operator):
    def random_arg_classes(self):
        if self.out_type is Scalar: choices = ((Scalar, Scalar),)
        else: choices = ((Scalar, Vector), (Vector, Scalar), (Vector, Vector))
        return choices[np.random.randint(len(choices))]

class Add(Binary):
    def __call__(self, inputs):
        return self.args[0](inputs) + self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} + {self.args[1]})"

class Mul(Binary):
    def __call__(self, inputs):
        return self.args[0](inputs) * self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} * {self.args[1]})"

class BMin(Binary):
    def __call__(self, inputs):
        return np.minimum(self.args[0](inputs), self.args[1](inputs))
    def __str__(self):
        return f"bmin({self.args[0]}, {self.args[1]})"

constants = tuple(Constant(v, Scalar) for v in range(-1,2))
variables = (Variable("w", Vector), Variable("x", Vector), Variable("y", Scalar), Variable("N", Scalar))
operators = (Add, Mul)

def random_terminal(out_type):
    choices = [t for t in constants + variables if issubclass(t.out_type, out_type)]
    return np.random.choice(choices)

neighbor_patterns = (
    (
        Mul(args=(Parameter(0), Parameter(1))),
        BMin(args=(
            Mul(args=(Parameter(0), Parameter(0))),
            Mul(args=(Parameter(1), Parameter(1))))),
    ), (
        Parameter(0),
        Add(args=(Parameter(0), Constant(1))),
        Add(args=(Parameter(0), Constant(-1))),
    )
)

if __name__ == "__main__":

    from geneng import load_data

    do_perceptron = True

    dataset = load_data(Ns=[3,4], perceptron=do_perceptron)
    # dataset[n]: [w_old, x, y, w_new, margins]

    dataset = [
        ({"w": dat[0], "x": dat[1], "y": dat[2], "N": dat[0].shape[1]}, dat[3])
        for dat in dataset]
    
    def fitness_function(formula):
        fitness = 0.
        for inputs, w_new in dataset:
            w_pred = formula(inputs)
            w_pred /= np.maximum(np.linalg.norm(w_pred, axis=1, keepdims=True), 10e-8)
            fitness += (w_pred * w_new).sum(axis=1).mean() # cosine similarity
        return fitness / len(dataset)


    term = random_terminal(Vector)
    print(term)
    term = random_terminal(Scalar)
    print(term)

    add = Add(args=(random_terminal(Scalar), random_terminal(Scalar)), out_type=Scalar)
    print(add)
    add2 = add.sprout(term_prob=.8)
    print('origin', add)
    print('sprout', add2)
    # for neighbor in add2.neighbors(): print(neighbor)

    mul = Mul((Variable("w", Vector), Variable("x", Vector)), Vector)
    success, match = mul.match(neighbor_patterns[0][0])
    print('success, match')
    print(success, match)

    success, match = mul.match(neighbor_patterns[1][0])
    print('success, match')
    print(success, match)

    formula = mul

    # formula = Add((
    #     Variable("w", Vector),
    #     Mul((
    #         Variable("w", Vector),
    #         Variable("x", Vector)
    #     ), Vector),
    # ), Vector)

    print(f"orig fitness({formula}) = {fitness_function(formula)}")
    print('neighbors:')
    for neighbor in formula.neighbors():
        print(neighbor)
        print(f"  fitness({neighbor}) = {fitness_function(neighbor)}")
        # for n2 in neighbor.neighbors():
        #     print(f"    fitness({n2}) = {fitness_function(n2)}")
    

    # w = Variable(Vector, "w")
    # y = Variable(Scalar, "y")
    # c = Constant(Scalar, 1)

    # print(w, y, c)
    # print(inputs["w"])
    # print(w(inputs))


