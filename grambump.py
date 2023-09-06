import numpy as np

# types
class Node: pass
class Scalar(Node): pass
class Vector(Node): pass

class Constant:
    def __init__(self, cls, value):
        self.cls = cls
        self.value = value
    def __call__(self, inputs):
        return self.value
    def __str__(self):
        return str(self.value)
    def neighbors(self):
        return [] #[Constant(self.cls, self.value - 1), Constant(self.cls, self.value + 1)]

class Variable:
    def __init__(self, cls, name):
        self.cls = cls
        self.name = name
    def __call__(self, inputs):
        return inputs[self.name]
    def __str__(self):
        return self.name
    def neighbors(self):
        return []

# base class for non-terminals in expression trees
class Operator:
    def __init__(self, cls, args=None):
        self.cls = cls
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
        for cls in arg_classes:

            # might be a random terminal, base case
            if np.random.rand() < term_prob:
                args += (random_terminal(cls),)

            # or another operator that sprouts, recursive case
            else:
                op = np.random.choice(operators)
                args += (op(cls).sprout(term_prob),)

        # return a copy
        return type(self)(self.cls, args)

    def match(self, pattern):
        # operators must match
        if type(self) != type(pattern): return False, {}
        # with compatible signature
        if not issubclass(self.cls, pattern.cls): return False, {}

        # base case, save matched arguments
        if type(pattern) == Parameter: return True, {pattern.num: self}

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
        return type(self)(self.cls, args)

    def neighbors(self):

        # recursively collect neighbors of sub-trees
        result = []
        for a, arg in enumerate(self.args):
            for sub_neighbor in arg.neighbors():
                args = self.args[:a] + (sub_neighbor,) + self.args[a+1:]
                neighbor = type(self)(self.cls, args)
                result.append(neighbor)

        # match patterns for any additional neighbors at self
        for patterns in neighbor_patterns:
            # check matches within current group
            checks = tuple(map(self.match, patterns))
            successes, _ = zip(*screen)
            if not any(successes): continue

            candidates = ()
            for (success, match), pattern in zip(checks, patterns):
                if not success: candidates += (pattern,)

            TODO: fix/finish this

        return result

# parameter placeholder in patterns
class Parameter(Operator):
    def __init__(self, cls, num):
        super().__init__(self, cls)
        self.num = num

class Binary(Operator):
    def random_arg_classes(self):
        if self.cls is Scalar: choices = ((Scalar, Scalar),)
        else: choices = ((Scalar, Vector), (Vector, Scalar), (Vector, Vector))
        return choices[np.random.randint(len(choices))]

class Add(Binary):
    def __call__(self, inputs):
        return self.args[0](inputs) + self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} + {self.args[1]})"
    # def neighbors(self):
    #     result = super().neighbors()
    #     result.append(Add(self.cls, (Constant(Scalar, 1), Mul(self.cls, self.args))))
    #     return result

class Mul(Binary):
    def __call__(self, inputs):
        return self.args[0](inputs) * self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} * {self.args[1]})"
    # def neighbors(self):
    #     result = super().neighbors()
    #     result.append(Add(self.cls, (Constant(Scalar, -1), Add(self.cls, self.args))))
    #     return result

class BMin(Binary):
    def __call__(self, inputs):
        return np.minimum(self.args[0](inputs), self.args[1](inputs))
    def __str__(self):
        return f"bmin({self.args[0]}, {self.args[1]})"

constants = tuple(Constant(Scalar, v) for v in range(-1,2))
variables = (Variable(Vector, "w"), Variable(Vector, "x"), Variable(Scalar, "y"), Variable(Scalar, "N"))
operators = (Add, Mul)

def random_terminal(cls):
    choices = [t for t in constants + variables if t.cls == cls]
    return np.random.choice(choices)

neighbor_patterns = (
    (
        Mul(Node, (Parameter(Node, 0), Parameter(Node, 1))),
        BMin(Node, (Mul(Node, (Parameter(Node, 0), Parameter(Node, 0))), Mul(Node, (Parameter(Node, 1), Parameter(Node, 1))))),
    ),
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

    add = Add(Scalar, (random_terminal(Scalar), random_terminal(Scalar)))
    print(add)
    add2 = add.sprout(term_prob=.8)
    print('origin', add)
    print('sprout', add2)
    for neighbor in add2.neighbors(): print(neighbor)

    formula = Add(Vector, (
        Variable(Vector, "w"),
        Mul(Vector, (
            Add(Scalar, None).sprout(term_prob=.8),
            Variable(Vector, "x")
        )),
    ))

    print(f"fitness({formula}) = {fitness_function(formula)}")
    for neighbor in formula.neighbors():
        print(f"fitness({neighbor}) = {fitness_function(neighbor)}")
    

    # w = Variable(Vector, "w")
    # y = Variable(Scalar, "y")
    # c = Constant(Scalar, 1)

    # print(w, y, c)
    # print(inputs["w"])
    # print(w(inputs))


