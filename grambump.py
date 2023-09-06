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
        return [Constant(self.cls, self.value - 1), Constant(self.cls, self.value + 1)]

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

class Binary:
    def __init__(self, cls, args=None):
        self.cls = cls
        self.args = args
    def random_arg_classes(self):
        if self.cls is Scalar: choices = ((Scalar, Scalar),)
        else: choices = ((Scalar, Vector), (Vector, Scalar), (Vector, Vector))
        return choices[np.random.randint(len(choices))]
    def sprout(self, term_prob=.5, arg_classes=None):
        args = ()

        if arg_classes is None:
            arg_classes = self.random_arg_classes()

        for cls in arg_classes:
            if np.random.rand() < term_prob:
                args += (random_terminal(cls),)
            else:
                op = np.random.choice(operators)
                args += (op(cls).sprout(term_prob),)

        return type(self)(self.cls, args)

    def neighbors(self):
        result = []
        for neighbor in self.args[0].neighbors():
            result.append(type(self)(self.cls, (neighbor, self.args[1])))
        for neighbor in self.args[1].neighbors():
            result.append(type(self)(self.cls, (self.args[0], neighbor)))
        return result


class Add(Binary):
    def __call__(self, inputs):
        return self.args[0](inputs) + self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} + {self.args[1]})"
    def neighbors(self):
        result = super().neighbors()
        result.append(Add(self.cls, (Constant(Scalar, 1), Mul(self.cls, self.args))))
        return result

class Mul(Binary):
    def __call__(self, inputs):
        return self.args[0](inputs) * self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} * {self.args[1]})"
    def neighbors(self):
        result = super().neighbors()
        result.append(Add(self.cls, (Constant(Scalar, -1), Add(self.cls, self.args))))
        return result

constants = tuple(Constant(Scalar, v) for v in range(-1,2))
variables = (Variable(Vector, "w"), Variable(Vector, "x"), Variable(Scalar, "y"), Variable(Scalar, "N"))
operators = (Add, Mul)

def random_terminal(cls):
    choices = [t for t in constants + variables if t.cls == cls]
    return np.random.choice(choices)

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


