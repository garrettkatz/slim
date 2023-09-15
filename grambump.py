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

    def __eq__(self, other):
        if type(self) != type(other): return False
        if self.out_type != other.out_type: return False
        if isinstance(self, Terminal) and self.value != other.value: return False
        for a, b in zip(self.args, other.args):
            if a != b: return False
        return True

    def tree_str(self, prefix=""):
        if isinstance(self, Terminal):
            return f"{prefix}{type(self).__name__}[{self.out_type.__name__}]({self.value})"
        else:
            args = [arg.tree_str(prefix+" ") for arg in self.args]
            return f"{prefix}{type(self).__name__}[{self.out_type.__name__}]\n" + "\n".join(args)

    def strip(self):
        s = str(self)
        if s[0] == "(": s = s[1:-1]
        return s

    def depth(self):
        if len(self.args) == 0: return 0
        return 1 + max(arg.depth() for arg in self.args)

    def binary_out_type(self, other):
        if self.out_type == other.out_type == Scalar: return Scalar
        elif Vector in (self.out_type, other.out_type): return Vector
        else: return Output

    def promote(self, other):
        if type(other) in (int, float): return Constant(other)
        return other

    def match(self, pattern, matches=None):

        # initialize match dict at top-level call
        if matches is None: matches = {}

        # output types must be compatible for match
        if not issubclass(self.out_type, pattern.out_type): return False, {}

        # unbound parameters with same output type are always matched
        if type(pattern) == Parameter:
            # check for already bound to different formula
            if pattern.value in matches and matches[pattern.value] != self: return False, {}
            # unbound or identical
            matches[pattern.value] = self
            return True, matches

        # otherwise, formula types must match
        if type(self) != type(pattern): return False, {}

        # if terminal, value must also match
        if isinstance(self, Terminal):
            return (self.value == pattern.value), matches

        # recursive case, collect matches
        for arg, pat in zip(self.args, pattern.args):
            status, matches = arg.match(pat, matches)
            if status == False: return False, {}

        return True, matches

    def substitute(self, matches):
        # matches[n] is substitution for parameter n

        # base case: replace parameter with its match
        if type(self) == Parameter:
            if self.value not in matches: print(self, matches)
            return matches[self.value]

        # base case: terminals have no args
        if isinstance(self, Terminal): return self

        # recursive case: return a copy with substituted args
        args = tuple(arg.substitute(matches) for arg in self.args)
        return type(self)(*args, out_type=self.out_type)

    def __neg__(self):
        return Neg(self, out_type=self.out_type)

    def __add__(self, other):
        other = self.promote(other)
        return Add(self, other, out_type=self.binary_out_type(other))

    def __sub__(self, other):
        other = self.promote(other)
        return Add(self, Neg(other), out_type=self.binary_out_type(other))

    def __mul__(self, other):
        other = self.promote(other)
        return Mul(self, other, out_type=self.binary_out_type(other))

    def __truediv__(self, other):
        other = self.promote(other)
        return Mul(self, Inv(other), out_type=self.binary_out_type(other))

    def lump_constants(self):
        # subclasses which can have constant args but non-constant output must override this method

        if isinstance(self, Terminal): return self

        args = tuple(arg.lump_constants() for arg in self.args)
        lumped = type(self)(*args, out_type=self.out_type)

        if all(isinstance(arg, Constant) for arg in args):
            # value = lumped(None) # evaluate lumped with null input since all constant
            try: # defensive
                value = lumped(None) # evaluate lumped with null input since all constant
            except:
                raise TypeError(f"non-Variable {type(self)} accesses input, must override lump_constants")
            return Constant(value.item())
        else:
            return lumped

class Terminal(Node):
    def __init__(self, value, out_type=Output):
        super().__init__(out_type=out_type)
        self.value = value

class Constant(Terminal):
    def __init__(self, value, out_type=Scalar):
        super().__init__(value, out_type=Scalar)
    def __call__(self, inputs):
        return np.array([[float(self.value)]])
    def __str__(self):
        return f"{self.value:.3f}" if type(self.value) == float else str(self.value)

class Variable(Terminal): # value treated as variable name
    def __call__(self, inputs):
        return inputs[self.value] # look up variable value by name
    def __str__(self):
        return self.value

# base class for non-terminals in expression trees
class Operator(Node):
    def __init__(self, *args, out_type=Output):
        super().__init__(out_type)
        self.args = args

    def random_arg_types(out_type):
        raise NotImplementedError("subclasses should implement this static method")

    def prohibited_arg_ops(): return ()

    def sprout(self, term_prob=.5, max_depth=1):
        # max_depth = 1 makes immediate child args terminals
        # creates copy of self with randomly replaced arguments
        args = []

        # get arg classes
        arg_types = self.random_arg_types()
        prohibit = self.prohibited_arg_ops()

        # assign argument classes
        for out_type in arg_types:

            # might be a random terminal, base case
            if max_depth <= 1 or np.random.rand() < term_prob:
                args.append(random_terminal(out_type))

            # or another operator that sprouts, recursive case
            else:
                op = random_operator(out_type, prohibit)
                args.append(op(out_type=out_type).sprout(term_prob, max_depth-1))

        # return a copy with lumped constants
        return type(self)(*args, out_type=self.out_type).lump_constants()

# parameter placeholder in patterns
class Parameter(Terminal):
    def __str__(self):
        return f"Param({self.value})"

class Guide:
    def __init__(self, terminals, operators, neighborhoods):
        self.terminals = terminals # (terminal tuple)
        self.operators = operators # {out_type: (op tuple)}
        self.neighborhoods = neighborhoods # (... (patterns ...) )

    def random_terminal(self, out_type):
        choices = [t for t in self.terminals if issubclass(t.out_type, out_type)]
        return np.random.choice(choices)
    
    def random_operator(self, out_type, prohibit=()):
        choices = []
        for ot, ops in self.operators.items():
            if not issubclass(out_type, ot): continue
            for op in ops:
                if op not in prohibit: choices.append(op)
        return np.random.choice(choices)

    def sprout(self, op_cls, out_type, term_prob=.5, max_depth=1):
        # max_depth = 1 makes immediate child args terminals
        # creates copy of self with randomly replaced arguments
        args = []

        # get arg classes
        arg_types = op_cls.random_arg_types(out_type)
        prohibit = op_cls.prohibited_arg_ops()

        # sprout arguments
        for arg_type in arg_types:

            # might be a random terminal, base case
            if max_depth <= 1 or np.random.rand() < term_prob:
                args.append(self.random_terminal(arg_type))

            # or another operator, recursive case
            else:
                arg_cls = self.random_operator(arg_type, prohibit)
                arg = self.sprout(arg_cls, arg_type, term_prob, max_depth-1)
                args.append(arg)

        # return new object
        return op_cls(*args, out_type=out_type)

    def neighbors(self, formula):

        # recursively collect neighbors of sub-trees
        result = []
        for a, arg in enumerate(formula.args):
            for sub_neighbor in self.neighbors(arg):
                args = formula.args[:a] + (sub_neighbor,) + formula.args[a+1:]
                neighbor = type(formula)(*args, out_type=formula.out_type)
                result.append(neighbor)

        # match patterns for any additional neighbors at formula
        for patterns in self.neighborhoods:

            # find matching patterns if any
            for match_idx, pattern in enumerate(patterns):
                success, matches = formula.match(pattern)
                if not success: continue

                # if this code is reached there was a match at match_idx
                alternates = patterns[:match_idx] + patterns[match_idx+1:]
                for alternate in alternates:
                    neighbor = alternate.substitute(matches)
                    # enforce out type
                    if issubclass(neighbor.out_type, formula.out_type):
                        result.append(neighbor) # don't lump at all
                        # result.append(neighbor.lump_constants()) # only lump below a change

        # # lump all constants to reduce tree sizes? only alternatives need be lumped again
        # result = [neighbor.lump_constants() for neighbor in result]

        return result

class ElementwiseUnary(Operator):
    def __init__(self, arg0, out_type=Output):
        # coerce out_type to arg_type
        super().__init__(arg0, out_type=arg0.out_type)

    def random_arg_types(out_type):
        if out_type is Scalar: choice = (Scalar,)
        else: choice = (Vector,)
        return choice

class Sign(ElementwiseUnary):
    def __call__(self, inputs):
        return np.sign(self.args[0](inputs))
    def __str__(self):
        return f"sign({self.args[0].strip()})"
    def prohibited_arg_ops():
        return (Sign, Abs)

class Abs(ElementwiseUnary):
    def __call__(self, inputs):
        return np.fabs(self.args[0](inputs))
    def __str__(self):
        return f"|{self.args[0].strip()}|"
    def prohibited_arg_ops():
        return (Abs, Neg)

class Neg(ElementwiseUnary):
    def __call__(self, inputs):
        return -self.args[0](inputs)
    def __str__(self):
        return f"-{self.args[0]}"
    def prohibited_arg_ops():
        return (Neg,)

class Inv(ElementwiseUnary):
    def __call__(self, inputs):
        arg = self.args[0](inputs)
        return np.divide(1., arg, where=(np.fabs(arg) > .01), out=100.*np.sign(arg))
    def __str__(self):
        return f"(1/{self.args[0]})"
    def prohibited_arg_ops():
        return (Inv,)

class Square(ElementwiseUnary):
    def __call__(self, inputs):
        return self.args[0](inputs)**2
    def __str__(self):
        return f"({self.args[0]}**2)"
    def prohibited_arg_ops():
        return (Sqrt,)

class Sqrt(ElementwiseUnary):
    def __call__(self, inputs):
        arg = self.args[0](inputs)
        return np.sqrt(np.fabs(arg))
    def __str__(self):
        return f"sqrt(|{self.args[0].strip()}|)"
    def prohibited_arg_ops():
        return (Square,)

class Log(ElementwiseUnary):
    def __call__(self, inputs):
        arg = self.args[0](inputs)
        return np.log(np.fabs(arg), where=(np.fabs(arg) > np.exp(-10)), out=-10*np.ones(arg.shape))
    def __str__(self):
        return f"log(|{self.args[0].strip()}|)"
    def prohibited_arg_ops():
        return (Exp,)

class Exp(ElementwiseUnary):
    def __call__(self, inputs):
        arg = self.args[0](inputs)
        return np.exp(arg, where=(arg < 5), out=np.exp(5)*np.ones(arg.shape))
    def __str__(self):
        return f"exp({self.args[0].strip()})"
    def prohibited_arg_ops():
        return (Log, Exp)

class ReducerUnary(Operator):
    def __init__(self, arg0, out_type=Scalar):
        super().__init__(arg0, out_type=Scalar)
    def random_arg_types(out_type):
        return (Vector,)

class Sum(ReducerUnary):
    def __call__(self, inputs):
        return self.args[0](inputs).sum(axis=1, keepdims=True)
    def __str__(self):
        return f"sum({self.args[0].strip()})"

class Prod(ReducerUnary):
    def __call__(self, inputs):
        return self.args[0](inputs).prod(axis=1, keepdims=True)
    def __str__(self):
        return f"prod({self.args[0].strip()})"

class Mean(ReducerUnary):
    def __call__(self, inputs):
        return self.args[0](inputs).mean(axis=1, keepdims=True)
    def __str__(self):
        return f"mean({self.args[0].strip()})"

class Least(ReducerUnary):
    def __call__(self, inputs):
        return self.args[0](inputs).min(axis=1, keepdims=True)
    def __str__(self):
        return f"least({self.args[0].strip()})"

class Largest(ReducerUnary):
    def __call__(self, inputs):
        return self.args[0](inputs).max(axis=1, keepdims=True)
    def __str__(self):
        return f"largest({self.args[0].strip()})"

class ElementwiseBinary(Operator):
    def __init__(self, arg0, arg1, out_type=Output):
        arg0, arg1 = self.promote(arg0), self.promote(arg1) # wrap constants
        # coerce out_type based on arg_types
        if arg0.out_type == arg1.out_type == Scalar: out_type = Scalar
        elif Vector in (arg0.out_type, arg1.out_type): out_type = Vector
        super().__init__(arg0, arg1, out_type=out_type)

    def random_arg_types(out_type):
        choices = ((Scalar, Scalar), (Scalar, Vector), (Vector, Scalar), (Vector, Vector))
        if out_type is Scalar: choices = choices[:1]
        if out_type is Vector: choices = choices[1:]
        return choices[np.random.randint(len(choices))]

class Add(ElementwiseBinary):
    def __call__(self, inputs):
        return self.args[0](inputs) + self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} + {self.args[1]})"
    def lump_constants(self):
        node = super().lump_constants()
        if isinstance(node, Constant): return node

        if node.args[0] == Constant(0): return node.args[1]
        if node.args[1] == Constant(0): return node.args[0]        

        if tuple(map(type, node.args)) == (Constant, Add):
            if isinstance(node.args[1].args[0], Constant):
                return node.args[1].args[1] + (node.args[0].value + node.args[1].args[0].value)
            if isinstance(node.args[1].args[1], Constant):
                return node.args[1].args[0] + (node.args[0].value + node.args[1].args[1].value)

        if tuple(map(type, node.args)) == (Add, Constant):
            if isinstance(node.args[0].args[0], Constant):
                return node.args[0].args[1] + (node.args[1].value + node.args[0].args[0].value)
            if isinstance(node.args[0].args[1], Constant):
                return node.args[0].args[0] + (node.args[1].value + node.args[0].args[1].value)

        return node

class Mul(ElementwiseBinary):
    def __call__(self, inputs):
        return self.args[0](inputs) * self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} * {self.args[1]})"
    def lump_constants(self):
        node = super().lump_constants()
        if isinstance(node, Constant): return node

        # if Constant(0) in node.args: return Constant(0) # can collapse vectors to scalars
        # leave vectors unlumped
        if node.args[0] == Constant(0) and node.args[1].out_type is Scalar: return Constant(0)
        if node.args[1] == Constant(0) and node.args[0].out_type is Scalar: return Constant(0)

        if node.args[0] == Constant(1): return node.args[1]
        if node.args[1] == Constant(1): return node.args[0]        

        if tuple(map(type, node.args)) == (Constant, Mul):
            if isinstance(node.args[1].args[0], Constant):
                return node.args[1].args[1] * (node.args[0].value * node.args[1].args[0].value)
            if isinstance(node.args[1].args[1], Constant):
                return node.args[1].args[0] * (node.args[0].value * node.args[1].args[1].value)

        if tuple(map(type, node.args)) == (Mul, Constant):
            if isinstance(node.args[0].args[0], Constant):
                return node.args[0].args[1] * (node.args[1].value * node.args[0].args[0].value)
            if isinstance(node.args[0].args[1], Constant):
                return node.args[0].args[0] * (node.args[1].value * node.args[0].args[1].value)

        return node


class Min(ElementwiseBinary):
    def __call__(self, inputs):
        return np.minimum(self.args[0](inputs), self.args[1](inputs))
    def __str__(self):
        return f"min({self.args[0].strip()}, {self.args[1].strip()})"

class Max(ElementwiseBinary):
    def __call__(self, inputs):
        return np.maximum(self.args[0](inputs), self.args[1](inputs))
    def __str__(self):
        return f"max({self.args[0].strip()}, {self.args[1].strip()})"

class Dot(Operator):
    def __init__(self, arg0, arg1, out_type=Scalar):
        super().__init__(arg0, arg1, out_type=Scalar)
    def random_arg_types(out_type):
        return (Vector, Vector)
    def __call__(self, inputs):
        return (self.args[0](inputs) * self.args[1](inputs)).sum(axis=1, keepdims=True)
    def __str__(self):
        return f"({self.args[0]} . {self.args[1]})"

class WhereLargest(Operator):
    def __init__(self, rank, result, out_type=Scalar):
        super().__init__(rank, result, out_type=Scalar)
    def random_arg_types(out_type):
        return (Vector, Vector)
    def __call__(self, inputs):
        idx = self.args[0](inputs).argmax(axis=1, keepdims=True)
        return np.take_along_axis(self.args[1](inputs), idx, axis=1)
    def __str__(self):
        return f"({self.args[1]}[argmax {self.args[0]}])"

class WherePositive(Operator):
    def __init__(self, indicator, positive, negative, out_type=Output):
        # coerce out_type based on arg_types
        out_types = (indicator.out_type, positive.out_type, negative.out_type)
        if out_types[0] == out_types[1] == out_types[2] == Scalar: out_type = Scalar
        elif Vector in out_types: out_type = Vector
        super().__init__(indicator, positive, negative, out_type=out_type)
    def random_arg_types(out_type):
        choices = [
            (Scalar, Scalar, Scalar),
            (Scalar, Scalar, Vector),
            (Scalar, Vector, Scalar),
            (Scalar, Vector, Vector),
            (Vector, Scalar, Scalar),
            (Vector, Scalar, Vector),
            (Vector, Vector, Scalar),
            (Vector, Vector, Vector),
        ]
        if out_type is Scalar: return choices[0]
        if out_type is Vector: return choices[np.random.randint(1, 8)]
        return (Output, Output, Output)
    def __call__(self, inputs):
        return np.where(self.args[0](inputs) > 0, self.args[1](inputs), self.args[2](inputs))
    def __str__(self):
        return f"where({self.args[0]} > 0, {self.args[1].strip()}, {self.args[2].strip()})"

class SpanRule(Operator):
    def __init__(self, alpha, beta, out_type=Vector):
        super().__init__(alpha, beta, out_type=Vector)
    def random_arg_types(out_type):
        return (Scalar, Scalar)
    def __call__(self, inputs):
        return self.args[0](inputs) * inputs["w"] + self.args[1](inputs) * inputs["x"]
    def __str__(self):
        return f"{self.args[0]}*w + {self.args[1]}*x"
    def lump_constants(self):
        args = tuple(arg.lump_constants() for arg in self.args)
        return SpanRule(*args)

if __name__ == "__main__":

    import pickle as pk
    from geneng import load_data

    do_perceptron = True

    dataset = load_data(Ns=[3,4], perceptron=do_perceptron)
    # dataset[n]: [w_old, x, y, w_new, margins]

    dataset = [
        ({"w": dat[0], "x": dat[1], "y": dat[2], "N": np.array([[float(dat[0].shape[1])]])}, dat[3])
        for dat in dataset]

    print("num examples:")
    print(dataset[0][0]["y"].shape, dataset[1][0]["y"].shape)

    def fitness_function(formula):
        fitness = 0.
        for inputs, w_new in dataset:
            # invoke formula
            w_pred = formula(inputs)

            # broadcast scalar outputs
            if w_pred.shape[1] == 1: w_pred = w_pred * np.ones(w_new.shape[1])

            # mean cosine similarity
            w_pred = w_pred / np.maximum(np.linalg.norm(w_pred, axis=1, keepdims=True), 10e-8)
            fitness += (w_pred * w_new).sum(axis=1).mean() # cosine similarity

        return fitness / len(dataset)

    variables = (Variable("w", Vector), Variable("x", Vector), Variable("y", Scalar), Variable("N", Scalar))
    w, x, y, N = variables

    if do_perceptron:
        constants = tuple(Constant(v, Scalar) for v in range(1,2))
        operators = {
            Output: (Sign, Neg, Add, Mul),
            Scalar: (Sum,),
        }
    else:
        constants = tuple(Constant(v, Scalar) for v in range(-1,2))
        operators = { # out_type: classes
            Output: (Sign, Abs, Neg, Inv, Sqrt, Square, Log, Exp, Add, Mul, Min, Max, WherePositive),
            Scalar: (Sum, Prod, Least, Largest, Mean, Dot, WhereLargest),
        }
    
    # all patterns in a group must contain the same parameters
    P = tuple(Parameter(n) for n in range(3))
    neighborhoods = (
        # terminals
        (w, Sign(w), ),
        # (w, x, y * w, y * x, w * x, Sign(w), ),
        (y, Prod(x), Sign(Sum(x)), Least(x), Largest(x), Sign(Sum(w * x)), Constant(1), Constant(-1)),
        (N, Abs(y), Constant(3), Constant(4), Sum(x), Largest(w), ),
    
        # unaries
        (Sign(P[0]), Sqrt(P[0]) * Sign(P[0]), Log(Abs(P[0]) + 1) * Sign(P[0]), ),
        (Neg(P[0]), Exp(Neg(P[0])) - 1, ),
        (Inv(P[0]), Sign(P[0]) * Exp(Abs(Neg(P[0]) - 1)), Sign(P[0]) * Max(Neg(Abs(P[0])) + 2, Constant(0)), ),
        # (Square(P[0]), Abs(P[0]), ), # omit redundancies
        (Abs(P[0]), P[0] * P[0], ),
        (Sqrt(P[0]), Abs(P[0]), ),
        (Log(P[0]), Sqrt(P[0]) - 1, ),
        (Exp(P[0]), Max(P[0] + 1, Constant(0)), Max(Sign(P[0] + 1) * (P[0] + 1) * (P[0] + 1), Constant(0)), ),
        (Sum(P[0]), Sum(P[0] * y), Sum(P[0] * x), (Least(P[0]) + Largest(P[0])) * Inv(Constant(2)) * N, ),
        (Prod(P[0]), Prod(Sign(P[0])), ),
        # (Mean(P[0]), (Least(P[0]) + Largest(P[0])) * Inv(Constant(2)), ), # omit redundancies
        (Least(P[0]), Least(Sign(P[0])), ),
        (Largest(P[0]), Largest(Sign(P[0])), ),
    
        # binaries
        (P[0] + P[1], Max(P[0], P[1]) * 2, Min(P[0], P[1]) * 2, ),
        (P[0] * P[1], Min(P[0] * P[0], P[1] * P[1]) * Sign(P[0] * P[1]), ),
        (P[0] * P[0], Exp(Abs(P[0])) - 1, ),
        (Min(P[0], P[1]), (P[0] + P[1]) * Inv(Constant(2)), ),
        (Max(P[0], P[1]), (P[0] + P[1]) * Inv(Constant(2)), ),
        # (Dot(P[0], P[1]), Least(P[0] * P[1]) + Largest(P[0] * P[1]), ), # leave out until simplifications
    
        # # open-ended
        # (P[0], P[0] + y, P[0] + x, P[0] + 1, P[0] - 1, ),
    
    )

    guide = Guide(constants + variables, operators, neighborhoods)

    # check __eq__
    assert w == w
    assert w != x
    assert Constant(1) == Constant(1)
    assert (w + x) != (w + y)

    term = guide.random_terminal(Vector)
    print('vec term', term)
    term = guide.random_terminal(Scalar)
    print('sca term', term)

    op = guide.random_operator(Vector)
    print('vec op', op)
    op = guide.random_operator(Scalar)
    print('sca op', op)

    add = guide.random_terminal(Scalar) + guide.random_terminal(Scalar)
    print(add)
    # add2 = add.sprout(term_prob=0., max_depth=3)
    add2 = guide.sprout(Add, Scalar, term_prob=0., max_depth=3)
    print('origin', add)
    print('sprout', add2)
    # for neighbor in add2.neighbors(): print(neighbor)

    # span = SpanRule(None, None).sprout(term_prob=.5, max_depth=3)
    span = guide.sprout(SpanRule, Vector, term_prob=.5, max_depth=3)
    print('rand span', span)

    mul = w * x
    success, match = mul.match(neighborhoods[0][0])
    print('success, match')
    print(success, match)

    # perceptron = SpanRule(Constant(1), y - Sign(Sum(w * x)))
    perceptron = SpanRule(Constant(1), Add(Neg(Sign(Sum(Mul(w, x, out_type=Vector), out_type=Scalar), out_type=Output), out_type=Scalar), y, out_type=Output))
    print(f"perceptron {perceptron} fitness = {fitness_function(perceptron)}")
    print(perceptron.tree_str())

    p2 = SpanRule(Neg(Sign(Constant(1) + Neg(N * N))), Sign(Neg(Sum(x * w))) + y)
    print(f"p2 {p2} fitness = {fitness_function(p2)}")

    abs2 = Abs(Constant(2))
    print(f"abs2 {abs2} fitness = {fitness_function(abs2)}")

    # success, match = mul.match(neighborhoods[1][0])
    # print('success, match')
    # print(success, match)

    formula = mul
    # formula = w + (w * x)

    print(f"{formula}   -->   {fitness_function(formula)}")
    print('neighbors:')
    for neighbor in guide.neighbors(formula):
        print(f"  {neighbor}  -->  {fitness_function(neighbor)}")
        # print(neighbor.tree_str("  "))
        # for n2 in neighbor.neighbors():
        #     print(f"    fitness({n2}) = {fitness_function(n2)}")


    print("\n********************** random sampling\n")
    max_evals = 10000
    max_fit = -1
    max_span = None
    for rep in range(max_evals):
        # span = SpanRule(None, None).sprout(term_prob=np.random.rand(), max_depth=6)
        span = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=6)
        fit = fitness_function(span)
        if fit > max_fit:
            max_fit, max_span = fit, span
            print(f"{rep}: {fit} vs {max_fit} <- {max_span}")
            # print(max_span.tree_str())
            if max_fit > .99999: break
        # print(f"{rep}: {fit} vs {max_fit} <- {span}, {max_span}")

