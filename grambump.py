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

    def __add__(self, other):
        other = self.promote(other)
        return Add(self, other, out_type=self.binary_out_type(other))

    def __sub__(self, other):
        other = self.promote(other)
        return Add(self, Neg(other), out_type=self.binary_out_type(other))

    def __mul__(self, other):
        other = self.promote(other)
        return Mul(self, other, out_type=self.binary_out_type(other))

class Terminal(Node):
    def __init__(self, value, out_type=Output):
        super().__init__(out_type=out_type)
        self.value = value
    def __str__(self):
        return str(self.value)

class Constant(Terminal):
    def __init__(self, value, out_type=Scalar):
        super().__init__(value, out_type=Scalar)
    def __call__(self, inputs):
        return np.array([[float(self.value)]])

class Variable(Terminal): # value treated as variable name
    def __call__(self, inputs):
        return inputs[self.value] # look up variable value by name

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

        # return a copy
        return type(self)(*args, out_type=self.out_type)

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

    def match(self, formula, pattern, matches=None):

        # initialize match dict at top-level call
        if matches is None: matches = {}

        # output types must be compatible for match
        if not issubclass(formula.out_type, pattern.out_type): return False, {}

        # unbound parameters with same output type are always matched
        if type(pattern) == Parameter:
            # check for already bound to different formula
            if pattern.value in matches and matches[pattern.value] != formula: return False, {}
            # unbound or identical
            matches[pattern.value] = formula
            return True, matches

        # otherwise, formula types must match
        if type(formula) != type(pattern): return False, {}

        # if terminal, value must also match
        if isinstance(formula, Terminal):
            return (formula.value == pattern.value), matches

        # recursive case, collect matches
        for arg, pat in zip(formula.args, pattern.args):
            status, matches = self.match(arg, pat, matches)
            if status == False: return False, {}

        return True, matches

    def substitute(self, pattern, matches):
        # matches[n] is substitution for parameter n

        # base case: replace parameter with its match
        if type(pattern) == Parameter:
            if pattern.value not in matches: print(pattern, matches)
            return matches[pattern.value]

        # base case: terminals have no args
        if isinstance(pattern, Terminal): return pattern

        # recursive case: return a copy with substituted args
        args = tuple(self.substitute(arg, matches) for arg in pattern.args)
        return type(pattern)(*args, out_type=pattern.out_type)

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
                success, matches = self.match(formula, pattern)
                if not success: continue

                # if this code is reached there was a match at match_idx
                alternates = patterns[:match_idx] + patterns[match_idx+1:]
                for alternate in alternates:
                    neighbor = self.substitute(alternate, matches)
                    # enforce out type
                    if issubclass(neighbor.out_type, formula.out_type):
                        result.append(neighbor)

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
        return np.log(np.fabs(arg), where=(arg != 0), out=np.zeros(arg.shape))
    def __str__(self):
        return f"log(|{self.args[0].strip()}|)"
    def prohibited_arg_ops():
        return (Exp,)

class Exp(ElementwiseUnary):
    def __call__(self, inputs):
        arg = self.args[0](inputs)
        return np.exp(arg, where=(arg < 3), out=np.exp(3)*np.ones(arg.shape))
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

class Mul(ElementwiseBinary):
    def __call__(self, inputs):
        return self.args[0](inputs) * self.args[1](inputs)
    def __str__(self):
        return f"({self.args[0]} * {self.args[1]})"

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

class SpanRule(Operator):
    def __init__(self, alpha, beta, out_type=Vector):
        super().__init__(alpha, beta, out_type=Vector)
    def random_arg_types(out_type):
        return (Scalar, Scalar)
    def __call__(self, inputs):
        return self.args[0](inputs) * inputs["w"] + self.args[1](inputs) * inputs["x"]
    def __str__(self):
        return f"{self.args[0]}*w + {self.args[1]}*x"

def greedy(guide, form, fit_fun, max_depth, max_itrs):
    fitness = fit_fun(form)
    explored = {}
    for itr in range(max_itrs):

        improved = False
        for neighbor in guide.neighbors(form):
            ns = str(neighbor)
            if ns in explored: continue

            nf = fit_fun(neighbor)
            explored[ns] = nf

            if neighbor.depth() > max_depth: continue

            if nf > fitness:
                form = neighbor
                fitness = nf
                improved = True

        if not improved: break

    return form, fitness, itr+1, explored

from heapq import heappop, heappush

def queued(guide, form, fit_fun, max_depth, max_itrs, max_queue, fit_target=.9999):
    max_form, max_fit = form, fit_fun(form)
    queue = [(-max_fit, 0, max_form)] # min heap
    push_order = 1 # break ties with FIFO
    explored = {}
    for itr in range(max_itrs):
        # stop if queue is exhausted
        if len(queue) == 0: break

        # pop best formula so far
        _, _, form = heappop(queue)

        # check its neighbors
        for neighbor in guide.neighbors(form):

            # skip neighbor if already explored/pushed
            ns = str(neighbor)
            if ns in explored: continue

            # not explored yet, evaluate
            nf = fit_fun(neighbor)
            explored[ns] = nf

            # track best found so far
            if nf > max_fit: max_form, max_fit = neighbor, nf

            # break if target reached
            if max_fit  > fit_target: break

            # don't push formulas over the depth limit
            if neighbor.depth() > max_depth: continue

            # don't push if queue is over the size limit
            if len(queue) >= max_queue: break

            # push new node, track order for FIFO
            heappush(queue, (-nf, push_order, neighbor)) # min heap
            push_order += 1

    return max_form, max_fit, itr+1, explored

if __name__ == "__main__":

    import pickle as pk
    from geneng import load_data

    do_perceptron = False

    do_random = False
    do_greedy = False
    do_queued = True
    do_show = False

    # max_evals = 50_000
    max_evals = 1_000_000_000

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
            Output: (Sign, Abs, Neg, Inv, Sqrt, Log, Exp, Add, Mul, Min, Max), #, Square) leave out redundancies
            Scalar: (Sum, Prod, Least, Largest) #, Mean, Dot) leave out redundancies
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
    success, match = guide.match(mul, neighborhoods[0][0])
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

    # success, match = guide.match(mul, neighborhoods[1][0])
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


    if do_random:

        print("\n********************** random sampling\n")
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

    if do_greedy:    

        print("\n********************** repeated greedy\n")
        max_fit = -1
        max_span = None
        num_evals = 0
        num_sprouts = 0
        results = []
        while num_evals < max_evals:
            num_sprouts += 1
            if do_perceptron:
                # span = SpanRule(None, None).sprout(term_prob=np.random.rand(), max_depth=np.random.randint(1,6+1))
                span = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=np.random.randint(1,6+1))
                span, fit, num_itrs, explored = greedy(guide, span, fitness_function, max_depth=6, max_itrs=100)
                # # check that final performance always better for queued, greedy is a special case
                # spanq, fitq, num_itrsq, exploredq = queued(guide, span, fitness_function, max_depth=6, max_itrs=30, max_queue=500)
                # print(f"{fit:.4f} vs {fitq:.4f}, {num_itrs} vs {num_itrsq}, {len(explored)} vs {len(exploredq)}")
            else:
                # span = SpanRule(None, None).sprout(term_prob=np.random.rand(), max_depth=np.random.randint(1,8+1))
                span = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=np.random.randint(1,10+1))
                span, fit, num_itrs, explored = greedy(guide, span, fitness_function, max_depth=10, max_itrs=500)
            num_evals += len(explored)
            if fit > max_fit:
                max_fit, max_span = fit, span
                print(f"{num_sprouts} sprouts {num_evals} evals ({num_itrs} itrs|{len(explored)} eval'd): {max_fit:.4f} <- {max_span}")
                # print(max_span.tree_str())
                results.append((num_sprouts, num_evals, max_fit, max_span))
                with open("grambump_greedy.pkl", "wb") as f: pk.dump(results, f)
                if max_fit > .99999: break

    if do_queued:
    
        print("\n********************** repeated queued\n")
        max_fit = -1
        max_span = None
        num_evals = 0
        num_sprouts = 0
        results = []
        while num_evals < max_evals:
            num_sprouts += 1
            if do_perceptron:
                # span = SpanRule(None, None).sprout(term_prob=np.random.rand(), max_depth=np.random.randint(1,6+1))
                span = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=np.random.randint(1,6+1))
                span, fit, num_itrs, explored = queued(guide, span, fitness_function, max_depth=6, max_itrs=100, max_queue=1000)
            else:
                # span = SpanRule(None, None).sprout(term_prob=np.random.rand(), max_depth=np.random.randint(1,8+1))
                span = guide.sprout(SpanRule, Vector, term_prob=np.random.rand(), max_depth=np.random.randint(1,8+1))
                span, fit, num_itrs, explored = queued(guide, span, fitness_function, max_depth=8, max_itrs=1000, max_queue=2000)
            # print(f"{num_evals} evals ({num_itrs} itrs|{len(explored)} eval'd): {max_fit:.4f} vs {fit:.4f} <- {span}")
            num_evals += len(explored)
            if fit > max_fit:
                max_fit, max_span = fit, span
                print(f"{num_sprouts} sprouts {num_evals} evals ({num_itrs} itrs|{len(explored)} eval'd): {max_fit} <- {max_span}")
                # print(max_span.tree_str())
                results.append((num_sprouts, num_evals, max_fit, max_span))
                with open("grambump_queued.pkl", "wb") as f: pk.dump(results, f)
                if max_fit > .99999: break

    if do_show:
        import matplotlib.pyplot as pt

        results = {}
        for key in ("greedy", "queued"):
            try:
                with open(f"grambump_{key}.pkl", "rb") as f: results[key] = pk.load(f)
            except:
                pass # didn't run that experiment yet

        pt.figure()
        for key in results:
            print(key)
            num_sprouts, num_evals, max_fit, max_span = zip(*results[key])
            pt.plot(num_evals, max_fit, 'o-', label=key)
            print(f"best fit {key}: {max_span}")
        pt.legend()
        pt.show()
