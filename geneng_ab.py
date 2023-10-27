from abc import ABC
from dataclasses import dataclass
from typing import Annotated

import pickle as pk
import numpy as np

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.metahandlers.ints import IntRange

from load_ab_data import load_data

### set up grammar
class Array(ABC): pass

@dataclass
class Constant(Array):
    value: Annotated[int, IntRange(-2, 2)]
    # value: float
    # def eval(self, line): return np.array([[self.value]], dtype=float)
    def eval(self, line): return np.array([[self.value]], dtype=float)
    def __str__(self): return str(self.value)

@dataclass
class Dimension(Array):
    def eval(self, line): return np.array([[line[0].shape[1]]], dtype=float)
    def __str__(self): return "N"

@dataclass
class Variable(Array):
    index: Annotated[int, IntRange(0, 2)] # inclusive
    def eval(self, line): return line[self.index]
    def __str__(self): return f"{'wxy'[self.index]}"

@dataclass
class Add(Array):
    left: Array
    right: Array
    def eval(self, line): return self.left.eval(line) + self.right.eval(line)
    def __str__(self): return f"({self.left} + {self.right})"

@dataclass
class Sub(Array):
    left: Array
    right: Array
    def eval(self, line): return self.left.eval(line) - self.right.eval(line)
    def __str__(self): return f"({self.left} - {self.right})"

@dataclass
class Mul(Array):
    left: Array
    right: Array
    def eval(self, line): return self.left.eval(line) * self.right.eval(line)
    def __str__(self): return f"({self.left} * {self.right})"

@dataclass
class Div(Array):
    left: Array
    right: Array
    def eval(self, line):
        left = self.left.eval(line)
        right = self.right.eval(line)
        shape = np.broadcast_shapes(left.shape, right.shape)
        right = np.broadcast_to(right, shape)
        return np.divide(left, right, where=(np.fabs(right) > .01), out=100.*np.sign(right))
    def __str__(self): return f"({self.left} / {self.right})"

@dataclass
class Power(Array):
    base: Array
    expo: Array
    def eval(self, line):
        base = self.base.eval(line).astype(float)
        expo = self.expo.eval(line)

        real = (base >= 0) | (expo == expo.round()) # otherwise numpy returns nan
        out = np.power(base, expo)
        out[~real] = 0.
        small = (np.fabs(out) < 2**10)
        out[~small] = 2**10 * np.sign(out[~small])

        return np.power(base, expo, where=(real & small), out=out)

    def __str__(self): return f"({self.base} ** {self.expo})"

@dataclass
class Maximum(Array):
    left: Array
    right: Array
    def eval(self, line):
        return np.maximum(self.left.eval(line), self.right.eval(line))
    def __str__(self): return f"max({self.left}, {self.right})"

@dataclass
class Minimum(Array):
    left: Array
    right: Array
    def eval(self, line):
        return np.minimum(self.left.eval(line), self.right.eval(line))
    def __str__(self): return f"min({self.left}, {self.right})"

@dataclass
class Dot(Array):
    left: Array
    right: Array
    def eval(self, line): return (self.left.eval(line) * self.right.eval(line)).sum(axis=1, keepdims=True)
    def __str__(self): return f"({self.left} . {self.right})"

@dataclass
class Sign(Array):
    arg: Array
    def eval(self, line): return np.sign(self.arg.eval(line))
    def __str__(self): return f"sign({self.arg})"

@dataclass
class Sqrt(Array):
    arg: Array
    def eval(self, line):
        arg = self.arg.eval(line)
        return np.where(arg >= 0, np.sqrt(arg), 0.)
    def __str__(self): return f"sqrt({self.arg})"

@dataclass
class Log2(Array):
    arg: Array
    def eval(self, line):
        arg = self.arg.eval(line)
        return np.log2(np.fabs(arg), where=(np.fabs(arg) > 2**-10), out=-10*np.ones(arg.shape))
    def __str__(self): return f"log2({self.arg})"

@dataclass
class Sum(Array):
    arg: Array
    def eval(self, line): return self.arg.eval(line).sum(axis=1, keepdims=True)
    def __str__(self): return f"sum({self.arg})"

@dataclass
class Min(Array):
    arg: Array
    def eval(self, line): return self.arg.eval(line).min(axis=1, keepdims=True)
    def __str__(self): return f"{self.arg}.min()"

@dataclass
class Max(Array):
    arg: Array
    def eval(self, line): return self.arg.eval(line).max(axis=1, keepdims=True)
    def __str__(self): return f"{self.arg}.max()"

# Drops all columns but leading to ensure scalar result
@dataclass
class Scalar(Array):
    arg: Array
    def eval(self, line): return self.arg.eval(line)[:,:1]
    def __str__(self): return f"{self.arg}[:,:1]"

### fitness functions

def fitness_function(n: Array, ab):
    # ab = 0 for alpha, 1 for beta
    fitness = 0.
    for line in dataset:
        pred, actual = n.eval(line[:-2]), line[-2+ab]
        fitness += np.fabs(pred - actual).mean() # MAD
    return fitness

def fitness_alpha(n: Array): return fitness_function(n, 0)
def fitness_beta(n: Array): return fitness_function(n, 1)

if __name__ == "__main__":

    dataset = load_data(Ns=[3,4])

    productions = [Constant, Dimension, Variable, Add, Sub, Mul, Div, Power, Maximum, Minimum, Dot, Sign, Sqrt, Log2, Sum, Min, Max]
    grammar = extract_grammar(productions, Scalar) # Scalar ensures that final output is 1D
    print("Grammar: {}.".format(repr(grammar)))
    
    prob = SingleObjectiveProblem(
        minimize=True, # smaller MAD is better
        fitness_function=fitness_alpha,
        # fitness_function=fitness_beta,
    )
    
    alg = SimpleGP(
        grammar,
        problem=prob,
        # probability_crossover=0.4,
        # probability_mutation=0.4,
        number_of_generations=1000,
        max_depth=7,
        population_size=1000,
        selection_method=("tournament", 2),
        n_elites=50,
        n_novelties=100,
        seed = np.random.randint(123456), # 123 for reproducible convergence on perceptron
        target_fitness=0.0,
        # favor_less_complex_trees=True,
    )

    best = alg.evolve()
    print(
        f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype}",
    )

