from abc import ABC
from dataclasses import dataclass
from typing import Annotated

import numpy as np

from geneticengine.algorithms.gp.simplegp import SimpleGP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.metahandlers.ints import IntRange

### load data

from svm_data import all_transitions

def load_data(Ns, perceptron=False):

    dataset = []
    for N in Ns:
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X, Y = {N: ltms["X"]}, {N: ltms["Y"]}
        w_new, w_old, x, y, margins = all_transitions(X, Y, N)
        
        w_new = np.concatenate(w_new, axis=0)
        w_old = np.concatenate(w_old, axis=0)
        x = np.stack(x).astype(float)
        y = np.stack(y).astype(float).reshape(-1, 1)
        margins = np.stack(margins).reshape(-1, 1)
        
        # sanity-check: try to recover perceptron instead of svm w_new
        if perceptron:
            w_new = w_old + (y - np.sign((w_old*x).sum(axis=1,keepdims=True)))*x
            w_new /= np.maximum(np.linalg.norm(w_new, axis=1, keepdims=True), 10e-8)
    
        dataset.append([w_old, x, y, w_new, margins])

    return dataset

### set up grammar
class Scalar(ABC): pass
class Vector(ABC): pass

@dataclass
class Value(Scalar):
    value: Annotated[int, IntRange(-1, 1)]
    # value: float
    def __str__(self): return str(self.value)
    def pretty(self): return str(self)

@dataclass
class VectorVar(Vector):
    index: Annotated[int, IntRange(0, 1)] # inclusive
    def __str__(self): return f"line[{self.index}]"
    def pretty(self): return f"{'wx'[self.index]}"

@dataclass
class ScalarVar(Scalar):
    index: Annotated[int, IntRange(2, 2)]
    def __str__(self): return f"line[{self.index}]"
    def pretty(self): return "y"

@dataclass
class Sub(Scalar):
    left: Scalar
    right: Scalar
    def __str__(self): return f"({self.left} - {self.right})"
    def pretty(self):return f"({self.left.pretty()} - {self.right.pretty()})"

@dataclass
class Dot(Scalar):
    left: Vector
    right: Vector
    def __str__(self): return f"({self.left} * {self.right}).sum(axis=1, keepdims=True)"
    def pretty(self): return f"dot({self.left.pretty()},{self.right.pretty()})"

@dataclass
class Sign(Scalar):
    arg: Scalar
    def __str__(self): return f"np.sign({self.arg})"
    def pretty(self): return f"sign({self.arg.pretty()})"

@dataclass
class SpanRule(Vector):
    alpha: Scalar
    beta: Scalar
    def __str__(self): return f"({self.alpha}*line[0] + {self.beta}*line[1])"
    def pretty(self): return f"{self.alpha.pretty()}*w + {self.beta.pretty()}*x"

def fitness_function(n: SpanRule):
    code = f"lambda line: {str(n)}"
    rule = eval(code)
    fitness = 0.
    for line in dataset:
        w_pred, w_new, margins = rule(line), line[-2], line[-1]
        w_pred /= np.maximum(np.linalg.norm(w_pred, axis=1, keepdims=True), 10e-8)
        fitness += (w_pred * w_new).sum(axis=1).mean() # cosine similarity
    return fitness / len(dataset)


if __name__ == "__main__":

    dataset = load_data(Ns = [3, 4], perceptron=True)
    
    grammar = extract_grammar([Value, ScalarVar, VectorVar, Sub, Dot, Sign], SpanRule)
    print("Grammar: {}.".format(repr(grammar)))
    
    prob = SingleObjectiveProblem(
        minimize=False,
        fitness_function=fitness_function,
    )
    
    alg = SimpleGP(
        grammar,
        problem=prob,
        probability_crossover=0.4,
        probability_mutation=0.4,
        number_of_generations=20,
        max_depth=5,
        population_size=200,
        selection_method=("tournament", 2),
        n_elites=15,
        seed = np.random.randint(123456), # 123 for reproducible convergence on perceptron
        target_fitness = 0.999999,
    )
    best = alg.evolve()
    print(
        f"Fitness of {best.get_fitness(prob)} by genotype: {best.genotype}\nwith phenotype: {best.get_phenotype().pretty()}",
    )
    
