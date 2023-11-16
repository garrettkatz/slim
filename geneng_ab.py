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

# import all arithmetic operators
from geneng_array import *

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

    dataset = load_data(Ns=[3])

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

