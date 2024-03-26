import itertools as it
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as pt
import matplotlib.patches as mp 
from matplotlib import rcParams
from check_span_rule import *

# rcParams['font.family'] = 'serif'
# rcParams['text.usetex'] = True

solver = "GUROBI"
N = 8

# load the regions and boundaries
with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

# sort ascending by number of boundaries
sorter = np.argsort(B.sum(axis=1))

# extract the counterexample
sample = sorter[[215, 503]] # indices after sorting

# keep all but the dichotomies in the counterexample
keep = np.ones(len(Y), dtype=bool)
keep[sample] = False

Y = Y[keep]
B = B[keep]
W = W[keep]

# double-check infeasibility
result = check_span_rule(X, Y, B, W, solver=solver, verbose=True)
status, u, g, D, E = result
print(status)

