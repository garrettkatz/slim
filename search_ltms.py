import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

N = 5

X = np.array(tuple(it.product((-1, +1), repeat=N-1))).T
X = np.vstack((np.ones(2**(N-1)), X))

frontier = set([(-1,)*(2**(N-1))])
explored = {}

num_pops = 0
while len(frontier) > 0:
    num_pops += 1
    # y = frontier.pop(0)
    y = frontier.pop()
    if y in explored: continue

    Xy = X * np.array(y)
    result = linprog(
        c = Xy.sum(axis=1),
        A_ub = -Xy.T,
        b_ub = -np.ones(Xy.shape[1]),
        bounds = (None, None),
    )
    if result.x is not None:
        w = result.x
        feasible = (w @ Xy > 0).all() # sanity check
    else:
        feasible = False

    if not feasible: continue

    explored[y] = w

    print(f"{num_pops} pops, |frontier|={len(frontier)}, |explored| = {len(explored)}")

    for k in range(len(y)):
        if y[k] == -1:
            frontier.add(y[:k] + (+1,) + y[k+1:])
            # frontier.append(y[:k] + (+1,) + y[k+1:])

print(len(explored))

