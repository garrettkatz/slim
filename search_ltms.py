import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

canonical = True

N = 7

X = np.array(tuple(it.product((-1, +1), repeat=N-1))).T
# X = np.vstack((-np.ones(2**(N-1)), X))
# X = np.vstack((+np.ones(2**(N-1)), X[:,::-1]))
X = np.vstack((+np.ones(2**(N-1)), X))

# frontier = set([(-1,)*(2**(N-1))])
frontier = set([(+1,)*(2**(N-1))])
explored = set()
dichotomies = {}

if canonical:
    # A_c = np.eye(N, k=-1) - np.eye(N) # w[0] >= 0, w[i] >= w[i-1]
    A_c = np.eye(N, k=+1) - np.eye(N) # w[-1] >= 0, w[i] >= w[i+1]
    b_c = np.zeros(N)

num_pops = 0
while len(frontier) > 0:
    num_pops += 1
    # y = frontier.pop(0)
    y = frontier.pop()

    if y in explored: continue
    explored.add(y)

    Xy = X * np.array(y)

    A_ub = -Xy.T
    b_ub = -np.ones(Xy.shape[1])
    c = -A_ub.sum(axis=0)

    if canonical:
        A_ub = np.concatenate((A_ub, A_c), axis=0)
        b_ub = np.concatenate((b_ub, b_c))
        c = np.ones(N) # minimum weight objective when all weights positive

    result = linprog(
        c = c,
        A_ub = A_ub,
        b_ub = b_ub,
        bounds = (None, None),
        method='simplex',
        # method='revised simplex', # this and high-ds miss some solutions
    )

    if result.x is not None:
        w = result.x
        if canonical:
            # canonicalize to counteract rounding error in w order and non-negative constraints
            w = np.sort(np.fabs(w))[::-1] # so that all 1 region is a starting point
        # sanity check region constraints
        feasible = (w @ Xy > 0).all()
    else:
        feasible = False

    if not feasible: continue

    dichotomies[y] = w

    print(f"{num_pops} pops, |frontier|={len(frontier)}, |dichotomies| = {len(dichotomies)}")

    for k in range(len(y)):
        # if y[k] == -1:
            # frontier.add(y[:k] + (+1,) + y[k+1:])
        if y[k] == +1:
            frontier.add(y[:k] + (-1,) + y[k+1:])
            # frontier.append(y[:k] + (+1,) + y[k+1:])

print(np.vstack(list(dichotomies.values())))
print(len(dichotomies))

