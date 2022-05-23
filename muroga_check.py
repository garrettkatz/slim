from scipy.optimize import linprog, OptimizeWarning
import numpy as np
import itertools as it
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

X = np.array(tuple(it.product((-1, 1), repeat=9))).T

# check multiple extrema of Muroga 1970
wa = np.array([13, 7, 6, 6, 4, 4, 4, 3, 2])
wb = np.array([13, 7, 6, 6, 4, 4, 4, 2, 3])
ya = np.sign(wa @ X)
yb = np.sign(wb @ X)
print((ya == yb).all())

# Sanity check by linprog
A = X[:, ya < 0].T
b = -np.ones(2**8)
c = np.ones(9)
result = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
print(result.x)

# Linprog the boundaries
Xh = X[:, ya < 0].T
boundaries = np.zeros(2**8, dtype=bool)
for b in range(2**8):
    others = list(range(b)) + list(range(b+1, 2**8))
    result = linprog(
        # c = np.ones(9),
        c = -Xh[others].mean(axis=0),
        A_eq = Xh[b].reshape(1,-1),
        b_eq = np.zeros(1),
        A_ub = Xh[others],
        b_ub = -np.ones(2**8 - 1),
        bounds=(None, None))

    w = result.x
    boundaries[b] = (Xh[others] @ w <= -.99).all() and np.fabs(Xh[b] @ w) <= 0.01
    print(f"{b} of {2**8}, {boundaries.sum()} boundaries")

w = np.linalg.lstsq(Xh[boundaries], np.ones(boundaries.sum()), rcond=None)[0]
print(w)
print(Xh[boundaries] @ w)
print(np.fabs((Xh[boundaries] @ w) - np.ones(boundaries.sum())).max())

# looks like w @ Xb = 1 does not generalize past N=7.

