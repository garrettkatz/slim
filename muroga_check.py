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
wc = np.array([13, 7, 6, 6, 4, 4, 4, 2.5, 2.5])
ya = np.sign(wa @ X)
yb = np.sign(wb @ X)
yc = np.sign(wc @ X)
print((np.fabs(ya) == 1).all())
print((ya == yb).all())
print((ya == yc).all())

print("wa|wb, wc norms:")
print((wb**2).sum())
print((wc**2).sum())

# check unique fractional weights 
wa = np.array([14.5, 12.5, 9.5, 7.5, 6, 4, 4, 1.5, 1.5])
ya = np.sign(wa @ X)

# Sanity check by linprog
A = X[:, ya < 0].T
b = -np.ones(2**8)
c = np.ones(9)
result = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method='simplex')
wlp = result.x

print("lp'd w:")
print(wlp)

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
        bounds=(None, None),
        method='highs',
        # method='simplex',
        )

    w = result.x
    # print(result.message)
    if w is not None:
        boundaries[b] = ((Xh[others] @ w) <= -.99).all() and np.fabs(Xh[b] @ w) <= 0.01
    print(f"{b} of {2**8}, {boundaries.sum()} boundaries")

print("lp'd w:")
print(wlp)

w = np.linalg.lstsq(Xh[boundaries], np.ones(boundaries.sum()), rcond=None)[0]
print("lstsq'd w, dot with boundaries, max residual:")
print(w)
print(Xh[boundaries] @ w)
print(np.fabs((Xh[boundaries] @ w) - np.ones(boundaries.sum())).max())

print("wa, wb, wc dots with boundaries:")
print(Xh[boundaries] @ wa)
print(Xh[boundaries] @ wb)
print(Xh[boundaries] @ wc)
print("wa, wb, wc min abs dots with non-boundaries:")
print(np.fabs(Xh[~boundaries] @ wa).min())
print(np.fabs(Xh[~boundaries] @ wb).min())
print(np.fabs(Xh[~boundaries] @ wc).min())

# looks like w @ Xb = 1 does not generalize past N=7.
# however, perhaps w @ Xb <= 2, w @ X~b >= 3? maybe minimum L2 norm w?
# also wonder if wc is invariant under hypercube symmetries that leave its region invariant. wa and wb definitely are not.


