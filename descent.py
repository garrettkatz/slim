import os, sys
from collections import deque
import itertools as it
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

N = int(sys.argv[1])
fname = f"hemigraph_{N}.npz"

X = np.array(tuple(it.product((-1, +1), repeat=N))).T
Nh = 2**(N-1)
Xh = X[:,:Nh]

npz = np.load(fname)
hemis, depths, boundaries, weights, anchors = npz["hemis"], npz["depths"], npz["boundaries"], npz["weights"], npz["anchors"]

deltas = np.zeros((N + N*(N-1)//2, N), dtype=int)
for d,(i,j) in enumerate(it.combinations_with_replacement(range(N), 2)):
    deltas[d,i] += 1
    deltas[d,j] += 1
# deltas = np.eye(N, dtype=int)
# print(deltas)
# input('.')

def descent(w, x):
    if np.fabs(w @ x) != 1: return False, w, {}
    if w @ x == -1: x = -x


    u = {0: N*(N-2)*w - (N-1)*x}
    # move to odd lattice if necessary
    if N % 2 == 0:
        steps = u[0] - np.diag(np.sign(u[0]))
        s = (steps @ u[0] / (steps**2).sum(axis=1)).argmax()
        u[0] = steps[s]

    # for t in it.count():
    for t in range(20):
        print(f" {t}: u =", u[t])
        steps = u[t] - np.sign(u[t])*deltas
        print(f" steps:")
        print("", steps.T)
        valid = (steps @ x <= -1) & (steps @ w >= 1) & (steps @ x >= u[t] @ x)
        u0sim = steps @ u[0] / (steps**2).sum(axis=1)
        print(" satisfy dot with x")
        print("", steps @ x <= -1)
        print(" contract dot with x")
        print("", steps @ x >= u[t] @ x)
        print(" satisfy dot with w")
        print("", steps @ w >= 1)
        print(" step norms")
        print("", (steps**2).sum(axis=1))
        print(" step sim with u0")
        print("", u0sim)
        print(" vs ut sim with u0:", u[t] @ u[0] / (u[t]**2).sum())

        if not valid.any() or (t > 0 and (u[t] == u[t-1]).all()):
            success = (u[t] @ x == -1)
            return success, u[t], u

        # s = (steps[valid]**2).sum(axis=1).argmin()
        s = u0sim[valid].argmax()
        u[t+1] = steps[valid][s]

    return False, u[len(u)-1], u

for a,anchor in enumerate(np.flatnonzero(anchors)):
    w = weights[anchor]
    h = hemis[anchor]
    for b,k in enumerate(np.flatnonzero(boundaries[anchor])):

        print(f"{a},{b} of {len(anchors)},{boundaries[anchor].sum()}:")

        hf = h.copy()
        hf[k] *= -1
        x = Xh[:,k]
        n = (hf == hemis).all(axis=1).argmax()
        wn = weights[n]
        assert (np.sign(wn @ Xh) == hf).all()

        print("w", w)
        print("x", x)
        print("wn", wn)
        # print("wn @ Xh", (wn @ Xh))
        # print("hf", hf)

        if w @ x < 0: x = -x

        success, wk, u = descent(w, x)

        correct_region = (np.sign(wk @ Xh) == hf).all()
        canonical = (wk == wn).all()
        
        print(f"success = {success}")
        print(f"{len(u)} steps")
        print(f"satisfies new region: {correct_region}")
        print(f"reached canonical: {canonical}")

        print("w", w)
        print("x", x)
        print("wn", wn)
        
        if not (correct_region and canonical): input('..')

