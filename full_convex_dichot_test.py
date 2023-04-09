import numpy as np
from numpy.linalg import norm
from adjacent_ltms import adjacency
import itertools as it
import matplotlib.pyplot as pt
import matplotlib as mp
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'

if __name__ == "__main__":

    # N = 4
    # num_reps = 100

    N = 5
    num_reps = 50

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    A, K = adjacency(Y, sym=False) # sym=False just halves the objective function

    cdiffs = np.empty(num_reps)
    for rep in range(num_reps):

        # two random feasible weight sets
        W = {q: np.empty((len(Y), N)) for q in (0, 1)}
        for q, i in it.product((0, 1), range(len(Y))):

            w = linprog(
                c = (X * Y[i]).sum(axis=1),
                A_ub = -(X * Y[i]).T,
                b_ub = -(1 + np.random.rand(2**(N-1))),
                bounds = (None, None),
            ).x

            feasible = (np.sign(w @ X) == Y[i]).all()
            if not feasible: print("unfeas!")
            assert feasible

            W[q][i] = w

        # random affine combination
        a = np.random.rand()
        W[2] = a*W[0] + (1-a)*W[1]

        # evaluate function at all three
        f = np.zeros(3)
        for q, i in it.product((0, 1, 2), range(len(Y))):
            for j, k in zip(A[i], K[i]):
                # if np.random.rand() < 0.9: continue # not convex after this, need them all

                # wi, wj, xk = W[q][i], W[q][j], X[:,k]
                # Pk = np.eye(N) - xk[:,np.newaxis] * xk / N
                # # f[q] += norm(wi)*norm(wj) - wi @ Pk @ wj # this is a bug!!! need Pk in the norms too!
                # f[q] += norm(wi @ Pk)*norm(wj @ Pk) - wi @ Pk @ wj

                # less computation
                wi, wj, xk = W[q][i], W[q][j], X[:,k]
                wiPk = wi - (wi @ xk) * xk / N
                wjPk = wj - (wj @ xk) * xk / N
                f[q] += norm(wiPk)*norm(wjPk) - wiPk @ wjPk

        # convex?
        fa = f[2]
        af = a*f[0] + (1-a)*f[1]

        # # convex:
        # fa <= af
        # 0 <= af - fa
        print(rep, f[0], f[1], af, fa, af - fa)
        cdiffs[rep] = af - fa
        if cdiffs[rep] < 0:
            input(":_ _ _ _ (")

    print("cdiffs (should be non-negative):")
    print(cdiffs)
    if (cdiffs >= 0).all():
        print("no violation!")
    else:
        print("most violation:", cdiffs[cdiffs < 0].min())
    print("max abs:", np.fabs(cdiffs).max())




