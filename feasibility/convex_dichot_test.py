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

    N = 4
    num_reps = 10

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # get adjacencies
    A, K = adjacency(Y, sym=True)

    tolerance = 0.0001

    cdiffs = np.empty(num_reps)
    for rep in range(num_reps):

        i = np.random.randint(len(A))
        m = np.random.randint(len(A[i]))
        j, k = A[i][m], K[i][m]

        Pk = np.eye(N) - X[:,k:k+1] * X[:,k:k+1].T / N 

        wP = {}
        for (p,q) in it.product((0, 1), (i, j)):

            w = linprog(
                c = (X * Y[q]).sum(axis=1),
                A_ub = -(X * Y[q]).T,
                b_ub = -np.random.rand(2**(N-1)),
                bounds = (None, None),
            ).x

            feasible = (np.sign(w @ X) == Y[q]).all()
            if not feasible:
                print("unfeas!")
            assert feasible

            wP[p,q] = Pk @ w

        a = np.random.rand()
        wP[2,i] = a*wP[0,i] + (1-a)*wP[1,i]
        wP[2,j] = a*wP[0,j] + (1-a)*wP[1,j]

        # # concave?
        # d0 = (wP[0,i].T @ wP[0,j] / (norm(wP[0,i]) * norm(wP[0, j])))**2
        # d1 = (wP[1,i].T @ wP[1,j] / (norm(wP[1,i]) * norm(wP[1, j])))**2
        # d2 = (wP[2,i].T @ wP[2,j] / (norm(wP[2,i]) * norm(wP[2, j])))**2
        # da = a*d0 + (1-a)*d1
        # if not (d2 >= da):
        #     print(d2, da)
        # assert d2 >= da

        # # convex?
        # f0 = (norm(wP[0,i]) * norm(wP[0, j]))**2 - (wP[0,i].T @ wP[0,j])**2
        # f1 = (norm(wP[1,i]) * norm(wP[1, j]))**2 - (wP[1,i].T @ wP[1,j])**2
        # fa = (norm(wP[2,i]) * norm(wP[2, j]))**2 - (wP[2,i].T @ wP[2,j])**2
        # af = a*f0 + (1-a)*f1

        # convex?
        f = lambda u, v: norm((u / norm(u)) - (v / norm(v)))**2 # this is just 2 - cosine sim
        f0 = f(wP[0,i], wP[0,j])
        f1 = f(wP[1,i], wP[1,j])
        fa = f(wP[2,i], wP[2,j])
        af = a*f0 + (1-a)*f1

        # # convex:
        # fa <= af
        # 0 <= af - fa

        print(rep, f0, f1, fa, af, fa - af)
        cdiffs[rep] = fa - af

    print("cdiffs (should be non-negative):")
    print(cdiffs)
    print("most violation:", cdiffs[cdiffs < 0].min())
    print("max abs:", np.fabs(cdiffs).max())


