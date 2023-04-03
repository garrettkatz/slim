import numpy as np
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

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # get adjacencies
    A, K = adjacency(Y, sym=True)

    for rep in range(10):
        print(rep)

        i = np.random.randint(len(A))
        m = np.random.randint(len(A[i]))
        j, k = A[i][m], K[i][m]

        Pk = np.eye(N) - X[:,k:k+1] * X[:,k:k+1].T / N 

        wP = {}
        for (p,q) in it.product((0, 1), (i, j)):

            w = linprog(
                c = (X * Y[q]).sum(axis=1) * np.random.rand(N),
                A_ub = -(X * Y[q]).T,
                b_ub = -np.ones(2**(N-1)),
                bounds = (None, None),
            ).x.reshape(N, 1)
            wP[p,q] = Pk @ w

        a = np.random.rand()
        wP[2,i] = a*wP[0,i] + (1-a)*wP[1,i]
        wP[2,j] = a*wP[0,j] + (1-a)*wP[1,j]

        d0 = wP[0,i].T @ wP[0,j] / (np.linalg.norm(wP[0,i]) * np.linalg.norm(wP[0, j]))
        d1 = wP[1,i].T @ wP[1,j] / (np.linalg.norm(wP[1,i]) * np.linalg.norm(wP[1, j]))
        d2 = wP[2,i].T @ wP[2,j] / (np.linalg.norm(wP[2,i]) * np.linalg.norm(wP[2, j]))
        da = a*d0 + (1-a)*d1

        assert d2 >= da



