# Check that for every region, there is a weight vector w_i, such that:
#     w_i X y_i >= eps (region constraint for i)
#     for every j,k adjacency to i, there is a scalar b, such that:
#         (w_i + b*x_k) X y_j >= eps (region constraint for j)

import sys
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import scipy.sparse as sp
from scipy.optimize import linprog
from load_ltm_data import *

if __name__ == "__main__":
    
    N = int(sys.argv[1]) # dim
    eps = 1. # constraint slack threshold

    # load all canonical regions
    Yc, Wc, X, _ = load_ltm_data([N])
    Yc, Wc, X = Yc[N], Wc[N], X[N]

    # load all neighbors of canonical regions
    with open(f"adjs_{N}_c.npz", "rb") as f: (Yn, _) = pk.load(f)

    # for every region
    for i in range(len(Yc)):

        # region constraint for i
        A_i = sp.csr_array(-(X*Yc[i]).T)
        blocks = [[A_i] + [None]*len(Yn[i])] # first row

        # for every j,k adjacency to i
        for j in range(len(Yn[i])):
            k = (Yn[i][j] != Yc[i]).argmax()

            row = [None]*(1+len(Yn[i]))
            row[0] = sp.csr_array(-(X*Yn[i][j]).T)
            row[1+j] = sp.csr_array(-(X*Yn[i][j]).T @ X[:,k:k+1])

            blocks.append(row)

        # inequality region constraints
        A_ub = sp.bmat(blocks, format='csr')
        b_ub = -eps*np.ones(A_ub.shape[0])

        # if N <= 4:
        #     pt.imshow(A_ub.toarray())
        #     pt.show()

        # setup objective (opposite constraints to keep problem bounded)
        c = -A_ub.mean(axis=0)

        result = linprog(c, A_ub, b_ub, bounds = (None, None), method = "highs", options={"disp": False})
        print(f"Region {i} of {len(Yc)}: {result.message}")
        assert result.success

        w_i, b = result.x[:N], result.x[N:]
        assert (np.sign(w_i @ X) == Yc[i]).all()

        for j in range(len(Yn[i])):

            k = (Yn[i][j] != Yc[i]).argmax()
            w_j = w_i + b[j]*X[:,k]
            assert (np.sign(w_j @ X) == Yn[i][j]).all()

