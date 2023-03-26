import pickle as pk
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt
import matplotlib as mp

np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'

if __name__ == "__main__":
    
    N = 4 # dim
    eps = 0.001 # constraint slack threshold

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    R = len(W) # num dichotomies

    # get adjacencies
    # A, K = adjacency(Y, sym=True)
    A, K = adjacency(Y, sym=False) # avoid redundancies
    E = sum(map(len, A.values())) # number of adjacency constraints

    A_ub = sp.csr_array(sp.block_diag(
        [sp.csr_array(-(X*Y[i]).T) for i in range(R)], format="csr"),
        shape = (2**(N-1)*R, N*R + E))
    b_ub = -eps*np.ones(A_ub.shape[0])
    c = -A_ub.sum(axis=0)

    A_eq_blocks = [[None for _ in range(R+E)] for _ in range(E)]
    e = 0
    for i in A:
        for j, k in zip(A[i], K[i]):
            A_eq_blocks[e][i] =  sp.eye(N)
            A_eq_blocks[e][j] = -sp.eye(N)
            A_eq_blocks[e][R+e] = sp.csr_array(X[:,k:k+1])
            e += 1
    assert e == E
    A_eq = sp.csr_array(sp.bmat(A_eq_blocks, format="csr"))
    b_eq = np.zeros(A_eq.shape[0])

    print(A_ub.shape, A_eq.shape)
    print(R, E, N*R, 2**(N-1) * R, N*E, N*R + E)

    if N <= 4:
        pt.subplot(1,2,1)
        pt.imshow(A_ub.toarray())
        pt.subplot(1,2,2)
        pt.imshow(A_eq.toarray())
        pt.show()
        # A_ub = A_ub.toarray()
        # A_eq = A_eq.toarray()

    # result = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None), method = "highs")
    result = linprog(c, A_ub, b_ub, bounds = (None, None), method = "highs")
    print(result.message)

    W = result.x[:R*N].reshape((R, N))
    β = result.x[R*N:]

    feasible = np.empty(R, dtype=bool)
    for i in range(R):
        # feasible[i] = (np.sign(W[i] @ X) == Y[i]).all()
        feasible[i] = (np.sign(W[i] @ X)*Y[i] >= eps).all()
    print(feasible.all())

    # W[k] = result.x
    # feasible[k] = (np.sign(W[k] @ X[:,:j+1]) == y).all()

