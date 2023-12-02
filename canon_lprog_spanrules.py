import sys
import pickle as pk
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt
import matplotlib as mp
from load_ltm_data import *

np.set_printoptions(threshold=10e6, linewidth=1000)
mp.rcParams['font.family'] = 'serif'

if __name__ == "__main__":
    
    N = int(sys.argv[1]) # dim
    eps = 1. # constraint slack threshold
    do_i0 = False # whether to constrain W[i0] == Wc[i0], seems feasible until N=7

    Yc, Wc, X, Ac = load_ltm_data([N])
    Yc, Wc, X, Ac = Yc[N], Wc[N], X[N], Ac[N]

    # canonical index to keep fixed
    i0 = 0
    # i0 = (Wc == np.eye(N)[-1]).all(axis=1).argmax() # infeasible at N=7
    if do_i0: eps = 0.0 # may need to be smaller since the solver cannot scale w0

    # remove redundant adjacencies for better numerics
    Ac = list(set([(min(i,j), max(i,j), k) for (i,j,k) in Ac]))

    # load all neighbors of canonical regions to remove redundant region constraints
    with open(f"adjs_{N}_c.npz", "rb") as f: (Yn, _) = pk.load(f)

    R = len(Wc) # num dichotomies
    E = len(Ac) # number of adjacency constraints

    # setup region constraints
    blocks = []
    for r in range(R):
        # extract irredundant boundary vertices
        k = (Yn[r] != Yc[r]).any(axis=0)
        blocks.append(sp.csr_array(-(X[:,k]*Yc[r,k]).T))
    # form per-region block diagonals
    diag = sp.block_diag(blocks, format="csr")
    # pad with zeros for beta variables
    A_ub = sp.csr_array(diag, shape = (diag.shape[0], N*R + E))
    # A_ub = sp.csr_array(sp.block_diag(
    #     [sp.csr_array(-(X*Yc[i]).T) for i in range(R)], format="csr"),
    #     shape = (2**(N-1)*R, N*R + E))
    # ensure positive region constraint slack
    b_ub = -eps*np.ones(A_ub.shape[0])

    # setup objective (opposite constraints to keep problem bounded)
    c = -A_ub.mean(axis=0)
    # print("c:")
    # print(c[:N*R])
    # input('.')

    # setup span constraints
    A_eq_blocks = [[None for _ in range(R+E)] for _ in range(E)]
    e = 0
    for i,j,k in Ac:
        A_eq_blocks[e][i] =  sp.eye(N) # w_i
        A_eq_blocks[e][j] = -sp.eye(N) # -w_j
        A_eq_blocks[e][R+e] = sp.csr_array(X[:,k:k+1]) # + β * x_k
        e += 1
    assert e == E

    # one more block for initial region weights
    if do_i0:
        # print(i0, Wc[i0])
        # input('..')
        A_eq_blocks.append([None for _ in range(R+E)])
        A_eq_blocks[-1][i0] = sp.eye(N)

    A_eq = sp.csr_array(sp.bmat(A_eq_blocks, format="csr")) # w_i - w_j + β * x_k
    b_eq = np.zeros(A_eq.shape[0]) # = 0

    if do_i0:
        b_eq[-N:] = Wc[i0]

    print(f"R = {R} regions, E = {E} adjacencies, N*R = {N*R} weights, N*R+E = {N*R+E} variables, N*E = {N*E} equality constraints, 2**(N-1)*R = {2**(N-1)*R} redundant region constraints")
    print(f"A_ub {A_ub.shape[0]}x{A_ub.shape[1]} vs {2**(N-1) * R}x{N*R + E}, A_eq = {A_eq.shape[0]}x{A_eq.shape[1]}")
    # print(R, E, N*R, 2**(N-1) * R, N*E, N*R + E)

    if N <= 4:
        pt.subplot(1,2,1)
        pt.imshow(A_ub.toarray())
        pt.title("A_ub")
        pt.subplot(1,2,2)
        pt.imshow(A_eq.toarray())
        pt.title("A_eq")
        pt.show()
        # A_ub = A_ub.toarray()
        # A_eq = A_eq.toarray()

    # result = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None), method = "simplex") # simplex doesn't do sparse
    result = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds = (None, None), method = "highs", options={"disp": True})
    print(result.message)

    W = result.x[:R*N].reshape((R, N))
    β = result.x[R*N:]

    # double-check region constraints
    region = np.empty(R, dtype=bool)
    for i in range(R):
        # region[i] = (np.sign(W[i] @ X) == Yc[i]).all()
        region[i] = ((W[i] @ X)*Yc[i] >= eps).all()
    print(f"region constraints all satisfied: {region.all()}")

    # double-check span constraints
    span = np.empty(E, dtype=bool)
    for e, (i,j,k) in enumerate(Ac):
        span[e] = np.allclose(W[i] + β[e] * X[:,k], W[j])
    print(f"span constraints satisfied: {span.all()}")

    print("\nWc:")
    print(Wc.round(2))

    print("\nW*:")
    print(W.round(2))

    print("\nβ:")
    print(β.round(4))

    print("\nmin slack:")
    print(((W @ X) * Yc).min())

    dots = np.array([(Wc[i] * np.maximum(0., X[:,k])).sum() for (i,j,k) in Ac])

    # pt.plot(β, 'k.')
    # pt.plot(dots, β, 'k.')
    pt.plot(np.sort(β), 'k.')
    pt.xlabel("Sort index")
    pt.ylabel("β")
    pt.show()


