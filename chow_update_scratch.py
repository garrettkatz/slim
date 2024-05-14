import sys
import pickle as pk
import itertools as it
import numpy as np
import cvxpy as cp

np.set_printoptions(linewidth=1000)

N = int(sys.argv[1])
solver = 'GUROBI'

with np.load(f"chow_regions_{N}_{solver}.npz") as regions:
    X, Y, B, W, C = (regions[key] for key in ("XYBWC"))

with open(f"canon_adjacencies_{N}.pkl","rb") as f: A = pk.load(f)

print("i,j,k, W[i], W[j], Y[i,k], X[k], Y[j,k], W[j] - W[i]")
for (i,j,k) in sorted(A):
    print(i,j,k, W[i], W[j], Y[i,k], X[k], Y[j,k], (W[j] - W[i]).round(1))

    ### update attempts

    # ## projection into linear space of chow constraints and required X[k] output
    # ## fails the assertion in N=4

    # # chow equality constraints
    # A = np.eye(N) - np.eye(N,k=1)
    # idx = np.flatnonzero(C[j,:-1] == C[j,1:])
    # if C[j,-1] == 0: idx = np.append(idx, N-1)

    # # required X[k] output
    # A = np.concatenate((A[idx], X[[k]]), axis=0)

    # # 0 on chow, +/- 1 on output
    # b = np.zeros(len(A))
    # b[-1] = Y[j,k]

    # # solve linear equalities for smallest  delta d, where
    # # A(wi + d) = b -> Ad = b - Awi
    # d = np.linalg.lstsq(A, b - A @ W[i], rcond=None)[0]
    # wj = (W[i] + d).round(1)

    # ## satisfy X[k] and maintain x' with hamming distance 1 from X[k]
    # ## satisfies assertion on N=4 but not chow symmetry
    # ## fails the assertion on N=5

    # # ham dist 1 vectors
    # A = X[k] * (1 - 2*np.eye(N))
    # A[0] = X[k] # except don't flip first bit in half-cube

    # # maintain same dots as W[i]
    # b = A @ W[i]
    # b[0] *= -1 # except X[k] must flip

    # # solve it: A wj = b
    # wj = np.linalg.lstsq(A, b, rcond=None)[0].round()

    # ## satisfy chow, and for remaining DoFs, flip bits for lowest w's
    # ## fails assert at N=4 and has issues, like invariance when low W[i] are 0
    # ## should refine this to account for chow symmetries, and try to iterate bit flips in order of changing effect on dots with w

    # # chow constraints
    # A_chow = np.eye(N) - np.eye(N,k=1)
    # idx = np.flatnonzero(C[j,:-1] == C[j,1:])
    # if C[j,-1] == 0: idx = np.append(idx, N-1)
    # A_chow = A_chow[idx]
    # b_chow = np.zeros(len(A_chow))

    # # low bit constraints
    # A_low = np.empty((N-len(A_chow), N))
    # b_low = np.empty(N - len(A_chow))
    # for n in range(len(A_low)):
    #     A_low[n] = X[k]
    #     A_low[n,-n-1:] *= -1
    #     b_low[n] = A_low[n] @ W[i]

    # A = np.concatenate((A_chow, A_low), axis=0)
    # b = np.concatenate((b_chow, b_low))

    # # solve it: A wj = b
    # wj = np.linalg.lstsq(A, b, rcond=None)[0].round()

    ## satisfy chow, canonical, projected dot, new dot, and minimize weight
    ## satisfies assert at N=4 but fails N=5, refine projected dot constraint?
    
    u = cp.Variable(N)

    # enforce chow symmetries
    n = np.flatnonzero(C[j,:-1] == C[j,1:])
    chow_constraints = [u[n] == u[n+1]]
    if C[j,-1] == 0: chow_constraints.append(u[-1] == 0)

    # stay canonical
    canonical_constraints = [u[-1] >= 0, u[:-1] >= u[1:]]

    # projected dot
    wiP = W[i] - (W[i] @ X[k]) * X[k] / N
    # dot_constraints = [wiP @ u >= 3.75, X[k] @ u >= 1]
    dot_constraints = [wiP @ u >= ((N+1)/N - 2**(2-N)), X[k] @ u >= 1]

    constraints = chow_constraints + canonical_constraints + dot_constraints
    objective = cp.Minimize(cp.sum(u))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    print(u.value)
    wj = u.value

    print('  wiP    ',wiP)

    ### compare result with desired
    print('  wj     ',wj)
    print('  x wj   ', (X @ wj).round(1))
    print('  s(x wj)', np.sign(X @ wj).astype(int))
    print('  yj     ', Y[j])
    assert (np.sign(X @ wj) == Y[j]).all()


