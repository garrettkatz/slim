import pickle as pk
import sys
from collections import deque
import numpy as np
import cvxpy as cp
import load_ltm_data as ld

# @profile
def main():

    # solver = 'GLPK'
    solver = 'CBC'
    # solver = 'GLOP'
    # solver = 'SCIPY'

    do_opt = True
    verbose = True
    eps = 1

    # input dimension for optimization
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 8

    # load canonical regions and adjacencies
    Yc, Wc, X, Ac = ld.load_ltm_data(N)
    A = ld.organize_by_source(Ac)
    with open(f"adjs_{N}_c.npz", "rb") as f: (Yn, _) = pk.load(f)

    # number of regions
    R = Yc.shape[0]

    # set up boundary indices to remove redundant region constraints
    K = {}
    for i in Yn: K[i] = (Yc[i] != Yn[i]).argmax(axis=1)

    # compute adjacency graph spanning tree
    # i0 = 0 # root region
    i0 = 1 # region containing identity row weight vector
    queue = deque([(i0, None)]) # (region, edge that reached it)
    explored = {} # {region: edge that reached it}
    while len(queue) > 0:
        i, a = queue.popleft() # BFS
        if i in explored: continue
        explored[i] = a
        # print(f"{len(explored)} of {Yc.shape[0]} explored")
        for (j,k) in A[i]: queue.append((j, (i,j,k)))

    # At = explored
    At = [a for a in explored.values() if a is not None]
    print(f"|Ac|={len(Ac)}, |At|={len(At)}, R = {R}")
    # print(At)

    ## variables
    w = cp.Variable((R, N)) # weight vector per region
    β = cp.Variable(len(At)) # beta per spanning tree edge

    ## region constraints
    region_constraints = [
        w[i:i+1] @ (X[:, K[i]] * Yc[i, K[i]]) >= eps
        for i in range(R)]

    ## span constraints
    span_constraints = [
        w[j] == w[i] + β[e] * (X[:,k] * Yc[j, k])
        for e, (i,j,k) in enumerate(At)]

    ## objective to bound problem
    c = np.stack([
        (X[:, K[i]] * Yc[i, K[i]]).mean(axis=1)
        for i in range(R)])
    objective = cp.Minimize(cp.sum(cp.multiply(w, c)))

    # saved results here
    fname = f"ab_necessary_lp_{N}_{solver}.pkl"

    if do_opt:

        caps = [len(span_constraints)] if N < 8 else [111, 112] # becomes infeasible at 112

        for cap in caps:
            problem = cp.Problem(objective, region_constraints + span_constraints[:cap])
            # problem = cp.Problem(objective, region_constraints)
            problem.solve(solver=solver, verbose=True)
            if problem.status == 'infeasible':
                print(f'reached infeasibility at cap={int(100*cap/len(span_constraints))}%')
                break
            with open(fname, 'wb') as f: pk.dump((problem.status, w.value, β.value), f)
    
    with open(fname, 'rb') as f: status, w, β = pk.load(f)

    if w is not None: print(w.round(3))
    if β is not None: print(β.round(3))

    print(status)


if __name__ == "__main__": main()

