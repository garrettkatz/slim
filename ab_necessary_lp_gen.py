import pickle as pk
import sys
from collections import deque
import numpy as np
import cvxpy as cp
import load_ltm_data as ld

# @profile
def main():

    # solver = 'GLPK'
    # solver = 'CBC'
    # solver = 'GLOP'
    solver = 'SCIPY'

    do_opt = True
    verbose = True
    eps = 1

    # input dimension for optimization
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 8

    # load canonical regions and adjacencies
    Yc, _, X, Ac = ld.load_ltm_data(N)
    A = ld.organize_by_source(Ac)
    with open(f"adjs_{N}_c.npz", "rb") as f: (Yn, _) = pk.load(f)

    # number of regions
    R = Yc.shape[0]

    # set up boundary indices to remove redundant region constraints
    K = {}
    for i in Yn: K[i] = (Yc[i] != Yn[i]).argmax(axis=1)

    # set up permutation of vertices
    perm = np.arange(X.shape[1]) # identity permutation
    # perm = np.random.permutation(X.shape[1]) # random

    # build binary label tree
    print("Building tree...")
    nodes = [(None, np.empty((N,0)), np.empty(0))] # parent index, x data, y data
    lookup = {nodes[0][2].tobytes(): 0} # maps y data bytes to node index
    max_k = X.shape[1] if N < 8 else 2*X.shape[1]//3 # limit size of optimization problem
    try halving Y.shape[0] instead of X, full dichotomies but fewer of them
    for k in range(1, max_k):

        # current set of input examples
        Xk = X[:,perm[:k]]

        # all output label possibilities
        Yk = np.unique(Yc[:, perm[:k]], axis=0)

        # each possibility has a node at depth k
        for i, yk in enumerate(Yk):

            # parent index
            p = lookup[yk[:-1].tobytes()]

            # new node index
            n = len(nodes)

            # save the node
            nodes.append((p, Xk, yk))
            lookup[yk.tobytes()] = n

    # for n, (p, x, y) in enumerate(nodes): print(n, p, x.shape, y)
    # input('.')

    # tree edge data
    At = [
        (n, p, x[:,-1], y[-1])
        for n, (p, x, y) in enumerate(nodes)
        if p is not None]

    print(f"{len(nodes)} nodes, {len(At)} edges")

    ## variables
    w = cp.Variable((len(nodes), N)) # weight vector per node
    β = cp.Variable(len(At)) # beta per spanning tree edge (nodes - 1)

    ## data constraints
    print("Building sample constraints...")
    sample_constraints = [
        w[n:n+1] @ (Xk * yk) >= eps
        for n, (p, Xk, yk) in enumerate(nodes)
        if p is not None] # no constraints on root

    ## span constraints
    print("Building span constraints...")
    span_constraints = [
        w[n] == w[p] + β[e] * (x * y)
        for e, (n,p,x,y) in enumerate(At)]

    ## objective to bound problem
    print("Building objective...")
    c = np.stack([
        (Xk * yk).mean(axis=1)
        for (p, Xk, yk) in nodes
        if p is not None])
    objective = cp.Minimize(cp.sum(cp.multiply(w[1:], c)))

    # saved results here
    fname = f"ab_necessary_lp_gen_{N}_{solver}.pkl"

    if do_opt:

        # caps = [len(span_constraints)] if N < 8 else [111, 112] # becomes infeasible at 112
        caps = [len(span_constraints)]

        for cap in caps:
            problem = cp.Problem(objective, sample_constraints + span_constraints[:cap])
            # problem = cp.Problem(objective, sample_constraints)
            problem.solve(solver=solver, verbose=True)
            if problem.status == 'infeasible':
                print(f'reached infeasibility at cap={int(100*cap/len(span_constraints))}%')
                break
            with open(fname, 'wb') as f: pk.dump((problem.status, w.value, β.value), f)
    
    with open(fname, 'rb') as f: status, w, β = pk.load(f)

    if w is not None: print(w.round(3))
    if β is not None: print(β.round(3))

    print(status)

    # for n, (p, xy) in enumerate(nodes):
    #     if p is None: continue

    #     print(n, p)
    #     print(w[n])
    #     print(xy)
    #     print(w[n:n+1] @ xy)


if __name__ == "__main__": main()

