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
    subsize = 3000

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
    # R = Yc.shape[0]

    # set up boundary indices to remove redundant region constraints
    B = np.zeros(Yc.shape, dtype=bool)
    for i in Yn: B[i, :] = (Yc[i] != Yn[i]).any(axis=0)

    # set up permutation of vertices
    perm = np.arange(X.shape[1]) # identity permutation
    # perm = np.random.permutation(X.shape[1]) # random
    B = B[:, perm]
    Y = Yc[:, perm]

    # get random subset of canonicals
    subsize = min(subsize, Yc.shape[0])
    print(f"sampling {subsize} of {Yc.shape[0]} regions")
    subset = np.random.choice(Yc.shape[0], size=subsize, replace=False)
    B = B[subset]
    Y = Y[subset]

    print(f"{B.any(axis=0).sum()} of {Yc.shape[1]} unioned boundary vertices")
    print(f"{B.sum(axis=1).mean()} per region on average")
    print(f"< {B.sum()} edges total")
    # input('.')
    # import matplotlib.pyplot as pt
    # pt.imshow(B)
    # pt.show()

    # build binary label tree
    print("Building tree...")
    nodes = [(None, (), -1, np.empty((N,0)), np.empty(0))] # parent index, kk, i, x data, y data
    lookup = {(): 0} # maps (...,k,...)+(...,yk,...) path to node index

    for (i,k) in zip(*np.nonzero(B)):

        # examples so far
        kk = np.flatnonzero(B[i, :k+1])
        Xk = X[:, kk]
        yk = Y[i, kk]

        # key for current node
        key = tuple(kk) + tuple(yk)

        # skip if node already exists
        if key in lookup: continue

        # otherwise, save new node index
        n = len(nodes)
        lookup[key] = n

        # get parent index
        key = tuple(kk[:-1])+tuple(yk[:-1])
        p = lookup[key]

        # and save the new node
        nodes.append((p, kk, i, Xk, yk))

    # for n, (p, kk, i, x, y) in enumerate(nodes): print(n, p, kk, i, x.shape, y)

    # tree edge data
    At = [
        (n, p, x[:,-1], y[-1])
        for n, (p, kk, i, x, y) in enumerate(nodes)
        if p is not None]

    print(f"{len(nodes)} nodes, {len(At)} edges")
    # input('.')

    # ### per-node weight variable version
    # ### finally finds infeasible at subsize == 2000! but slow.
    # ### saved in ab_necessary_lp_gen_8_CBC.infeas.pkl
    
    # ## variables
    # w = cp.Variable((len(nodes), N)) # weight vector per node
    # β = cp.Variable(len(At)) # beta per spanning tree edge (nodes - 1)

    # ## data constraints
    # print("Building sample constraints...")
    # sample_constraints = [
    #     w[n:n+1] @ (Xk * yk) >= eps
    #     for n, (p, kk, i, Xk, yk) in enumerate(nodes)
    #     if p is not None] # no constraints on root

    # ## span constraints
    # print("Building span constraints...")
    # span_constraints = [
    #     w[n] == w[p] + β[e] * (x * y)
    #     for e, (n,p,x,y) in enumerate(At)]

    # ## objective to bound problem
    # print("Building objective...")
    # c = np.stack([
    #     (Xk * yk).mean(axis=1)
    #     for (p, kk, i, Xk, yk) in nodes
    #     if p is not None])
    # objective = cp.Minimize(cp.sum(cp.multiply(w[1:], c)))

    # constraints = sample_constraints + span_constraints

    ### root weight variable version

    ## variables
    w = cp.Variable(N) # weight vector at root
    β = cp.Variable(len(At)) # beta per spanning tree edge (nodes - 1)

    # path index lookup
    path_index = {}
    for e, (n, p, _, _) in enumerate(At): path_index[n,p] = e

    ## data constraints
    print("Building constraints and objective...")
    constraints = []
    obj_w, obj_β = np.zeros(N), np.zeros(len(At))
    for n, (p, kk, i, Xk, yk) in enumerate(nodes):

        # no region constraint on initial weights
        if p is None: continue

        # get path edge index
        e = []
        a = n
        while p is not None:
            e.insert(0, path_index[a,p])
            # move to parent
            a = p
            p = nodes[a][0]

        # add sample constraint
        Xyk = Xk * yk
        Xyk_T_Xyk = Xyk.T @ Xyk
        constraints.append( w @ Xyk + β[e] @ Xyk_T_Xyk >= eps )

        # accumulate objective data for constraint slacks
        obj_w += Xyk.sum(axis=1)
        obj_β[e] += Xyk_T_Xyk.sum(axis=1)

    ## objective to bound problem: minimize net slack
    objective = cp.Minimize(w @ obj_w + β @ obj_β)

    # saved results here
    fname = f"ab_necessary_lp_gen_{N}_{solver}.pkl"

    if do_opt:

        input("Press Enter to do opt")
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver, verbose=True)
        with open(fname, 'wb') as f: pk.dump((subset, problem.status, w.value, β.value), f)
    
    with open(fname, 'rb') as f: subset, status, w, β = pk.load(f)

    if w is not None: print(w.round(3))
    if β is not None: print(β.round(3))

    print(f"region subset = {subset}")
    print(status)

    # for n, (p, xy) in enumerate(nodes):
    #     if p is None: continue

    #     print(n, p)
    #     print(w[n])
    #     print(xy)
    #     print(w[n:n+1] @ xy)


if __name__ == "__main__": main()

