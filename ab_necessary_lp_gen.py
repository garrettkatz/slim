import itertools as it
from time import perf_counter
import pickle as pk
import sys
from collections import deque
import numpy as np
import cvxpy as cp
import load_ltm_data as ld

# @profile
def do_lp(eps, N, num_regions, shuffle, solver, verbose):

    # load canonical regions and adjacencies
    Y, _, X, _ = ld.load_ltm_data(N)
    with open(f"adjs_{N}_c.npz", "rb") as f: (Yn, _) = pk.load(f)

    # number of regions
    R = Y.shape[0]

    # set up boundary indices to remove redundant region constraints
    B = np.zeros(Y.shape, dtype=bool)
    for i in Yn: B[i, :] = (Y[i] != Yn[i]).any(axis=0)

    # get random subset of canonicals
    if num_regions < Y.shape[0]:
        print(f"sampling {num_regions} of {Y.shape[0]} regions")
        subset = np.random.choice(Y.shape[0], size=num_regions, replace=False)
        B = B[subset]
        Y = Y[subset]
    else:
        subset = None

    # shuffle presentation order if requested
    if shuffle:
        perm = np.random.permutation(X.shape[1])
        X = X[:, perm]
        B = B[:, perm]
        Y = Y[:, perm]

    print(f"{B.sum(axis=1).mean()} boundaries per region on average")
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

    ### per-node weight variable version
    ### finally finds infeasible at subsize == 2000! but slow.
    ### saved in ab_necessary_lp_gen_8_CBC.infeas.pkl
    
    ## variables
    w = cp.Variable((len(nodes), N)) # weight vector per node
    β = cp.Variable(len(At)) # beta per spanning tree edge (nodes - 1)

    ## data constraints
    print("Building sample constraints...")
    sample_constraints = [
        w[n:n+1] @ (Xk * yk) >= eps
        for n, (p, kk, i, Xk, yk) in enumerate(nodes)
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
        for (p, kk, i, Xk, yk) in nodes
        if p is not None])
    objective = cp.Minimize(cp.sum(cp.multiply(w[1:], c)))

    constraints = sample_constraints + span_constraints

    # ### root weight variable version

    # ## variables
    # w = cp.Variable(N) # weight vector at root
    # β = cp.Variable(len(At)) # beta per spanning tree edge (nodes - 1)

    # # path index lookup
    # path_index = {}
    # for e, (n, p, _, _) in enumerate(At): path_index[n,p] = e

    # ## data constraints
    # print("Building constraints and objective...")
    # constraints = []
    # obj_w, obj_β = np.zeros(N), np.zeros(len(At))
    # for n, (p, kk, i, Xk, yk) in enumerate(nodes):

    #     # no region constraint on initial weights
    #     if p is None: continue

    #     # get path edge index
    #     e = []
    #     a = n
    #     while p is not None:
    #         e.insert(0, path_index[a,p])
    #         # move to parent
    #         a = p
    #         p = nodes[a][0]

    #     # add sample constraint
    #     Xyk = Xk * yk
    #     Xyk_T_Xyk = Xyk.T @ Xyk
    #     constraints.append( w @ Xyk + β[e] @ Xyk_T_Xyk >= eps )

    #     # accumulate objective data for constraint slacks
    #     obj_w += Xyk.sum(axis=1)
    #     obj_β[e] += Xyk_T_Xyk.sum(axis=1)

    # ## objective to bound problem: minimize net slack
    # objective = cp.Minimize(w @ obj_w + β @ obj_β)

    # input("Press Enter to do opt")
    start_time = perf_counter()
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=True)
    opt_time = perf_counter() - start_time

    result = (
        problem.status,
        w.value,
        β.value,
        subset,
        len(nodes),
        opt_time,
    )
    return result

if __name__ == "__main__":

    do_opt = True

    # input dimension for optimization
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 8

    # solver = 'GLPK' # 106s
    # solver = 'CBC' # 13s
    # solver = 'GLOP' # 11s
    # solver = 'SCIPY' # 14s
    solver = 'ECOS' # 6s, ~15min

    eps = 1
    num_regions = 3000
    shuffle = False
    verbose = True

    # saved results here
    fname = f"ab_necessary_lp_gen_{N}_{num_regions}_{solver}.pkl"

    if do_opt:
        result = do_lp(eps, N, num_regions, shuffle, solver, verbose)
        with open(fname, 'wb') as f: pk.dump(result, f)
    
    with open(fname, 'rb') as f: result = pk.load(f)

    status, w, β, subset, num_nodes, opt_time, = result

    if w is not None: print(w.round(3))
    if β is not None: print(β.round(3))

    print(f"region subset = {subset}")
    print(f"optimization time = {opt_time}")
    print(f"solver status = {status}")


