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

    do_opt = False
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

    # compute adjacency graph spanning tree
    i0 = 0 # root region
    nodes = []
    queue = deque([(i0, None, np.empty((N,0)))]) # (node region, parent node index, examples on path to it)
    explored = set() # regions that have been popped
    while len(queue) > 0:
        # pop queued nodes in BFS order
        i, p, xy = queue.popleft()

        # save new node for new path to this region
        nodes.append( (p, xy) )
        n = len(nodes) - 1 # index of new node

        # but don't expand children more than once
        if i in explored: continue

        # mark new regions as explored and expand children
        explored.add(i)
        for (j,k) in A[i]:

            # don't include children that repeat/undo previous examples
            if (xy == +X[:,k:k+1]).all(axis=0).any(): continue
            if (xy == -X[:,k:k+1]).all(axis=0).any(): continue

            # queue nodes for new examples
            xy_new = np.concatenate((xy, X[:,k:k+1]*Yc[j,k]), axis=1)
            queue.append((j, n, xy_new))

    # All nodes in spanning tree
    print(f"|Ac|={len(Ac)}, |tree|={len(nodes)}, R = {R}")
    # print(At)

    for n, (p, xy) in enumerate(nodes): print(n, p, xy.shape)
    input('.')

    ## variables
    w = cp.Variable((len(nodes), N)) # weight vector per spanning tree node
    β = cp.Variable(len(nodes)) # beta per spanning tree edge (0 at root)

    ## sample constraints
    sample_constraints = [
        w[n:n+1] @ xy >= eps
        for n, (p, xy) in enumerate(nodes) if p is not None]

    ## span constraints
    span_constraints = [
        w[n] == w[p] + β[n] * xy[:,-1]
        for n, (p, xy) in enumerate(nodes) if p is not None]

    ## objective to bound problem
    c = np.zeros((len(nodes), N))
    for n, (p, xy) in enumerate(nodes):
        if p is not None: c[n] = xy.mean(axis=1)
    objective = cp.Minimize(cp.sum(cp.multiply(w, c)))

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

