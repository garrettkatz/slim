import sys
import itertools as it
import numpy as np
import cvxpy as cp
from multiprocessing import Pool, cpu_count

def check_feasibility(X, y, ε, tag=None, verbose=False):
    N = X.shape[1]
    w = cp.Variable(N)
    wxy = w @ (X.T * y)
    objective = cp.Minimize(cp.sum(wxy))
    constraints = [
        wxy >= ε,       # linearly separable
        w[-1] >= 0,     # non-negative
        w[:-1] >= w[1:] # descending order
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='CBC', verbose=verbose) # ECOS has a single false positive (reports feasible) at N=8
    feasible = (problem.status == 'optimal')
    w = w.value if feasible else np.empty(N)
    if tag is not None:
        print(f"{tag} complete: feasible={feasible}")
    return feasible, w

def check_feasibility_pooled(args):
    return check_feasibility(*args)

if __name__ == "__main__":

    do_enum = True
    N = int(sys.argv[1])
    ε = 1

    if do_enum:

        # generate half-cube vertices
        X = np.array(tuple(it.product((-1, +1), repeat=N)))
        X = X[:2**(N-1)]
        print(X)
    
        # for 2-monotonicity
        Xd = np.cumsum(X, axis=1)
    
        # initialize leading dichotomies and redundancy mask
        Y = np.array([[-1]])
        B = np.array([[True]])
    
        for k in range(1, len(X)):
            print(f"vertex {k} of {len(X)}: {Y.shape[0]} candidate dichotomies")
            print(X[k])

            # all-negative filter
            if (Xd[k] <= 0).all():
                Y = np.block([Y, np.full((Y.shape[0], 1), -1)])
                B = np.block([B, np.full((Y.shape[0], 1), False)])
                continue

            # 2-monotonicity filters
            negative = ((Xd[k] <= Xd[:k]).all(axis=1) & (Y == -1)).any(axis=1)
            positive = ((Xd[k] >= Xd[:k]).all(axis=1) & (Y == +1)).any(axis=1)

            # other half of cube
            negative |= ((Xd[k] <= -Xd[:k]).all(axis=1) & (-Y == -1)).any(axis=1)

            # Expand Y and B
            unknowns = ~ (negative | positive)
    
            Y = np.block([
                [Y[negative], np.full((negative.sum(),1), -1)],
                [Y[unknowns], np.full((unknowns.sum(),1), -1)],
                [Y[unknowns], np.full((unknowns.sum(),1), +1)],
                [Y[positive], np.full((positive.sum(),1), +1)],
            ])
            B = np.block([
                [B[negative], np.full((negative.sum(), 1), False)],
                [B[unknowns], np.full((unknowns.sum(), 1), True)],
                [B[unknowns], np.full((unknowns.sum(), 1), True)],
                [B[positive], np.full((positive.sum(), 1), False)],
            ])

        print(f"all {len(X)} vertices: {Y.shape[0]} candidate dichotomies")
   
        # don't use all cores when multiprocessing
        num_procs = cpu_count()-3
        pool_args = [(X[b], y[b], ε, f"{i} of {len(Y)}") for i,(y,b) in enumerate(zip(Y, B))]
        with Pool(num_procs) as pool:
            results = pool.map(check_feasibility_pooled, pool_args)
        feasible, W = map(np.array, zip(*results))
        Y = Y[feasible]
        B = B[feasible]
        W = W[feasible]

        np.savez(f"regions_{N}.npz", X=X, Y=Y, B=B, W=W)

    with np.load(f"regions_{N}.npz") as regions:
        X, Y, B, W = (regions[key] for key in ("XYBW"))

    assert (np.sign(W @ X.T) == Y).all()
    assert len(Y) == len(np.unique(Y, axis=0))
    print(f"{len(Y)} feasible regions total")
    print(f"{B.sum(axis=1).mean()} constraints per region")


