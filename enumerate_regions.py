import sys
import itertools as it
import numpy as np
from multiprocessing import Pool, cpu_count
from check_separability import check_separability_pooled

if __name__ == "__main__":

    do_enum = True
    solver = sys.argv[1]
    N = int(sys.argv[2])
    ε = 1
    canonical = True

    if do_enum:

        # generate half-cube vertices
        X = np.array(tuple(it.product((-1, +1), repeat=N)))
        X = X[:2**(N-1)]
    
        # cumulative sums for for 2-monotonicity
        Xd = np.cumsum(X, axis=1)
    
        # initialize leading dichotomies and redundancy mask
        Y = np.array([[-1]])
        B = np.array([[True]])
    
        # build up candidates one vertex at a time
        for k in range(1, len(X)):
            print(f"vertex {k} of {len(X)}: {Y.shape[0]} candidate dichotomies")

            # all-negative filter
            if (Xd[k] <= 0).all():
                Y = np.block([Y, np.full((Y.shape[0], 1), -1)])
                B = np.block([B, np.full((Y.shape[0], 1), False)])
                continue

            # 2-monotonicity filters
            positive = ((Xd[k] >= Xd[:k]).all(axis=1) & (Y == +1)).any(axis=1)
            negative = ((Xd[k] <= Xd[:k]).all(axis=1) & (Y == -1)).any(axis=1)

            # other half of cube (where w[0] > 0, positive filter will never apply)
            negative |= ((Xd[k] <= -Xd[:k]).all(axis=1) & (-Y == -1)).any(axis=1)

            # Expand Y and B, branching on indeterminate signs
            unknowns = ~ (negative | positive)    
            Y = np.block([
                [Y[negative], np.full((negative.sum(),1), -1)],
                [Y[unknowns], np.full((unknowns.sum(),1), -1)],
                [Y[unknowns], np.full((unknowns.sum(),1), +1)],
                [Y[positive], np.full((positive.sum(),1), +1)],
            ])

            # entries with determinate signs are redundant
            B = np.block([
                [B[negative], np.full((negative.sum(), 1), False)],
                [B[unknowns], np.full((unknowns.sum(), 1), True)],
                [B[unknowns], np.full((unknowns.sum(), 1), True)],
                [B[positive], np.full((positive.sum(), 1), False)],
            ])

        print(f"all {len(X)} vertices: {Y.shape[0]} candidate dichotomies")

        pool_args = [
            (X[b], y[b], ε, canonical, solver, f"{i} of {len(Y)}")
            for i,(y,b) in enumerate(zip(Y, B))]

        # # singleprocessing version
        # results = list(map(check_separability_pooled, pool_args))

        # multiprocessing version (don't use all cores)
        num_procs = max(1, cpu_count()-2)
        with Pool(num_procs) as pool:
            results = pool.map(check_separability_pooled, pool_args)

        feasible, W = map(np.array, zip(*results))
        Y = Y[feasible]
        B = B[feasible]
        W = W[feasible]

        np.savez(f"regions_{N}_{solver}.npz", X=X, Y=Y, B=B, W=W)

    with np.load(f"regions_{N}_{solver}.npz") as regions:
        X, Y, B, W = (regions[key] for key in ("XYBW"))

    assert (np.sign(W @ X.T) == Y).all()
    assert len(Y) == len(np.unique(Y, axis=0))
    nB = B.sum(axis=1)
    print(f"{len(Y)} feasible regions total")
    print(f"{nB.min()} <= ~{nB.mean()} <= {nB.max()} constraints per region")


