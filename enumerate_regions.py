import sys
import itertools as it
import numpy as np
import cvxpy as cp
from multiprocessing import Pool, cpu_count

def check_feasibility(X, y, ε, verbose=False):
    N = X.shape[0]
    w = cp.Variable(N)
    wXy = w @ (X * y)
    objective = cp.Minimize(cp.sum(wXy))
    constraints = [
        wXy >= ε,       # linearly separable
        w[-1] >= 0,     # non-negative
        w[:-1] >= w[1:] # descending order
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='CBC', verbose=verbose) # ECOS has a single false positive (reports feasible) at N=8
    feasible = (problem.status == 'optimal')
    w = w.value if feasible else np.empty(N)
    return feasible, w

def check_feasibility_pooled(args):
    return check_feasibility(*args)

if __name__ == "__main__":

    do_enum = True
    N = int(sys.argv[1])
    ε = 1

    if do_enum:

        # generate half-cube vertices
        X = np.array(tuple(it.product((-1, +1), repeat=N-1))).T
        X = np.vstack((-np.ones(2**(N-1), dtype=int), X))
        # X = np.vstack((X, -np.ones(2**(N-1), dtype=int)))
        print(X)
    
        # for 2-monotonicity
        Xd = np.cumsum(X, axis=0)
    
        # initialize leading dichotomies
        Y = np.array([[-1]])
        B = np.array([[True]])
    
        for k in range(1, X.shape[1]):
            print(f"vertex {k} of {X.shape[1]}: {Y.shape[0]} candidate dichotomies")
            print(X[:,k])
    
            # duplicate Y
            Y = np.block([
                [Y, -np.ones((Y.shape[0], 1))],
                [Y, +np.ones((Y.shape[0], 1))]])
    
            # 2-monotonicity filter
            infeasible = ((Xd[:,k:k+1] >= Xd[:,:k]).all(axis=0) & (Y[:,-1:] < Y[:,:-1])).any(axis=1)
            infeasible |= ((Xd[:,k:k+1] <= Xd[:,:k]).all(axis=0) & (Y[:,-1:] > Y[:,:-1])).any(axis=1)
            Y = Y[~infeasible, :]

            # trailing negation filter
            if k > 0:
                n = (X[:,k] > 0).argmax()

                # kn = (X[:n+1,:k+1] < 0).all(axis=0) & (X[n:,:k+1] <= -X[n:,k:k+1]).all(axis=0)
                # print(kn.astype(int))
                # infeasible = (Y[:,kn] > 0).any(axis=1) & (Y[:, k] > 0)

                kn = (X[n:,:k] == -X[n:,k:k+1]).all(axis=0).argmax()
                infeasible = (Y[:,kn] > 0) & (Y[:,k] > 0)

                Y = Y[~infeasible, :]
    
            # must_be_negative = ((Xd[:,k:k+1] <= Xd[:,:k]).all(axis=0) & (Y < 0)).any(axis=1)
            # must_be_positive = ((Xd[:,k:k+1] >= Xd[:,:k]).all(axis=0) & (Y > 0)).any(axis=1)
            # can_be_either_or = ~ (must_be_negative | must_be_positive)
    
            # Y = np.block([
            #     [Y[must_be_negative], np.full((must_be_negative.sum(),1), -1)],
            #     [Y[can_be_either_or], np.full((can_be_either_or.sum(),1), -1)],
            #     [Y[can_be_either_or], np.full((can_be_either_or.sum(),1), +1)],
            #     [Y[must_be_positive], np.full((must_be_positive.sum(),1), +1)],
            # ])
            # B = np.block([
            #     [B[must_be_negative], np.full((must_be_negative.sum(), 1), False)],
            #     [B[can_be_either_or], np.full((can_be_either_or.sum(), 1), True)],
            #     [B[can_be_either_or], np.full((can_be_either_or.sum(), 1), True)],
            #     [B[must_be_positive], np.full((must_be_positive.sum(), 1), False)],
            # ])
    
        # W = []
        # for i,(y,b) in enumerate(zip(Y, B)):
        #     print(f"checking candidate {i} of {Y.shape[0]}")
        #     feasible, w = check_feasibility(X[:,b], y[b], ε)
        #     if feasible: W.append(w)
        # W = np.stack(W)
    
        # don't use all cores when multiprocessing
        num_procs = cpu_count()-2
        # pool_args = [(X[:,b], y[b], ε) for (y,b) in zip(Y, B)]
        pool_args = [(X, y, ε) for y in Y]
        with Pool(num_procs) as pool:
            results = pool.map(check_feasibility_pooled, pool_args)
        feasible, W = map(np.array, zip(*results))
        W = W[feasible]
        Y = Y[feasible]

        np.savez(f"regions_{N}.npz", X=X, Y=Y, W=W)

    with np.load(f"regions_{N}.npz") as regions:
        X, Y, W = (regions[key] for key in ("XYW"))

    if not (np.sign(W @ X) == Y).all():
        bad = (np.sign(W @ X) != Y).any(axis=1).argmax()
        print(bad)
        print(W[bad])
        print(Y[bad])
        print(np.sign(W[bad] @ X))

        feas, w = check_feasibility(X, Y[bad], ε, verbose=True)
        print(feas)
        print(w)

    assert (np.sign(W @ X) == Y).all()
    assert len(Y) == len(np.unique(Y, axis=0))
    print(f"{W.shape[0]} feasible regions total")


