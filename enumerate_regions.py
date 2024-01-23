import sys
import itertools as it
import numpy as np
import cvxpy as cp
from multiprocessing import Pool, cpu_count

def check_feasibility(X, y, ε):
    N = X.shape[0]
    w = cp.Variable(N)
    wXy = w @ (X * y)
    objective = cp.Minimize(cp.sum(wXy))
    constraints = [
        wXy >= ε,            # linearly separable
        w[-1] >= 0,           # non-negative
    ] + [
        w[n] >= w[n+1]       # descending order
        for n in range(N-1)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='ECOS')
    feasible = (problem.status == 'optimal')
    return feasible, w.value

def check_feasibility_pooled(args):
    return check_feasibility(*args)

if __name__ == "__main__":

    N = int(sys.argv[1])
    ε = 1

    # generate half-cube vertices
    X = np.array(tuple(it.product((-1, +1), repeat=N-1))).T
    X = np.vstack((-np.ones(2**(N-1), dtype=int), X))

    # Xd = X # 1-monotonicity
    Xd = np.cumsum(X, axis=0) # 2-monotonicity

    # initialize leading dichotomies
    Y = np.array([[-1]])
    B = np.array([[True]])

    for k in range(1, X.shape[1]):
        print(f"vertex {k} of {X.shape[1]}: {Y.shape[0]} candidate dichotomies")

        must_be_negative = ((Xd[:,k:k+1] <= Xd[:,:k]).all(axis=0) & (Y < 0)).any(axis=1)
        must_be_positive = ((Xd[:,k:k+1] >= Xd[:,:k]).all(axis=0) & (Y > 0)).any(axis=1)
        can_be_either_or = ~ (must_be_negative | must_be_positive)

        Y = np.block([
            [Y[must_be_negative], np.full((must_be_negative.sum(),1), -1)],
            [Y[can_be_either_or], np.full((can_be_either_or.sum(),1), -1)],
            [Y[must_be_positive], np.full((must_be_positive.sum(),1), +1)],
            [Y[can_be_either_or], np.full((can_be_either_or.sum(),1), +1)],
        ])
        B = np.block([
            [B[must_be_negative], np.full((must_be_negative.sum(), 1), False)],
            [B[can_be_either_or], np.full((can_be_either_or.sum(), 1), True)],
            [B[must_be_positive], np.full((must_be_positive.sum(), 1), False)],
            [B[can_be_either_or], np.full((can_be_either_or.sum(), 1), True)],
        ])

    # W = []
    # for i,(y,b) in enumerate(zip(Y, B)):
    #     print(f"checking candidate {i} of {Y.shape[0]}")
    #     feasible, w = check_feasibility(X[:,b], y[b], ε)
    #     if feasible: W.append(w)
    # W = np.stack(W)

    # don't use all cores when multiprocessing
    num_procs = cpu_count()-2
    pool_args = [(X[:,b], y[b], ε) for (y,b) in zip(Y, B)]
    with Pool(num_procs) as pool:
        results = pool.map(check_feasibility_pooled, pool_args)
    W = np.stack([w for (feasible, w) in results if feasible])

    print(f"{W.shape[0]} feasible regions total")
    


