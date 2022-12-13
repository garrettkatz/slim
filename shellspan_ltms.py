import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import null_space, LinAlgWarning
from adjacent_ltms import adjacency

if __name__ == "__main__":

    N = 4
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # label dichotomies by distance to identity regions
    sX = np.concatenate((X, -X), axis=0)
    dist = (Y != sX[:,np.newaxis,:]).sum(axis=2).min(axis=0)
    udist = np.unique(dist)
    # print(sX)
    # print(np.concatenate((Y, dist[:,np.newaxis], W.round().astype(int)), axis=1))

    A = adjacency(Y, sym=True)

    # For identity dichotomies (Y is a row of X) use identity weights (seem linprog already does this)
    fit = (dist == 0)

    # # Only fit the farthest from identity dichotomies
    # fit = (dist < udist.max())

    # Fit remaining dichotomies, maintaining span constraints with all adjacent previous fits
    unfit = np.flatnonzero(~fit)
    for z,i in enumerate(unfit):
        print(f"far ltm {z} (d={dist[i]}) of {len(unfit)}")

        # dichotomy to fit
        y = Y[i]

        # construct null-spaces for each already-fit neighbor span
        A_null = []
        for j in A[i]:
            # all d==2 neighbors are in d==1, should be already fit
            if not fit[j]: continue # omit neighbors that have not been fit yet

            # column where neighbors differ
            k = (Y[i] != Y[j]).argmax()

            # null space of neighbor w and x, new w should be orth to this null space
            A_null.append( null_space(np.stack((W[j], X[:,k]))) ) # (N, N-2)

        # Big (redundant) null space basis over all neighbors
        if len(A_null) > 0:

            # Fit dichotomy and span constraints simultaneously
            A_null = np.concatenate(A_null, axis=1)
            result = linprog(
                c = (X * y).sum(axis=1),
                A_ub = -(X * y).T,
                b_ub = -np.ones(len(y)),
                A_eq = A_null.T,
                b_eq = np.zeros(A_null.shape[1]),
                bounds = (None, None),
            )

        else:

            # No span constraints yet
            result = linprog(
                c = (X * y).sum(axis=1),
                A_ub = -(X * y).T,
                b_ub = -np.ones(len(y)),
                bounds = (None, None),
            )

        W[i] = result.x
        fit[i] = True
        feasible = (np.sign(W[i] @ X) == y).all()

        if not feasible:
            print(f"failure at ltm {z}")
            break

    if feasible:

        assert (np.sign(W @ X) == Y).all()
        print(W)
        print("it worked??")



