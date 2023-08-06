import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import null_space, LinAlgWarning
from adjacent_ltms import adjacency

if __name__ == "__main__":

    N = 4
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    A = adjacency(Y)

    # For initial dichotomy (all -1) use symmetric weights
    W[0] = -np.eye(N)[0]

    # Fit remaining dichotomies one after another, maintaining span constraints with all adjacent previous fits
    for i in range(1,len(A)):
        print(f"ltm {i} of {len(A)}")

        # construct null-spaces for each already-fit neighbor span
        A_null = []
        for j in A[i]:
            if j > i: break # omit neighbors that have not been fit yet

            # column where neighbors differ
            k = (Y[i] != Y[j]).argmax()

            # null space of neighbor w and x, new w should be orth to this null space
            A_null.append( null_space(np.stack((W[j], X[:,k]))) ) # (N, N-2)

        # Big (redundant) null space basis over all neighbors
        A_null = np.concatenate(A_null, axis=1)

        # Fit dichotomy and span constraints simultaneously
        y = Y[i]
        result = linprog(
            c = (X * y).sum(axis=1),
            A_ub = -(X * y).T,
            b_ub = -np.ones(len(y)),
            A_eq = A_null.T,
            b_eq = np.zeros(A_null.shape[1]),
            bounds = (None, None),
        )
        W[i] = result.x
        feasible = (np.sign(W[i] @ X) == y).all()

        if not feasible:
            print(f"failure at ltm {i}")
            break



