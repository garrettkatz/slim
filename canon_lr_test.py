import numpy as np
import cvxpy as cp

solver = "GUROBI"
N = 7

# load region data
with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

print(Y.shape, B.shape)

for i in range(len(Y)):

    # check canonical boundaries
    K = []
    for b in np.flatnonzero(B[i]):
        y = Y[i].copy()
        y[b] = -y[b]
        if (y == Y).all(axis=1).any(): K.append(b)

    print(i, K)

    # See if you can realize y with wx = 1 at each canonical boundary x
    y = Y[i]

    # weight variable
    w = cp.Variable(N)

    # dots with each (signed) training sample
    wxy = w @ (X.T * y)

    constraints = []

    # linear separability constraint
    constraints += [wxy >= 1]

    # canonical boundary constraints
    constraints += [wxy[k] == 1 for k in K]

    # canonical weight constraints
    constraints += [
        w[-1] >= 0,      # non-negative
        w[:-1] >= w[1:]] # descending order

    # minimize slack to make problem bounded
    objective = cp.Minimize(cp.sum(wxy))

    # solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    feasible = (problem.status == 'optimal')
    if not feasible:
        print(f"Failed at i={i}")

        print(X[K,:].T * y[K])

        break

if feasible:
    print("All worked!")


