import itertools as it
import numpy as np
import cvxpy as cp

def chow_check(X, y):

    # shifted scaled chow version, still encodes zeros and order
    chow = y @ X

    # chow symmetry constraint indices
    chow_zero = np.flatnonzero(chow == 0)
    chow_equal = np.flatnonzero(chow[:-1] == chow[1:])

    # check that you can solve with chow symmetry constraints
    w = cp.Variable(N)
    objective = cp.Minimize(cp.sum(w))

    data_constraints = [(y[:,None] * X) @ w >= 1]
    canonical_constraints = [w[-1] >= 0, w[:-1] >= w[1:]]
    
    chow_constraints = []
    if len(chow_zero) > 0: chow_constraints.append(w[chow_zero] == 0)
    if len(chow_equal) > 0: chow_constraints.append(w[chow_equal] == w[chow_equal+1])

    constraints = data_constraints + canonical_constraints + chow_constraints
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    # pack results
    feasible = (problem.status == 'optimal')
    w = w.value if feasible else np.empty(N)

    return w, chow, feasible

N = 5
solver = 'GUROBI'

with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, B, W = (regions[key] for key in ("XYBW"))

Wc = []
diff_idx = []
for i,y in enumerate(Y):

    w, chow, feasible = chow_check(X, y)

    cy = np.sign(X @ chow)

    assert feasible
    print(i, chow, W[i], w, "N" if (cy != y).any() else "")
    Wc.append(w)
    if not np.allclose(W[i], w): diff_idx.append(i)
    
    if (cy != y).any():
        for k in range(len(X)):
            print(" ", cy[k], y[k], X[k])
        input('..')

print("\n***diff***\n")
for i in diff_idx:
    print(i, chow, W[i], w)
    
# check on muroga case
w = np.array([13, 7, 6, 6, 4, 4, 4, 3, 2])
N = len(w)
X = np.array(tuple(it.product((-1, +1), repeat=N)))[:2**(N-1)]

y = np.sign(X @ w)

wc, chow, feasible = chow_check(X, y)
assert feasible
print('muroga', chow, w, wc)


