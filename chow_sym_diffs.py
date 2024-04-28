import itertools as it
import numpy as np
import cvxpy as cp

N = 5
solver = 'GUROBI'

with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, B, W = (regions[key] for key in ("XYBW"))

B = np.load(f"boundaries_{N}_{solver}.npy")

# extract canonical adjacencies
A = set()
for i,j in it.combinations(range(len(Y)), r=2):
    k = np.flatnonzero(Y[i] != Y[j])
    if len(k) != 1: continue
    k = k[0]
    
    if Y[j,k] > 0:
        A.add((i,j,k))
    else:
        A.add((j,i,k))

# shifted/scaled chow parameters
c = Y @ X

# solve the regions with chow symmetries
for i, y in enumerate(Y):

    # linear program with symmetries
    u = cp.Variable(N)

    data_constraints = [u @ (X[B[i]].T * y[B[i]]) >= 1]
    
    canonical_constraints = [u[-1] >= 0, u[:-1] >= u[1:]]
    
    n = np.flatnonzero(c[i,:-1] == c[i,1:])
    chow_constraints = [u[n] == u[n+1]]
    if c[i,-1] == 0: chow_constraints.append(u[-1] == 0)
    
    constraints = data_constraints + canonical_constraints + chow_constraints
    objective = cp.Minimize(cp.sum(cp.multiply(u,c[i])))
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    assert problem.status == "optimal"

# W = W * c[:,[0]] / W[:,[0]]
# W = W * 2**(N-1) / W[:,[0]]

print("C")
print(c)
print("W")
print(W)

print("i,j, W[i], W[j], X[k], W[j] - W[i]")
for (i,j,k) in sorted(A):
    print(i,j, W[i], W[j], X[k], (W[j] - W[i]).round(1))




