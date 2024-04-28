import itertools as it
import numpy as np
import cvxpy as cp

N = 4
solver = 'GUROBI'

with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, B, W = (regions[key] for key in ("XYBW"))

B = np.load(f"boundaries_{N}_{solver}.npy")

# extract canonical adjacencies
A = set()
for i,j in it.combinations(range(len(Y)), r=2):
    k = np.flatnonzero(Y[i] != Y[j])
    if len(k) == 1: A.add((i,j,k[0]))

# shifted/scaled chow parameters
c = Y @ X

# linear program for span rule
u = cp.Variable(W.shape)
g = cp.Variable(len(A))

span_constraints = [
    u[i] == (u[j] + g[a] * X[k])
    for a, (i, j, k) in enumerate(A)]

data_constraints = [
    u[i] @ (X[B[i]].T * y[B[i]]) >= 1
    for i, y in enumerate(Y)]

canonical_constraints = [u[:,-1] >= 0, u[:,:-1] >= u[:,1:]]

chow_equal = [np.flatnonzero(c[i,:-1] == c[i,1:]) for i in range(len(c))]
chow_constraints = [
    u[i,n] == u[i,n+1]
    for i, n in enumerate(chow_equal)] \
    # + [\
    # u[i,-1] == 0
    # for i in range(len(c)) if c[i,-1] == 0]

# constraints = span_constraints + data_constraints + canonical_constraints + chow_constraints
constraints = data_constraints + canonical_constraints + chow_constraints
objective = cp.Minimize(cp.sum(cp.multiply(u,c)))

problem = cp.Problem(objective, constraints)
problem.solve(solver=solver, verbose=False)

print(problem.status)
print('A:')
for (i,j,k) in sorted(A): print(i,j,k, X[k])
print('u*:')
print(u.value * c[:,[0]] / u[:,[0]].value)
print('chow:')
print(c)
print('g:')
print(g.value)
print("chow fit:")
print(((c @ X.T) * Y >= 1).all(axis=1))
print((c @ X.T) * Y)

print(X.T)
print(Y)


