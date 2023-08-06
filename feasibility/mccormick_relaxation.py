import numpy as np
from numpy.linalg import norm
from adjacent_ltms import adjacency
import itertools as it
import matplotlib.pyplot as pt
import matplotlib as mp
import scipy.sparse as sp
import cvxpy as cp

# np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'

N = 4
ltms = np.load(f"ltms_{N}.npz")
Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
A, K = adjacency(Y, sym=True)

# full dichotomy for formulation
X = np.concatenate((X, -X), axis=1)
Y = np.concatenate((Y, -Y), axis=1)

# dims
M, N = W.shape
V = X.shape[1]

# minimum slack
eps = .5 # linprog used 1, but small numerical errors
R = np.linalg.norm(W, axis=1).max() # upper bound on |w[i]|
e2R = eps**2 / R
e2N = eps * np.ones(2**N) / 2**N

# save initial feasible point
W0 = W

# decision variables
# M = 8
ws = {i: cp.Variable(N) for i in range(M)}
# qs = {(i,k): cp.Variable(1) for i in range(M) for k in K[i]}
qs = {i: cp.Variable(len(K[i])) for i in range(M)} # more scalable for cvxpy

# region constraints
regions = [ws[i] @ (X * Y[i]) >= eps for i in range(M)]

# w norm constraints
w_norms = [cp.atoms.norm(ws[i], p=2) <= R for i in range(M)]

# span loss relaxation and wP norm constraints
loss = 0
wp_norms = []
wi_coefs = []
for i in range(M):
    Pi = np.zeros((N,N))
    for a, (j, k) in enumerate(zip(A[i], K[i])):
        if j >= M: continue

        # precompute recurring terms
        wi, wj, xk = ws[i], ws[j], X[:,k]
        Pk = np.eye(N) - xk.reshape((N, 1)) * xk / N
        wiPk = wi @ Pk
        wjPk = wj @ Pk

        # wP norm constraints
        wp_norms.append(qs[i][a] >= cp.atoms.norm(wiPk, p=2))

        # boundary projection sum
        Pi += Pk

    wi_coefs.append(Pi @ (X * Y[i]) @ e2N)

# summed mccormick envelopes
# U = sum(qs.values()) * e2R - len(qs) * e2R**2
U = sum([cp.sum(q) for q in qs.values()]) * e2R - len(qs) * e2R**2
D = cp.hstack([ws[i] for i in range(M)]) @ np.concatenate(wi_coefs) - len(qs) * eps**2 * (1 - 1 / 2**(N-1))
loss = U - D
# loss = U
# loss = -D

constraints = regions + w_norms + wp_norms
objective = cp.Minimize(loss)
prob = cp.Problem(objective, constraints)

prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)

print(ws[0].value)
print(f"{U.value} - {D.value} = {U.value - D.value}")

from span_loss_derivatives import calc_derivatives

W = np.vstack([ws[i].value for i in range(M)])
print(W.shape)

loss, grad, hess = calc_derivatives(Y, W, X, A, K)
print(loss)
print(np.fabs(grad).max())


