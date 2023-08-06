import numpy as np
from numpy.linalg import norm
from adjacent_ltms import adjacency
import itertools as it
import matplotlib.pyplot as pt
import matplotlib as mp
import scipy.sparse as sp
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

# from span_loss_derivatives import calc_derivatives
from sq_span_loss_derivatives import calc_derivatives
import cvxpy as cp

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

# np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'

N = 4
ltms = np.load(f"ltms_{N}.npz")
Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
# A, K = adjacency(Y, sym=False) # sym=False just halves the objective function
A, K = adjacency(Y, sym=True) # for "other way" of hand-calced gradient

# dims
M, N = W.shape
V = X.shape[1]

# minimum slack
eps = .5 # linprog used 1, but small numerical errors

# save initial feasible point
W0 = W

# variables for each weight
M = 8
ws = [cp.Variable(N) for m in range(M)]

# region constraints
constraints = [ws[i] @ (X * Y[i]) >= eps for i in range(M)]

# span loss objective
loss = 0
for i in range(M):
    for j, k in zip(A[i], K[i]):
        if j >= M: continue

        # precompute recurring terms
        wi, wj, xk = ws[i], ws[j], X[:,k]
        Pk = np.eye(N) - xk.reshape((N, 1)) * xk / N
        wiPk = wi @ Pk
        wjPk = wj @ Pk

        # accumulate span loss
        loss += (wiPk @ wiPk) * (wjPk @ wjPk) - (wiPk @ wjPk)**2

objective = cp.Minimize(loss)

prob = cp.Problem(objective, constraints)

prob.solve()  # Returns the optimal value.
# print("status:", prob.status)
# print("optimal value", prob.value)
# print("optimal var", x.value, y.value)




