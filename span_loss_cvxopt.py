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

from span_loss_derivatives import calc_derivatives
from cvxopt import matrix, spmatrix, sparse, spdiag, solvers

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
eps = .05 # linprog used 1, but small numerical errors

# save initial feasible point
W0 = W

def F(x=None, z=None):

    if x is None:

        # initial non-negative slacks
        S = np.fabs(W0 @ X) - eps
        assert (S >= 0).all()

        # initial feasible point
        x0 = matrix(np.concatenate((W0.flat, S.flat)))

        return 0, x0 # 0 non-linear constraints

    else:

        W = np.array(x[:M*N]).reshape((M, N)) # extract weights
        f, grad, hess = calc_derivatives(Y, W, X, A, K)

        # print("f", f)

        # convert flattened gradient
        Df = matrix(np.concatenate((grad.flat, np.zeros(M*V))).reshape(1,-1))

        # print("x", x.size)
        # print("Df", Df.size)

        if z is None: return (f, Df)

        # convert hessian
        H = [[None for _ in range(M+1)] for _ in range(M+1)]
        for (i,j) in it.product(range(M), repeat=2):
            # cvxopt wants blocks in column-major order, swap i and j
            if hess[i][j] is None:
                H[j][i] = spmatrix([], [], [], (N,N))
            else:
                H[j][i] = z * matrix(hess[i][j])
        for m in range(M):
            H[m][-1] = spmatrix([], [], [], (M*V, N))
            H[-1][m] = spmatrix([], [], [], (N, M*V))
        H[-1][-1] = spmatrix([], [], [], (M*V, M*V))
        H = sparse(H)

        # print("H", H.size)

        return (f, Df, H)

# # sanity check no crashes
# _, x0 = F()
# f, Df, H = F(x0, z=1)

G = sparse([[
    spmatrix([], [], [], (M*V, M*N)) # weights in any orthant
    ], [
    spdiag([-1]*M*V) # non-negative slack
]])
h = matrix([0.]*M*V)
dims = {
    'l': M*V, # non-negative slack variables
    'q': [], 's': []}

zeros = spmatrix([], [], [], (V,N))
YX = sparse([[zeros if i != j else matrix((X * Y[j]).T) for i in range(M)] for j in range(M)])
A_eq = sparse([[YX], [spdiag([-1]*M*V)]]) # Y[m] X.T @ W[m] - S[m]
b_eq = matrix([eps]*M*V) # = eps

print("G", G.size)
print("h", h.size)
print("A_eq", A_eq.size)
print("b_eq", b_eq.size)
print(dims['l'])

sol = solvers.cp(F, G, h, dims, A_eq, b_eq)

f, Df = F(sol['x'])
print('w', np.fabs(np.array(sol['x'])[:M*N]).min(), np.fabs(np.array(sol['x'])[:M*N]).max())
print("f", f)
print("|Df|", norm(np.array(Df)))

print('status', sol['status'])

