import numpy as np
from numpy.linalg import norm
from adjacent_ltms import adjacency
import itertools as it
import matplotlib.pyplot as pt
import matplotlib as mp
import scipy.sparse as sp
from scipy.optimize import minimize, LinearConstraint

# from span_loss_derivatives import calc_derivatives
from sq_span_loss_derivatives import calc_derivatives

# np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'

N = 5
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

def loss_fn(x):
    W = np.array(x).reshape((M, N)) # extract weights
    f = calc_derivatives(Y, W, X, A, K, out=('loss',))
    return f

def jac(x):
    W = np.array(x).reshape((M, N)) # extract weights
    _, grad = calc_derivatives(Y, W, X, A, K, out=('loss', 'grad'))
    return grad.flatten()

def hess(x):
    W = np.array(x).reshape((M, N)) # extract weights
    _, _, H = calc_derivatives(Y, W, X, A, K)
    H = sp.bmat(H)
    return H

YX = sp.bmat([[None if i != j else (X * Y[j]).T for i in range(M)] for j in range(M)])
constraint = LinearConstraint(YX, lb=eps, ub=np.inf)

x0 = W0.flatten()
result = minimize(loss_fn, x0, jac=jac, hess=hess, constraints=constraint, method='trust-constr', options = {'verbose': 3})
print(result.message)

W = result.x.reshape((M, N))
loss, grad, hess = calc_derivatives(Y, W, X, A, K)

print("loss", loss)
feas = ((W @ X) * Y)
print("feas", (feas - eps).min(), (feas - eps).max())
print("|w|", norm(W, axis=1).min(), norm(W, axis=1).max())


