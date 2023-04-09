import numpy as np
from numpy.linalg import norm
from adjacent_ltms import adjacency
import itertools as it
import matplotlib.pyplot as pt
import matplotlib as mp
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'

# cvxopt function factory
# Y, W0, X from ltms, A, K from adjacency
def F_factory(Y, W0, X, A, K, eps):

    M, N = W0.shape # number of ltms, dimensions
    V = X.shape[1] # number of vertices (V == 2**N)

    def F(x=None, z=None):

        if (x is None) and (z is None):

            # initial non-negative slacks
            S = np.empty((M, V))
            for m in range(M):
                S[m] = W0[m] @ (X * Y[m]) - eps

            # initial feasible point
            x0 = np.concatenate((W0.flat, S.flat))

            return 0, x0 # 0 non-linear constraints

        if z is None:

            # extract variables
            W = x[:M*N].reshape((M, N))
            S = x[M*N:].reshape((M, V))

            # # check feasibility
            # for m in range(M):
            #     if (W[m] @ (X * Y[m]) - S[m] != eps).any():
            #         return None

            # evaluate span loss and derivative
            f = 0.
            DW = np.zeros(W.shape)
            for i in range(M):
                for j, k in zip(A[i], K[i]):

                    # precompute recurring terms
                    wi, wj, xk = W[i], W[j], X[:,k]
                    wiPk = wi - (wi @ xk) * xk / N
                    wjPk = wj - (wj @ xk) * xk / N
                    wiPk_n, wjPk_n = norm(wiPk), norm(wjPk)

                    # accumulate span loss
                    f += wiPk_n*wjPk_n - wiPk @ wjPk

                    # accumulate gradient
                    DW[i] += wiPk * wjPk_n / wiPk_n - wjPk
                    DW[j] += wjPk * wiPk_n  / wjPk_n - wiPk

            # flatten gradient
            Df = np.concatenate((DW.flat, np.zeros(S.size)))[np.newaxis,:]

            return f, Df

        # at this point neither x nor z are None
        # calculate H, square dense/sparse matrix whose lower triangular part is z * Hessian(span loss)
        # https://cvxopt.org/userguide/solvers.html#cvxopt.solvers.cp
        

    return F

if __name__ == "__main__":

    N = 4
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    A, K = adjacency(Y, sym=False) # sym=False just halves the objective function

    F = F_factory(Y, W, X, A, K, eps=1.)

    _, x0 = F()
    f, Df = F(x0)
    DW = Df[0, :W.size].reshape(W.shape)

    print(f"span loss = {f}")
    # print(Df)

    # check against pytorch
    import torch as tr

    W, X = tr.tensor(W, requires_grad=True), tr.tensor(X).double()
    f = 0.
    for i in range(W.shape[0]):
        for j, k in zip(A[i], K[i]):

            # precompute recurring terms
            wi, wj, xk = W[i], W[j], X[:,k]
            wiPk = wi - (wi @ xk) * xk / N
            wjPk = wj - (wj @ xk) * xk / N
            wiPk_n, wjPk_n = tr.linalg.norm(wiPk), tr.linalg.norm(wjPk)

            # accumulate gradient
            loss = wiPk_n*wjPk_n - wiPk @ wjPk
            loss.backward()
            f += loss.item()

    print(f"torch f = {f}")
    if N <= 3:
        print("hand DW:")
        print(DW)
        print("torch DW:")
        print(W.grad.numpy())
    print(f"pytorch error = {np.fabs(DW - W.grad.numpy()).max()}")

    # for G see cvxopt floor planning example under:
    # https://cvxopt.org/userguide/solvers.html#cvxopt.solvers.cpl
