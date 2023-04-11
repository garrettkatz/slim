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

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

# np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'

# cvxopt function factory
# Y, W0, X from ltms, A, K from adjacency
def F_factory(Y, W0, X, A, K, eps):

    M, N = W0.shape # number of ltms, dimensions
    V = X.shape[1] # number of vertices (V == 2**N)

    def F(x=None, z=None):

        if x is None:

            # initial non-negative slacks
            S = np.empty((M, V))
            for m in range(M):
                S[m] = W0[m] @ (X * Y[m]) - eps

            # initial feasible point
            x0 = np.concatenate((W0.flat, S.flat))

            return 0, x0 # 0 non-linear constraints

        else:

            # extract variables
            W = x[:M*N].reshape((M, N))
            S = x[M*N:].reshape((M, V))

            # # check feasibility
            # for m in range(M):
            #     if (W[m] @ (X * Y[m]) - S[m] != eps).any():
            #         return None

            # evaluate span loss and derivatives
            f = 0.
            DW = np.zeros(W.shape)
            if z is not None:
                H = [[None for _ in range(M+1)] for _ in range(M+1)]
                H[-1][-1] = sp.csr_array((S.size, S.size))
            for i in range(M):

                if z is not None:
                    # accumulate hessian diagonal blocks
                    H[i][i] = sp.csr_matrix((N, N))

                for j, k in zip(A[i], K[i]):

                    # precompute recurring terms
                    wi, wj, xk = W[i], W[j], X[:,k]
                    Pk = np.eye(N) - xk.reshape((N, 1)) * xk / N # need for hessian
                    wiPk = wi - (wi @ xk) * xk / N
                    wjPk = wj - (wj @ xk) * xk / N
                    wiPk_n, wjPk_n = norm(wiPk), norm(wjPk)

                    # accumulate span loss
                    f += wiPk_n*wjPk_n - wiPk @ wjPk

                    # # accumulate gradient (one way, A sym False)
                    # DW[i] += wiPk * wjPk_n / wiPk_n - wjPk
                    # DW[j] += wjPk * wiPk_n  / wjPk_n - wiPk

                    # accumulate gradient (other way, when A sym=True)
                    DW[i] += 2 * (wiPk * wjPk_n / wiPk_n - wjPk)

                    # hessian blocks
                    if z is not None:
                        H[i][i] += z * 2 * (wjPk_n / wiPk_n) * (Pk - wiPk.reshape((N, 1)) * wiPk / wiPk_n**2)
                        # H[i][j]  = z * 2 * (wjPk.reshape((N, 1)) * wiPk / (wjPk_n * wiPk_n) - Pk)
                        H[i][j]  = z * 2 * (wiPk.reshape((N, 1)) * wjPk / (wiPk_n * wjPk_n) - Pk) # this is the right one, according to pytorch

            # flatten gradient
            Df = np.concatenate((DW.flat, np.zeros(S.size)))[np.newaxis,:]

            # convert hessian
            H = sp.bmat(H, format='csr')

            return (f, Df) if z is None else (f, Df, H)
                

        # at this point neither x nor z are None
        # calculate H, square dense/sparse matrix whose lower triangular part is z * Hessian(span loss)
        # https://cvxopt.org/userguide/solvers.html#cvxopt.solvers.cp
        

    return F

if __name__ == "__main__":

    N = 3
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    # A, K = adjacency(Y, sym=False) # sym=False just halves the objective function
    A, K = adjacency(Y, sym=True) # for "other way" of hand-calced gradient

    F = F_factory(Y, W, X, A, K, eps=1.)

    _, x0 = F()
    f, Df, H = F(x0, 1)
    DW = Df[0, :W.size].reshape(W.shape)
    H = H.toarray()[:W.size, :W.size]

    # print(H.shape)
    # pt.subplot(1,2,1)
    # pt.imshow(DW)
    # pt.subplot(1,2,2)
    # pt.imshow(H.toarray() != 0.)
    # pt.show()

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

    def span_loss(*ws):
        loss = tr.tensor(0.)
        for i in range(len(ws)):
            for j, k in zip(A[i], K[i]):
    
                # precompute recurring terms
                wi, wj, xk = ws[i], ws[j], X[:,k]
                wiPk = wi - (wi @ xk) * xk / N
                wjPk = wj - (wj @ xk) * xk / N
                wiPk_n, wjPk_n = tr.linalg.norm(wiPk), tr.linalg.norm(wjPk)
    
                # accumulate loss
                loss += wiPk_n*wjPk_n - wiPk @ wjPk
        return loss

    ws = tuple(w.clone().detach() for w in W)
    tH = tr.autograd.functional.hessian(span_loss, ws)
    tH = tr.vstack(tuple(map(tr.hstack, tH)))

    print(f"pytorch H error = {np.fabs(H - tH.numpy()).max()}")

    pt.subplot(1,2,1)
    pt.imshow(H)
    pt.subplot(1,2,2)
    pt.imshow(tH)
    pt.show()


    # for G see cvxopt floor planning example under:
    # https://cvxopt.org/userguide/solvers.html#cvxopt.solvers.cpl
