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

# helper function for derivative calculation.  returns:
# loss: scalar
# grad: (M,N) numpy array; grad[m] is dL / dW[m]
# hess: (M,M) list of lists.  hess[i][j] is None if all zeros else (N,N) numpy array d2L / (dW[i] dW[j])
# expects A, K from sym=True
def calc_derivatives(Y, W, X, A, K, sym=True):

    M, N = W.shape # number of ltms, dimensions
    V = X.shape[1] # number of vertices (V == 2**(N-1))

    loss = 0.
    grad = np.zeros(W.shape)
    hess = [[None for _ in range(M)] for _ in range(M)]

    min_norm = np.inf
    for i in range(M):

        # accumulate hessian diagonal blocks
        hess[i][i] = np.zeros((N, N))

        for j, k in zip(A[i], K[i]):

            # precompute recurring terms
            wi, wj, xk = W[i], W[j], X[:,k]
            Pk = np.eye(N) - xk.reshape((N, 1)) * xk / N
            wiPk = wi @ Pk
            wjPk = wj @ Pk
            wiPk_n, wjPk_n = norm(wiPk), norm(wjPk)

            min_norm = min(min_norm, wiPk_n, wjPk_n)

            # accumulate span loss
            loss += wiPk_n*wjPk_n - wiPk @ wjPk

            # accumulate gradient (one way, A sym False)
            if not sym:
                grad[i] += wiPk * wjPk_n / wiPk_n - wjPk
                grad[j] += wjPk * wiPk_n  / wjPk_n - wiPk

            # accumulate gradient (other way, when A sym=True)
            else:
                grad[i] += 2 * (wiPk * wjPk_n / wiPk_n - wjPk)

            # hessian blocks, not working for non-sym yet
            if sym:
                hess[i][i] += 2 * (wjPk_n / wiPk_n) * (Pk - wiPk.reshape((N, 1)) * wiPk / wiPk_n**2)
                hess[i][j]  = 2 * (wiPk.reshape((N, 1)) * wjPk / (wiPk_n * wjPk_n) - Pk)

    # print(f'|wiPk| >= {min_norm}')

    return loss, grad, hess

if __name__ == "__main__":

    N = 4
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    # A, K = adjacency(Y, sym=False) # sym=False just halves the objective function
    A, K = adjacency(Y, sym=True) # for "other way" of hand-calced gradient

    f, DW, H = calc_derivatives(Y, W, X, A, K)
    H = sp.bmat(H, format='csr').toarray()

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

    # # quick check on off-chance hessian is itself an orthogonal projection?
    # HH = H @ H
    # print(f"|HH - H| <= {np.fabs(HH - H).max()}")
    # print(f"|HH[H == 0]| <= {np.fabs(HH[H == 0]).max()}")

    # imgs = [H, tH, HH]
    imgs = [H, tH]

    for i, img in enumerate(imgs):
        pt.subplot(1,len(imgs),i+1)
        pt.imshow(img)
    pt.show()

    # for G see cvxopt floor planning example under:
    # https://cvxopt.org/userguide/solvers.html#cvxopt.solvers.cpl
