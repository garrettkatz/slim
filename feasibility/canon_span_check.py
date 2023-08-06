import itertools as it
import numpy as np
import pickle as pk
from adjacent_ltms import adjacency

def sign_sorter(w):
    # returns matrix S such that (S @ w) canonicalizes w
    N = len(w)
    S = np.zeros((N, N), dtype=int)
    sorter = np.argsort(np.fabs(w)) # w[sorter] is sorted
    S[np.arange(N), sorter] = 1
    S[:,w < 0] *= -1
    return S    

if __name__ == "__main__":

    # # check for sign sorter
    # w = np.random.permutation(np.arange(6))
    # w[::2] *= -1
    # S = sign_sorter(w)
    # wc = (S @ w)
    # ww = S.T @ wc
    # print(w)
    # print(wc)
    # print(ww)
    # assert (0 <= wc).all() and (wc[:-1] < wc[1:]).all()

    N = 4
    T = 1000 # number of transitions

    # load full regions and adjacencies
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    A, K = adjacency(Y)

    # load canonical regions, neighbors, span rule weights
    ltms = np.load(f"ltms_{N}_c.npz")
    Yc, Wc = ltms["Y"], ltms["W"]
    with open(f"adjs_{N}_c.npz", "rb") as f:
        (Yn, Wn) = pk.load(f)
    with open(f"ccg_ltm_{N}.pkl", "rb") as f:
        (Ws, loss_curve, extr_curve, gn_curve, pgn_curve) = pk.load(f)

    assert Wc.shape == Ws.shape
    assert (np.sign(Wc @ X) == np.sign(Ws @ X)).all()

    # initial point, transformed to match span representatives
    i = np.random.randint(len(W))
    wi, yi = W[i], Y[i]
    Si = sign_sorter(wi)
    yci = np.sign((Si @ wi) @ X)
    ci = (yci == Yc).all(axis=1).argmax()
    wi = Si.T @ Ws[ci]
    assert (np.sign(wi @ X) == yi).all()

    for t in range(T):

        # choose random neighbor
        n = np.random.randint(len(A[i]))
        j, k = A[i][n], K[i][n]
        print("ijk:", i, j, k)
        wj, xk, yj = W[j], X[:,k], Y[j]

        # joint-canonicalize adjacency
        yci = np.sign(np.sort(np.fabs(wi)) @ X)
        ycj = np.sign(np.sort(np.fabs(wj)) @ X)
        ci = (yci == Yc).all(axis=1).argmax()
        cj = (ycj == Yc).all(axis=1).argmax()
        ck = (Yc[ci] == Yc[cj]).argmin()
        # print(ci, cj, ck)
        assert (np.sign(Ws[ci] @ X) == yci).all()
        assert (np.sign(Ws[cj] @ X) == ycj).all()
        assert (Yc[ci,ck] != Yc[cj,ck])
        assert (Yc[ci] == Yc[cj]).sum() == (Yc.shape[1]-1)

        # get alpha-beta
        a, b = np.linalg.lstsq(np.vstack((Ws[ci], X[:,ck])).T, Ws[cj], rcond=None)[0]
        resid = np.fabs(a*Ws[ci] + b*X[:,ck] - Ws[cj]).max()
        # print(f"a={a}, b={b}, resid={resid}")
        assert((np.sign((a*Ws[ci]+b*X[:,ck]) @ X) == np.sign(Ws[cj] @ X)).all())

        # region-invariant symmetries
        Si = sign_sorter(wi)
        Ssi = sign_sorter(Ws[ci])
        Sj = sign_sorter(wj)
        Ssj = sign_sorter(Ws[cj])
        Ssx = sign_sorter(X[:,ck])
        Sx = sign_sorter(xk)

        # apply alpha-beta
        # wj = a*Sj.T @ Ssj @ Ssi.T @ Si @ wi + b*Sj.T @ Ssj @ X[:,ck]
        AS = Sj.T @ Ssj @ Ssi.T @ Si
        BS = Sj.T @ Ssj @ Ssx.T @ Sx
        wj = a * AS @ wi + b* BS @ xk
        assert (np.sign(wj @ X) == yj).all()

        # check that region is invariant under A
        # assert (np.sign((AS @ wi) @ X) == np.sign(wi @ X)).all()

        # any_match = False
        # for sa, sb in it.product((-1, +1), repeat=2):
        #     wj = sa*a*w + sb*b*xk
        #     # print(np.sign(wj @ X) - yj)
        #     print((np.sign(wj @ X) != yj) * np.fabs(wj @ X))
        #     if (np.sign(wj @ X) == yj).all():
        #         any_match = True
        #         break
        # assert any_match

        # update current region index
        wi = wj
        i = j

        print(f"{t} of {T} success...")

    print(f"all {T} passed.")

    
