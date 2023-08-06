import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from sklearn.svm import LinearSVC
from adjacent_ltms import adjacency  

def runner(X, Y, A, K, w, i, k, max_len):
    # w for current region
    # taking step across boundary normal k, landing in region i

    # w' = aw + bx = [w, x] @ [a; b]
    # x.T' @ w' = x.T' @ [w, x] @ [a; b]

    wx = np.stack((w, X[:,k])).T

    inp = X.T @ wx
    out = Y[i]

    inp = np.concatenate((inp, -inp))
    out = np.concatenate((out, -out))

    svc = LinearSVC(fit_intercept=False, max_iter=1000)
    svc.fit(inp, out)
    acc = svc.score(inp, out)
    # assert acc == 1.0
    print(" " * (5 - max_len), k, i, acc)

    ab = svc.coef_.flatten()
    w = wx @ ab

    if max_len == 0: return

    for j, k in zip(A[i], K[i]):
        runner(X, Y, A, K, w, j, k, max_len - 1)

if __name__ == "__main__":

    # N, L = 3, 8
    N, L = 4, 3

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    A, K = adjacency(Y)

    w = np.ones(N)
    y = np.sign(w @ X)
    i = (Y == y).all(axis=1).argmax()

    for j, k in zip(A[i], K[i]):
        runner(X, Y, A, K, w, j, k, max_len = L)

