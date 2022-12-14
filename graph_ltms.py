import numpy as np
from adjacent_ltms import adjacency

np.set_printoptions(threshold=10e6)

if __name__ == "__main__":

    N = 3
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # # rearrange antipodes to be opposite
    # for i in range(Y.shape[0]//2):
    #     j = Y.shape[0]//2 + (Y[i] == Y[Y.shape[0]//2:]).all(axis=1).argmax()
    #     k = Y.shape[0] - i - 1
    #     print(k,j)
    #     Y[[j, k],:] = Y[[k, j],:]

    # label dichotomies by distance to identity regions
    sX = np.concatenate((X, -X), axis=0)
    dist = (Y != sX[:,np.newaxis,:]).sum(axis=2).min(axis=0)
    udist = np.unique(dist)
    # print(sX)
    # print(np.concatenate((Y, dist[:,np.newaxis], W.round().astype(int)), axis=1))

    for d in udist:
        print(f"d = {d}: {(dist==d).sum()} regions")

    # get adjacencies for graph edges
    A = adjacency(Y, sym=True)

    # assign node indices and x,y coordinates in each ring
    idx = np.empty(Y.shape[0], dtype=int)
    x, y = np.empty((2, Y.shape[0]))
    for d in udist:
        mask = (dist == d)
        num = mask.sum()
        ang = 2 * np.pi * np.arange(num) / num
        rad = d + 1
        x[mask] = rad * np.cos(ang)
        y[mask] = rad * np.sin(ang)
        idx[mask] = np.arange(num)

    # draw graph
    import matplotlib.pyplot as pt
    for i in range(len(A)):
        for j in A[i]:
            pt.plot(x[[i, j]], y[[i, j]], 'ko-')
    pt.show()


