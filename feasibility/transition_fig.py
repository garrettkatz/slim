import itertools as it
import numpy as np
import matplotlib.pyplot as pt

if __name__ == "__main__":

    N = 3

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # full X and Y
    X = np.array(tuple(it.product((-1, +1), repeat=N))).T
    Y = np.sign(W @ X)
    pad = 2

    for a in [.1, 1.0]:

        pt.figure(figsize=(4,3.8))

        js = [0, 1, 3, 0]
        for sp, i in enumerate([0, 1, 3, 9, 8]):
    
            YWX = np.full((X.shape[0], Y.shape[1] + W.shape[1] + X.shape[1] + 2*pad), np.nan)
            YWX[1,:Y.shape[1]] = Y[i]
            YWX[1,Y.shape[1]+pad:Y.shape[1]+pad+W.shape[1]] = W[i]
            YWX[:X.shape[0], Y.shape[1]+pad+W.shape[1]+pad:] = X
    
            A = np.full((X.shape[0], Y.shape[1] + W.shape[1] + X.shape[1] + 2*pad), a)
            A[1, Y.shape[1]+pad:Y.shape[1]+pad+W.shape[1]] = 1
            for j in js[:sp]:
                A[1,j] = 1
                A[:X.shape[0], Y.shape[1]+pad+W.shape[1]+pad+j] = 1
    
            pt.subplot(5,1,sp+1)
            pt.imshow(YWX, alpha=A)
            pt.axis('off')

        pt.tight_layout()
        pt.savefig(f"transitions_{a}.png")
        pt.show()


