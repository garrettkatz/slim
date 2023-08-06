import numpy as np
from adjacent_ltms import adjacency

if __name__ == "__main__":

    N = 5
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    A = adjacency(Y)

    residuals = []
    na_i, na_j = [], []
    for i in range(len(A)):
        for j in A[i]:
            
            # column where they differ
            k = (Y[i] != Y[j]).argmax()

            # span condition: x_k ~ w_i * a + w_j * b = [w_i, w_j] @ [a; b]
            result = np.linalg.lstsq(np.stack((W[i], W[j]), axis=1), X[:,k], rcond=None)

            na_i.append(len(A[i]))
            na_j.append(len(A[j]))
            residuals.append( result[1][0] )

    import matplotlib.pyplot as pt
    # pt.bar(range(len(residuals)), residuals)
    pt.subplot(2,1,1)
    pt.plot(residuals)
    pt.xlabel("adjacent pair i,j")
    pt.ylabel("span residual")
    pt.subplot(2,1,2)
    pt.plot(na_i, label="|a_i|")
    pt.plot(na_j, label="|a_j|")
    pt.plot([x+y for (x,y) in zip(na_i, na_j)], label="|a_i|+|a_j|")
    pt.xlabel("adjacent pair")
    pt.legend()
    pt.tight_layout()
    pt.show()


