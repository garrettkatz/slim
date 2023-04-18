import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

# warnings.filterwarnings("ignore", category=LinAlgWarning)
# warnings.filterwarnings("ignore", category=OptimizeWarning)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

def enumerate_ltms(N):

    X = np.array(tuple(it.product((-1, +1), repeat=N))).T
    X = X[:,:2**(N-1)] # more numerically stable linprog without antiparallel data

    Y = np.array([[1, -1]]).T
    for j in range(1, 2**(N-1)):
        print(f"{j} of {2**(N-1)}")

        Y = np.block([
            [Y, +np.ones((Y.shape[0], 1), dtype=int)],
            [Y, -np.ones((Y.shape[0], 1), dtype=int)]])

        feasible = np.empty(Y.shape[0], dtype=bool)
        W = {}
        for k, y in enumerate(Y):
        
            result = linprog(
                c = (X[:,:j+1] * y).sum(axis=1),
                A_ub = -(X[:,:j+1] * y).T,
                b_ub = -np.ones(j+1),
                bounds = (None, None),
            )
            if result.x is not None:
                W[k] = result.x
                feasible[k] = (np.sign(W[k] @ X[:,:j+1]) == y).all() # sanity check
            else:
                feasible[k] = False

        Y = Y[feasible]

    W = np.stack([W[k] for k in np.flatnonzero(feasible)])

    return Y, W, X

if __name__ == "__main__":

    # for N in range(3,6):
    for N in range(3,5):
    # for N in range(3,4):
        print(N)

        Y, W, X = enumerate_ltms(N)
        print(Y.shape, W.shape, X.shape)
        np.savez(f"ltms_{N}.npz", Y=Y, W=W, X=X)

        # print(W)

    # # full X and Y
    # X = np.array(tuple(it.product((-1, +1), repeat=N))).T
    # Y = np.sign(W @ X)
    # pad = 2
    # YWX = np.full((Y.shape[0], Y.shape[1] + W.shape[1] + X.shape[1] + 2*pad), np.nan)
    # YWX[:,:Y.shape[1]] = Y
    # YWX[:,Y.shape[1]+pad:Y.shape[1]+pad+W.shape[1]] = W
    # YWX[:X.shape[0], Y.shape[1]+pad+W.shape[1]+pad:] = X

    # pt.imshow(YWX)
    # pt.title("Y = sign(WX)")
    # pt.axis('off')
    # pt.colorbar()
    # pt.show()