import itertools as it
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

def enumerate_ltms(N):

    X = np.array(tuple(it.product((-1, +1), repeat=N))).T
    Xh = X[:,:2**(N-1)] # more numerically stable linprog without antiparallel data

    Y = np.array([[1, -1]]).T
    for j in range(1, 2**(N-1)):
        print(f"{j} of {2**(N-1)}")

        Y = np.block([
            [Y, +np.ones((Y.shape[0], 1))],
            [Y, -np.ones((Y.shape[0], 1))]])

        feasible = np.empty(Y.shape[0], dtype=bool)
        for k, y in enumerate(Y):
        
            result = linprog(
                c = (Xh[:,:j+1] * y).sum(axis=1),
                A_ub = -(Xh[:,:j+1] * y).T,
                b_ub = -np.ones(j+1),
                bounds = (None, None),
            )
            w = result.x
            feasible[k] = (np.sign(w @ Xh[:,:j+1]) == y).all()

        Y = Y[feasible]

    return Y

if __name__ == "__main__":

    Y = enumerate_ltms(5)
    print(Y.shape)

