import numpy as np
import scipy.optimize as so

def solve(X, Y):
    # sign(W @ X) = Y
    # sign(X.T @ W.T) == Y.T
    #   Y.T[:,i] * X.T @ W[i,:].T > 0
    # - Y.T[:,i] * X.T @ W[i,:].T <= -1
    W = []
    for i in range(Y.shape[0]):
        A_ub = -Y.T[:,i:i+1] * X.T
        b_ub = -np.ones(X.shape[1])
        c = -A_ub.mean(axis=0)
        #c = A_ub.mean(axis=0)
        result = so.linprog(c, A_ub, b_ub, bounds=(None, None))#, method='simplex')
        if not result.status in (0, 3): return False, W
        W.append(result.x)
    W = np.array(W)
    assert (np.sign(W @ X) == Y).all()
    return True, W

