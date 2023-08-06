import os, sys
from collections import deque
import itertools as it
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

def intersect(p, x):
    N = len(x)
    anum, aden, xa = None, None, None
    for k in range(1, int(np.ceil(N / 2))):
        for idx in map(list, it.combinations(range(N), k)):
            x_ki = x.copy()
            x_ki[idx] *= -1
            # if 0 < -(p @ x_ki) / (N - 2*k) < anum / aden:
                # a = -(p @ x_ki) / (N - 2*k)
            if 0 > -p @ x_ki: continue
            if xa is None or -(p @ x_ki) * aden < (N - 2*k) * anum:
                anum = -(p @ x_ki)
                aden = (N - 2*k)
                xa = x_ki            
    return anum / aden, xa

if __name__ == "__main__":

    # warnings.filterwarnings("ignore", category=LinAlgWarning)
    # warnings.filterwarnings("ignore", category=OptimizeWarning)
    np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)
    
    N = int(sys.argv[1])
    fname = f"hemigraph_{N}.npz"
    
    X = np.array(tuple(it.product((-1, +1), repeat=N))).T
    Nh = 2**(N-1)
    Xh = X[:,:Nh]
    
    npz = np.load(fname)
    hemis, depths, boundaries, weights, anchors = npz["hemis"], npz["depths"], npz["boundaries"], npz["weights"], npz["anchors"]

    for a,anchor in enumerate(np.flatnonzero(anchors)):
        for b,bound in enumerate(np.flatnonzero(boundaries[anchor])):
            w = weights[anchor]
            x = Xh[:,bound] * hemis[anchor,bound]
            # p = w - x / N
            p = N*w - x

            # alpha, x_int = intersect(p, x)
            # bound_int = (np.fabs(x_int @ Xh) == N).argmax()
            
            # brute, confirmed up to N=6
            alphas = (1 - p @ X) / (x @ X)
            gz = np.flatnonzero(alphas >= 0)
            bound_int = gz[alphas[gz].argmin()]
            alpha = alphas[bound_int]
            if bound_int >= Nh: bound_int = 2**N - 1 - bound_int
            x_int = Xh[:,bound_int]

            # # hueristic
            # pidx = np.argsort(-np.fabs(p)) # largest to smallest magnitude
            # numer = 1
            # denom = 0
            # x_pr = []
            # for k in range(N):
            #     nkp = numer - p[pidx[k]]
            #     dkp = denom + x[k]
            #     nkn = numer + p[pidx[k]]
            #     dkn = denom - x[k]
            #     if (nkp * dkp < 0)

            # wrong direction? intersect explicitly with each boundary
            # (p + alpha x) xb = 0
            # p xb + alpha x xb = 0
            # alpha = -(p xb) / (x xb)
            alphab = - (p @ Xh[:,boundaries[anchor]]) / (x @ Xh[:,boundaries[anchor]])

            print(a,b, "w,x,wx,p,h,s(pX),a,xi,bi,p+ax,(p+ax)xi,boundaries, a with boundaries, Xh, w @ Xh[bi], w @ x_int")
            print(w)
            print(x)
            print(w @ x)
            print(p)
            print(hemis[anchor])
            print(np.sign(p @ Xh))
            print(alpha)
            print(x_int)
            print(bound_int)
            print(p + alpha*x)
            print((p + alpha*x) @ x_int)
            print(boundaries[anchor].astype(int))
            print(alphab)
            print(Xh)
            print(w @ Xh[:,bound_int])
            print(w @ x_int)

            assert boundaries[anchor, bound_int]

