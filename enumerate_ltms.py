import sys
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
from scipy.special import factorial
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
# np.set_printoptions(formatter={"int": lambda x: "%+d" % x, "float": lambda x: "%.3f" % x}, linewidth=1000)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

def enumerate_ltms(N, canonical=False):

    # more numerically stable linprog without antiparallel data
    X = np.array(tuple(it.product((-1, +1), repeat=N-1))).T
    X = np.vstack((-np.ones(2**(N-1), dtype=int), X))

    # additional constraints for canonical
    if canonical:
        A_c = np.eye(N, k=-1) - np.eye(N) # w[0] >= 0, w[i] >= w[i-1]
        b_c = np.zeros(N)

    Y = np.array([[-1, +1]]).T
    for j in range(1, 2**(N-1)):
        print(f"{j} of {2**(N-1)}, {2*Y.shape[0]} dichots to check")

        Y = np.block([
            [Y, -np.ones((Y.shape[0], 1), dtype=int)],
            [Y, +np.ones((Y.shape[0], 1), dtype=int)]])

        feasible = np.empty(Y.shape[0], dtype=bool)
        W = {}
        Y_set = set() # for canonical
        for k, y in enumerate(Y):

            A_ub = -(X[:,:j+1] * y).T
            b_ub = -np.ones(j+1)
            c = -A_ub.sum(axis=0)

            if canonical:
                A_ub = np.concatenate((A_ub, A_c), axis=0)
                b_ub = np.concatenate((b_ub, b_c))
                c = np.ones(N) # minimum weight objective when all weights positive

            result = linprog(
                c = c,
                A_ub = A_ub,
                b_ub = b_ub,
                bounds = (None, None),
                method='simplex',
                # method='highs-ipm',
                # method='revised simplex', # this and high-ds miss some solutions
            )
            if result.x is not None:
                W[k] = result.x

                if canonical:
                    # canonicalize to counteract rounding error in w order and non-negative constraints
                    W[k] = np.sort(np.fabs(W[k]))
                    yk = np.sign(W[k] @ X[:,:j+1]).astype(int)
                    feasible[k] = (yk == y).all()

                    # avoid canonical duplicates
                    yk = tuple(yk)
                    if yk in Y_set:
                        feasible[k] = False # flag duplicate for removal
                    elif feasible[k]:
                        Y_set.add(yk)

                else:
                    # not canonical, just sanity check region constraints
                    yk = np.sign(W[k] @ X[:,:j+1]).astype(int)
                    feasible[k] = (yk == y).all()

            else:
                feasible[k] = False

        Y = Y[feasible]

    W = np.stack([W[k] for k in np.flatnonzero(feasible)])

    return Y, W, X

if __name__ == "__main__":

    do_gen = False
    canonical = True

    if len(sys.argv) > 1:
        Ns = [int(sys.argv[1])]
    else:
        # Ns = np.arange(3,6)
        # Ns = np.arange(3,5)
        # Ns = np.arange(3,4)
        # Ns = np.arange(6, 7)
        Ns = [3]

    for N in Ns:
        print(N)
        fname = f"ltms_{N}{'_c' if canonical else ''}.npz"

        if do_gen:
            Y, W, X = enumerate_ltms(N, canonical)
            np.savez(fname, Y=Y, W=W, X=X)
        else:
            ltms = np.load(fname)
            Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    
        print(W.round(1))
        # print(np.hstack((W.round(1), Y)))
        print(Y.shape, W.shape, X.shape)

        # count with all symmetries, only works for integer weights N < 8
        if canonical:
            num_sym = 0
            for i, w in enumerate(W.round()):
                # get multiset coefficient
                uni = {}
                for n in range(N): uni[w[n]] = uni.get(w[n], 0) + 1
                num_sym_i = factorial(sum(uni.values()))
                for v in uni.values(): num_sym_i /= factorial(v)
                num_sym += num_sym_i * 2**np.count_nonzero(w)

            print(f"{num_sym} regions with symmetry")

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
