import sys
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
from scipy.special import factorial
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
# np.set_printoptions(formatter={"int": lambda x: "%+d" % x, "float": lambda x: "%.3f" % x}, linewidth=1000)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

num_procs = cpu_count()-1

def check_feasibility(args):
    X, y, canonical = args

    # region constraints
    A_ub = -(X * y).T
    b_ub = -np.ones(A_ub.shape[0])

    if canonical:

        # non-negative sorted constraint
        A_c = np.eye(N, k=-1) - np.eye(N) # w[0] >= 0, w[i] >= w[i-1]
        b_c = np.zeros(N)

        # combine with region constraints
        A_ub = np.concatenate((A_ub, A_c), axis=0)
        b_ub = np.concatenate((b_ub, b_c))

        # minimum weight objective when all weights positive
        c = np.ones(N)

    else:

        # make the problem bounded
        c = -A_ub.sum(axis=0)

    # run the linear program
    result = linprog(
        c = c,
        A_ub = A_ub,
        b_ub = b_ub,
        bounds = (None, None),
        method='simplex', # other methods miss some solutions
    )
    w = result.x

    # count failed runs as infeasible
    if w is None:
        return False, w # feasible=False, w

    # re-canonicalize solution in case of small round-off error
    if canonical:
        w = np.sort(np.fabs(w))

    # separate check that all region constraints satisfied
    feasible = (np.sign(w @ X) == y).all()

    # return results
    return feasible, w

def enumerate_ltms(N, canonical=True):

    # generate half-cube vertices
    X = np.array(tuple(it.product((-1, +1), repeat=N-1))).T
    X = np.vstack((-np.ones(2**(N-1), dtype=int), X))

    # initialize leading portion of hemichotomies
    Y = np.array([[-1, +1]]).T

    # initialize irredundant constraint tracking
    if canonical:
        irredundant = np.ones(Y.shape, dtype=bool)

    # iteratively extend hemichotomy tail
    for k in range(1, 2**(N-1)):
        print(f"{k} of {2**(N-1)}, {2*Y.shape[0]} dichots to check")

        # identify new redundancies
        if canonical:
            must_be_negative = ((X[:,k:k+1] <= X[:,:k]).all(axis=0) & (Y < 0)).any(axis=1)
            must_be_positive = ((X[:,k:k+1] >= X[:,:k]).all(axis=0) & (Y > 0)).any(axis=1)

        # append next possible bits to leading hemichotomy
        Y = np.block([
            [Y, -np.ones((Y.shape[0], 1), dtype=int)],
            [Y, +np.ones((Y.shape[0], 1), dtype=int)]])

        # track irredundant constraints
        if canonical:
            irredundant = np.block([
                [irredundant, ~must_be_negative.reshape(-1, 1)],
                [irredundant, ~must_be_positive.reshape(-1, 1)]])

        # check feasibility of each leading hemichotomy
        with Pool(num_procs) as pool:
            if canonical:
                args = [(X[:,:len(y)][:,keep], y[keep], canonical) for y, keep in zip(Y, irredundant)]
            else:
                args = [(X[:,:len(y)], y, canonical) for y in Y]
            results = pool.map(check_feasibility, args)
            feasible, W = zip(*results)

        # infeasible hemichotomies are not linearly separable, prune
        feasible = np.array(feasible)
        Y = Y[feasible]
        if canonical:
            irredundant = irredundant[feasible]

    W = np.stack(W)[feasible]

    return Y, W, X

if __name__ == "__main__":

    do_gen = True
    canonical = True

    if len(sys.argv) > 1:
        Ns = [int(sys.argv[1])]
    else:
        Ns = list(range(3,9))

    for N in Ns:
        fname = f"ltms_{N}{'_c' if canonical else ''}.npz"

        if do_gen:
            Y, W, X = enumerate_ltms(N, canonical)
            np.savez(fname, Y=Y, W=W, X=X)
        else:
            ltms = np.load(fname)
            Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

        message = f"N={N}: {len(Y)} hemis"
    
        # print(W.round(1).T)
        # print(np.hstack((W.round(1), Y)))
        # print(Y.shape, W.shape, X.shape)

        # count with all symmetries, only works for integer weights N < 9
        if canonical:
            num_sym = 0
            for i, w in enumerate(W.round()):
                # get multiset coefficient
                uni = {}
                for n in range(N): uni[w[n]] = uni.get(w[n], 0) + 1
                num_sym_i = factorial(sum(uni.values()))
                for v in uni.values(): num_sym_i /= factorial(v)
                num_sym += num_sym_i * 2**np.count_nonzero(w)

            message += f" ({num_sym} non canonical)"

        print(message)

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
