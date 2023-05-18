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
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

# save one core when multiprocessing
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
        feasible, w = False, np.empty(N)
        return feasible, w

    # re-canonicalize solution in case of small round-off error
    if canonical:
        w = np.sort(np.fabs(w))

    # separate check that all region constraints satisfied
    feasible = (np.sign(w @ X) == y).all()

    # return results
    return feasible, w


# assumes integer weights w invariant to same symmetries as their regions
# may break for N >= 8 [Muroga 1970]
def get_equivalence_class_size(w):

    # get multiset of weight values and their counts
    multiset = {}
    for n in range(N):
        multiset[w[n]] = multiset.get(w[n], 0) + 1

    # count distinct permutations by multiset coefficient
    num_perms = factorial(sum(multiset.values()))
    for v in multiset.values():
        num_perms /= factorial(v)

    # any non-zero weight can also have its sign flipped
    size = num_perms * 2**np.count_nonzero(w)

    return size

def enumerate_ltms(N, canonical=True):

    # generate half-cube vertices
    X = np.array(tuple(it.product((-1, +1), repeat=N-1))).T
    X = np.vstack((-np.ones(2**(N-1), dtype=int), X))

    # initialize weights and leading portion of hemichotomies
    if canonical:
        # canonical weights can not produce +1 on all X[:,0] == -1
        Y = np.array([[-1]]).T
    else:
        Y = np.array([[-1, +1]]).T

    # initialize irredundant constraint tracking
    if canonical:
        irredundant = np.ones(Y.shape, dtype=bool)
        # also maintain feasibilities and weights in this mode
        feasible = np.ones(len(Y), dtype=bool)
        W = np.empty((len(Y), N))

    # iteratively extend hemichotomy tails
    for k in range(1, 2**(N-1)):
        print(f"{k} of {2**(N-1)}, {2*Y.shape[0]} leading hemis to check")

        # track new redundancies
        if canonical:
            # uses positivity of the weights
            must_be_negative = ((X[:,k:k+1] <= X[:,:k]).all(axis=0) & (Y < 0)).any(axis=1)
            must_be_positive = ((X[:,k:k+1] >= X[:,:k]).all(axis=0) & (Y > 0)).any(axis=1)
            irredundant = np.block([
                [irredundant, ~must_be_negative.reshape(-1, 1)],
                [irredundant, ~must_be_positive.reshape(-1, 1)]])

            # duplicate W to match Y below
            W = np.tile(W, (2,1))

        # append next possible bits to leading hemichotomies
        Y = np.block([
            [Y, -np.ones((Y.shape[0], 1), dtype=int)],
            [Y, +np.ones((Y.shape[0], 1), dtype=int)]])

        # prepare arguments for multiprocessing feasibility check
        Xk = X[:,:k+1] # leading vertices
        if canonical:
            # only check hemichotomies where new constraint is not redundant
            check = irredundant[:,-1]
            # omit redundant constraints from check
            args = [(Xk[:,irr], y[irr], canonical) for y, irr in zip(Y[check], irredundant[check])]
        else:
            args = [(Xk, y, canonical) for y in Y]

        # check feasibility of leading hemichotomies in parallel
        with Pool(num_procs) as pool:
            results = pool.map(check_feasibility, args)

        # prune hemichotomies that are not linearly separable
        if canonical:
            # gather results
            feasible, w = zip(*results)

            # expand to all hemichotomies, not only those checked
            keep = np.ones(len(Y), dtype=bool)
            keep[check] = feasible
            W[check] = np.stack(w)

            # only keep the feasible ones
            Y = Y[keep]
            W = W[keep]
            irredundant = irredundant[keep]

        else:
            feasible, W = zip(*results)
            Y = Y[feasible]
            W = np.stack(W)[feasible]

    return Y, W, X

if __name__ == "__main__":

    do_gen = False # whether to re-generate the hemichotomies or only load them
    canonical = True # whether to enumerate canonical hemichotomies only

    # enumerate up to dimension N_max
    if len(sys.argv) > 1:
        N_max = int(sys.argv[1])
    Ns = list(range(3, N_max + 1))

    # process dimensions one at a time
    for N in Ns:
        fname = f"ltms_{N}{'_c' if canonical else ''}.npz"

        if do_gen:
            Y, W, X = enumerate_ltms(N, canonical)
            np.savez(fname, Y=Y, W=W, X=X)
        else:
            ltms = np.load(fname)
            Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

        # display number of linearly separable hemichotomies
        message = f"N={N}: {len(Y)} hemis"
    
        # count all regions (including non-canonial)
        if canonical:

            # accumulate equivalence class size of each weight vector
            region_count = 0
            for i, w in enumerate(W.round()):
                region_count += get_equivalence_class_size(w)

            message += f" ({region_count} regions including non-canonical)"

        print(message)

