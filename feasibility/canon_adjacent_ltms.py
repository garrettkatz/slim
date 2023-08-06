import sys
import pickle as pk
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings
from multiprocessing import Pool, cpu_count
from enumerate_ltms import check_feasibility

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

# save one core when multiprocessing
num_procs = cpu_count()-1

# If i and j are adjacent, so are Si and Sj for any hypercube symmetry S.
# This includes S that canonicalize i.
# So we only need compute adjacencies where one region is canonical.
#     X is the half-cube
#     Y[r] is the hemichotomy for canonical region r
def canon_adjacency(X, Y):

    # generate all neighbors of each canonical hemichotomy
    Yn, Wn = {}, {}
    for r, y in enumerate(Y):
        print(f"adjacencies to region {r} of {len(Y)}")

        # set up all potential neighbors of y with one bit flipped
        Yn[r] = y * ((-1) ** np.eye(len(y)))

        # prepare arguments for multiprocessing feasibility checks
        canonical = False # neighbor might not be canonical
        args = [(X, yn, canonical) for yn in Yn[r]]

        # check feasibilities of neighbors in parallel
        with Pool(num_procs) as pool:
            results = pool.map(check_feasibility, args)

        # discard infeasible neighbors
        feasible, w = zip(*results)
        feasible, w = np.array(feasible), np.stack(w)
        Yn[r] = Yn[r][feasible]
        Wn[r] = w[feasible]

    return Yn, Wn

if __name__ == "__main__":

    do_adj = True # whether to re-generate the adjacencies or only load them

    # get adjacencies up to dimension N_max
    if len(sys.argv) > 1:
        N_max = int(sys.argv[1])
    Ns = list(range(3, N_max + 1))

    for N in Ns:

        # load canonical hemis
        ltms = np.load(f"ltms_{N}_c.npz")
        Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    
        if do_adj:
            # calculate all adjacent neighbors
            Yn, Wn = canon_adjacency(X, Y)
            with open(f"adjs_{N}_c.npz", "wb") as f:
                pk.dump((Yn, Wn), f)
    
            # extract subset of joint-canonical adjacencies only
            A = set()
            for i, (yi, wi) in enumerate(zip(Y, W)):
    
                # process each neighbor
                for n, (yn, wn) in enumerate(zip(Yn[i], Wn[i])):
    
                    # canonicalize neighbor
                    wj = np.sort(np.fabs(wn))
                    yj = np.sign(wj @ X)
                    j = (yj == Y).all(axis=1).argmax()
                    assert (yj == Y[j]).all()

                    # empirically check that canonicalized neighbor is also adjacent
                    assert (yi == yj).sum() == len(yi)-1 # exactly one flipped bit 

                    # get boundary and add joint-canonicalized adjacency
                    k = (yi == yj).argmin()
                    A.add((i,j,k))
    
            A = tuple(A)
            with open(f"adjs_{N}_jc.npz", "wb") as f:
                pk.dump(A, f)

        # load results
        with open(f"adjs_{N}_c.npz", "rb") as f:
            (Yn, Wn) = pk.load(f)
        with open(f"adjs_{N}_jc.npz", "rb") as f:
            A = pk.load(f)
    
        # get total number of adjacencies stored
        num_edges = sum(map(len, Yn.values()))

        print(f"{N}: {len(Y)} canonical regions, {num_edges} neighbors stored, {len(A)} joint-canonical adjacencies")


