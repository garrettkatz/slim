import sys
import pickle as pk
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

def canon_adjacency(X, Yc):

    Yn, Wn = {}, {}
    for i, yc in enumerate(Yc):
        print(f"adjacencies to region {i} of {len(Yc)}")
        Yn[i], Wn[i] = [], []
        for k in range(Yc.shape[1]):
            yn = yc.copy()
            yn[k] *= -1

            # check neighbor feasibility
            A_ub = -(X * yn).T
            b_ub = -np.ones(len(A_ub))
            c = -A_ub.sum(axis=0)

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
                wn = result.x
                if (yn == np.sign(wn @ X)).all():
                    Yn[i].append(yn)
                    Wn[i].append(wn)

        Yn[i] = np.stack(Yn[i])
        Wn[i] = np.stack(Wn[i])

    return Yn, Wn

if __name__ == "__main__":

    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 3

    do_adj = True
    do_adj = False

    ltms = np.load(f"ltms_{N}_c.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    if do_adj:
        Yn, Wn = canon_adjacency(X, Y)
        with open(f"adjs_{N}_c.npz", "wb") as f:
            pk.dump((Yn, Wn), f)

    with open(f"adjs_{N}_c.npz", "rb") as f:
        (Yn, Wn) = pk.load(f)

    num_edges = 0
    for i, (yc, wc) in enumerate(zip(Y, W)):
        print("% 2d" % i, yc, wc.round().astype(int))
        for yn, wn in zip(Yn[i], Wn[i]):
            # print("  ", yn, wn.astype(int), (yc != yn).argmax())
            num_edges += 1

    print(f"{num_edges} adjacencies stored")

    # canonical adjacency matrix
    Ac = set()
    ec_neighbors = {}
    for i in range(len(Yn)):
        ec_neighbors[i] = set()
        for j in range(len(Yn[i])):
            # canonicalize neighbor
            w = np.sort(np.fabs(Wn[i][j]))
            y = np.sign(w @ X)
            j = (y == Y).all(axis=1).argmax()
            assert (y == Y[j]).all()
            k = (Y[j] == Y[i]).argmin()
            Ac.add((i,j,k))
            ec_neighbors[i].add(j)
    Ac = tuple(Ac)
    num_ec_neighbors = [(i, len(ec_neighbors[i])) for i in ec_neighbors]
    print(f"{len(W)} canonical regions, {len(Ac)//2} joint-canonical adjacencies, ec neighbors:")
    print(num_ec_neighbors)
    print(ec_neighbors)

    # check whether all adjacencies have a joint canonicalization
    for i, (yc, wc) in enumerate(zip(Y, W)):
        for yn, wn in zip(Yn[i], Wn[i]):

            # canonicalize neighbor
            wnc = np.sort(np.fabs(wn))
            # get its hemichotomy
            ync = np.sign(wnc @ X)
            # check that canonicalized hemichotomy is also adjacent
            assert (ync == Yn[i]).all(axis=1).any()

    print("all adjacencies joint-canonicalizable")

