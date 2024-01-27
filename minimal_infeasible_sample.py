import sys
import itertools as it
import pickle as pk
import numpy as np
import matplotlib
import matplotlib.pyplot as pt
from check_span_rule import *

do_exp = True
do_show = True

solver = sys.argv[1]
N = int(sys.argv[2])

with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, B, W = (regions[key] for key in ("XYBW"))

# B = np.load(f"boundaries_{N}_{solver}.npy")

sorter = np.argsort(B.sum(axis=1))
Y = Y[sorter]
B = B[sorter]
W = W[sorter]

if do_exp:

    found_infeasible = False
    for num_regions in range(1, len(Y)):
        for sample in it.combinations(range(len(Y)), num_regions):

            print("region sample", sample)
            result = check_span_rule(X, Y[sample], B[sample], W[sample], solver, verbose=False)
            status, u, ɣ, num_nodes = result

            if status != "optimal":
                found_infeasible = True
                fname = f"minimal_infeasible_{N}.pkl"
                with open(fname, 'wb') as f: pk.dump((result, sample), f)

            if found_infeasible: break
        if found_infeasible: break

    if not found_infeasible:
        print("All sub-samples feasible")

if do_show:

    fname = f"minimal_infeasible_{N}.pkl"
    if not os.path.exists(fname):
        print("All sub-samples feasible")
        sys.exit(1)

    with open(fname, 'rb') as f: result, sample = pk.load(f)

    pt.subplot()
    pt.imshow(B[sample].T)
    pt.title("Boundaries")
    pt.axis("off")

    pt.subplot()
    pt.imshow(X)
    pt.title("Vertices")
    pt.axis("off")

    pt.subplot()
    pt.imshow(Y.T)
    pt.title("Dichotomies")
    pt.axis("off")

    pt.show()

