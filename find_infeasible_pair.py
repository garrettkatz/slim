import sys, os
import itertools as it
import pickle as pk
import numpy as np
import matplotlib
import matplotlib.pyplot as pt
import scipy.sparse as sp
from check_span_rule import *

np.set_printoptions(linewidth=400)

do_find = True
do_show = True

solver = sys.argv[1]
N = int(sys.argv[2])

with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

# sort ascending by constraints to find smallest tree
sorter = np.argsort(B.sum(axis=1))
Y = Y[sorter]
B = B[sorter]
W = W[sorter]

if do_find:

    found_infeasible = False

    # for d2 in range(1, len(Y)):
    #     for d1 in range(d2):
    for d2 in range(500, 505):
        for d1 in range(210, 220):
            sample = [d1, d2]
    
            print(f"region sample of {len(Y)}", sample)
            result = check_span_rule(X, Y[sample], B[sample], W[sample], solver, verbose=False)
            status, u, g, D, E = result
    
            if status != "optimal":
                found_infeasible = True
                fname = f"infeasible_pair_{solver}_{N}.pkl"
                with open(fname, 'wb') as f: pk.dump((result, sample), f)
    
            if found_infeasible: break
        if found_infeasible: break

    if not found_infeasible:
        print("All sub-samples feasible")

if do_show:

    fname = f"infeasible_pair_{solver}_{N}.pkl"
    if not os.path.exists(fname):
        print("All sub-samples feasible")
        sys.exit(1)

    with open(fname, 'rb') as f: result, sample = pk.load(f)

    print("sample:", sample)
    print("sorter[sample]:", sorter[sample])

    Bu = B[sample].any(axis=0)

    ### zeroing in on submatrices
    B = B[sample][:, Bu]
    XT = X.T[:, Bu]
    Y = Y[sample][:, Bu]
    W = W[sample]

    print("B:")
    print(B.astype(int))
    print("X.T:")
    print(XT)
    print("Y:")
    print(Y)
    print("W:")
    print(W)


