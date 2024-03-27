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
stop_after_first = False

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

    num_samples = 0
    infeasible_samples = []

    for d2 in range(1, len(Y)):
        for d1 in range(d2):
            num_samples += 1
            if d2 < 503: continue # restart after crash

            sample = [d1, d2]

            print(f"checked {num_samples} samples of {len(Y)*(len(Y)-1)//2}, {len(infeasible_samples)} infeasible, sample", sample)
            result = check_span_rule(X, Y[sample], B[sample], W[sample], solver, verbose=False)
            status, u, g, D, E = result
    
            if status != "optimal":
                infeasible_samples.append(sample)
                fname = f"infeasible_pairs_{solver}_{N}.pkl"
                with open(fname, 'wb') as f: pk.dump((result, infeasible_samples, num_samples), f)

            if len(infeasible_samples) > 0 and stop_after_first: break
        if len(infeasible_samples) > 0 and stop_after_first: break

    if len(infeasible_samples) > 0:
        print(f"{len(infeasible_samples)} of {num_samples} infeasible sub-samples found")
    else:
        print("All sub-samples feasible")

if do_show:

    fname = f"infeasible_pairs_{solver}_{N}.pkl"
    if not os.path.exists(fname):
        print("All sub-samples feasible")
        sys.exit(1)

    with open(fname, 'rb') as f: result, infeasible_samples, num_samples = pk.load(f)
    print(f"{len(infeasible_samples)} of {num_samples} infeasible sub-samples found")

    sample = infeasible_samples[0]
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


