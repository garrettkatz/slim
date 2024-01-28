import sys, os
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
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

# sorter = np.argsort(-B.sum(axis=1))
sorter = np.argsort(B.sum(axis=1)) # sequential infeasible at sample [0, 1, ..., 503]
Y = Y[sorter]
B = B[sorter]
W = W[sorter]

print('infeas', sorter[215], sorter[503])

if do_exp:

    found_infeasible = False

    # # sequential
    # for num_regions in range(2, len(Y)):
    #     sample = np.arange(num_regions)

    #     print(f"region sample of {len(Y)}", sample)
    #     result = check_span_rule(X, Y[sample], B[sample], W[sample], solver, verbose=False)
    #     status, u, g, D, E = result

    #     if status != "optimal":
    #         found_infeasible = True
    #         fname = f"minimal_infeasible_sequential_{solver}_{N}.pkl"
    #         with open(fname, 'wb') as f: pk.dump((result, sample), f)

    #     if found_infeasible: break

    # combos
    hi = 504 # from sequential
    for num_regions in range(2, hi):
        for sample in map(list, it.combinations(range(hi), num_regions)):

            # sample = np.arange(hi) # what sequential found
            sample = [215, 503] # what combos found.  GUROBI and SCIPY agree, even with reoptimize=True

            # sample = np.random.choice(np.arange(504, len(Y)), 500, replace=False)
            # sample[:2] = [215, 503]

            print(f"region sample of {len(Y)}", sample)
            result = check_span_rule(X, Y[sample], B[sample], W[sample], solver="ECOS", verbose=False)
            status, u, g, D, E = result

            if status != "optimal":
                print(f"status={status}")
                found_infeasible = True
                fname = f"minimal_infeasible_combos_{solver}_{N}.pkl"
                with open(fname, 'wb') as f: pk.dump((result, sample), f)

            if found_infeasible: break
        if found_infeasible: break

    if not found_infeasible:
        print("All sub-samples feasible")

if do_show:

    fname = f"minimal_infeasible_combos_{solver}_{N}.pkl"
    if not os.path.exists(fname):
        print("All sub-samples feasible")
        sys.exit(1)

    with open(fname, 'rb') as f: result, sample = pk.load(f)

    print("sample:", sample)
    print("sorter[sample]:", sorter[sample])

    Bu = B[sample].any(axis=0)

    pt.subplot(3,1,1)
    pt.imshow(B[sample][:,Bu])
    pt.title("Boundaries")
    pt.ylabel("i")
    # pt.axis("off")

    pt.subplot(3,1,2)
    pt.imshow(X.T[:, Bu])
    pt.title("Vertices")
    # pt.axis("off")

    pt.subplot(3,1,3)
    pt.imshow(Y[sample][:, Bu])
    pt.ylabel("i")
    pt.xlabel("k")
    pt.title("Dichotomies")
    # pt.axis("off")

    print("B:")
    print(B[sample][:,Bu])
    print("X.T:")
    print(X.T[:, Bu])
    print("Y:")
    print(Y[sample][:, Bu])

    pt.show()

