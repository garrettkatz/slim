import sys
import itertools as it
import pickle as pk
import numpy as np
import matplotlib
import matplotlib.pyplot as pt
from check_span_rule import *

do_exp = True
do_show = True

region_sampling = 5
num_reps = 20

solver = sys.argv[1]
N = int(sys.argv[2])

with np.load(f"regions_{N}_{solver}.npz") as regions:
    X, Y, W = (regions[key] for key in ("XYW"))
B = np.load(f"boundaries_{N}_{solver}.npy")

# num_region_samples = np.linspace(Y.shape[0]//2, Y.shape[0], region_sampling+1)[:-1].astype(int)
num_region_samples = (np.linspace(.1, .9, region_sampling) * len(Y)).astype(int)
print(num_region_samples)
input('.')

if do_exp:

    for (nr, rep) in it.product(range(region_sampling), range(num_reps)):
        num_regions = num_region_samples[nr]

        # sub-sample the regions
        sample = np.random.choice(len(Y), num_regions, replace=False)

        print(f"Running N={N}, {num_regions} regions ({nr} of {region_sampling}), rep={rep} of {num_reps}...")
        result = check_span_rule(X, Y[sample], B[sample], W[sample], solver, verbose=True)
        status, u, g, D, E = result

        print(f"N={N}, {num_regions} regions ({len(D)} nodes), rep {rep}: {status}")

        if (status == "optimal") and (117 in sample) and (1688 in sample):
            print(sample)
            print("What happened??")
            input('.')

        fname = f"high_cap_{solver}_{N}_{num_regions}_{rep}.pkl"
        with open(fname, 'wb') as f: pk.dump((result, sample), f)

if do_show:

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 12

    pt.figure(figsize=(7,3))

    feasibility_rate = []
    for num_regions in num_region_samples:

        num_feas = 0
        for rep in range(num_reps):

            fname = f"high_cap_{solver}_{N}_{num_regions}_{rep}.pkl"
            with open(fname, 'rb') as f: result, sample = pk.load(f)
            status, u, g, D, E = result

            if status == 'optimal': num_feas += 1

            print(f"N={N}, {num_regions} regions ({len(D)} nodes), rep {rep}: {status}")

        feasibility_rate.append( num_feas / num_reps )

    pt.plot(100 * num_region_samples // len(Y), feasibility_rate, 'ko-')
    pt.xlabel("Dichotomy sample size (%)")
    pt.ylabel("Feasibility Rate")
    pt.tight_layout()
    pt.savefig('highcap.pdf')
    pt.show()


