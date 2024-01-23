import sys
import itertools as it
import pickle as pk
import numpy as np
import matplotlib
import matplotlib.pyplot as pt
from ab_necessary_lp_gen import do_lp
from load_ltm_data import *

do_exp = True
do_show = True

region_sampling = 10
num_reps = 30

eps = 1
N = int(sys.argv[1])
shuffle = False
solver = 'ECOS'
verbose = True

Y, _, _, _ = load_ltm_data(N)
R = Y.shape[0]
# num_region_samples = R - np.geomspace(1, Y.shape[0]-1, region_sampling).astype(int)
num_region_samples = np.linspace(Y.shape[0]//2, Y.shape[0]-1, region_sampling).astype(int)
print(num_region_samples)
input('.')

if do_exp:
    
    for (nr, rep) in it.product(range(region_sampling), range(num_reps)):
        num_regions = num_region_samples[nr]

        print(f"Running N={N}, {num_regions} regions ({nr} of {region_sampling}), rep={rep} of {num_reps}...")
        result = do_lp(eps, N, num_regions, shuffle, solver, verbose)

        status, w, β, subset, num_nodes, opt_time, = result
        print(f"N={N}, {num_regions} regions ({num_nodes} nodes), rep {rep}: {status} in {opt_time/60}min")

        fname = f"high_cap_{N}_{num_regions}_{rep}.pkl"
        with open(fname, 'wb') as f: pk.dump(result, f)

if do_show:

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 12

    pt.figure(figsize=(7,3))

    feasibility_rate = []
    for num_regions in num_region_samples:

        num_feas = 0
        for rep in range(num_reps):

            fname = f"high_cap_{N}_{num_regions}_{rep}.pkl"
            with open(fname, 'rb') as f: result = pk.load(f)
            status, w, β, subset, num_nodes, opt_time, = result

            if status == 'optimal': num_feas += 1

            print(f"N={N}, {num_regions} regions ({num_nodes} nodes), rep {rep}: {status} in {opt_time/60}min")

        feasibility_rate.append( num_feas / num_reps )

    pt.plot(num_region_samples / R, feasibility_rate, 'ko-')
    pt.xlabel("Fraction of Dichotomies")
    pt.ylabel("Feasibility Rate")
    pt.tight_layout()
    pt.savefig('highcap.pdf')
    pt.show()


