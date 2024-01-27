import sys
import itertools as it
import pickle as pk
import numpy as np
import matplotlib
import matplotlib.pyplot as pt
from ab_necessary_lp_gen import do_lp

do_exp = False
do_show = True

N_max = int(sys.argv[1])

eps = 1
Ns = np.arange(3, N_max+1)
num_regions = np.inf # no sub-sampling
shuffle = False
# solvers = ('GLPK', 'SCIPY', 'CBC', 'ECOS')
solvers = ("GUROBI", "MOSEK")
verbose = True

if do_exp:
    
    for N, solver in it.product(Ns, solvers): 

        print(f"Running N={N}, solver={solver}...")
        result = do_lp(eps, N, num_regions, shuffle, solver, verbose)

        fname = f"full_cap_{N}_{solver}.pkl"
        with open(fname, 'wb') as f: pk.dump(result, f)

if do_show:

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 12

    pt.figure(figsize=(3,3))

    opt_times = {}
    coefs = []
    markers = 'ox+^'
    for solver, marker in zip(solvers, markers):

        opt_times[solver] = []
        for N in Ns:

            fname = f"full_cap_{N}_{solver}.pkl"
            with open(fname, 'rb') as f: result = pk.load(f)
            status, w, β, subset, num_nodes, opt_time, = result

            opt_times[solver].append(opt_time)
            if solver == "GUROBI": coefs.append(β)

            print(f"N={N}, solver={solver}: {status} in {opt_time/60}min")

        pt.plot(Ns, opt_times[solver], 'k-'+marker, label=solver)

    pt.xlabel("N")
    pt.ylabel("Run time (s)")
    pt.yscale('log')
    pt.xticks(Ns, list(map(str, Ns)))
    pt.legend()
    pt.tight_layout()
    pt.savefig('fullcap.pdf')
    pt.show()

    pt.figure(figsize=(3,3))
    for N, coef in zip(Ns, coefs):
        if coef is None: continue
        pt.plot([N]*len(coef), np.fabs(coef), 'k.')
    pt.xlabel("N")
    pt.xticks(Ns, list(map(str, Ns)))
    pt.ylabel("$|\\gamma|$", rotation=0)
    pt.tight_layout()
    pt.savefig('gammas.pdf')
    pt.show()



