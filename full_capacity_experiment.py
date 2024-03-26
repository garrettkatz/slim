import sys
from time import perf_counter
import itertools as it
import pickle as pk
import numpy as np
import matplotlib
import matplotlib.pyplot as pt
from check_span_rule import *

do_exp = True
do_show = True

solvers = sys.argv[1].split(",")
N_max = int(sys.argv[2])
Ns = np.arange(3, N_max+1)
verbose = True

if do_exp:

    for N, solver in it.product(Ns, solvers):

        # load region data
        with np.load(f"regions_{N}_{solver}.npz") as regions:
            X, Y, W = (regions[key] for key in ("XYW"))
        B = np.load(f"boundaries_{N}_{solver}.npy")

        print(f"Running N={N}, solver={solver}...")
        start = perf_counter()
        result = check_span_rule(X, Y, B, W, solver, verbose=True)
        run_time = perf_counter() - start

        # check feasibility
        status, u, g, D, E = result
        if status == "optimal":
            for e, (n, p, x, _) in enumerate(E):
                Xn, yn, _ = D[n]
                assert (np.sign(u[n] @ Xn.T) == yn).all()
                assert np.allclose(u[n], u[p] + g[e] * x)

        fname = f"full_cap_{N}_{solver}.pkl"
        with open(fname, 'wb') as f: pk.dump((result, run_time), f)

if do_show:

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 12

    pt.figure(figsize=(3,3))

    run_times = {}
    coefs = []
    markers = 'ox+^'
    for solver, marker in zip(solvers, markers):

        run_times[solver] = []
        for N in Ns:

            fname = f"full_cap_{N}_{solver}.pkl"
            with open(fname, 'rb') as f: result, run_time = pk.load(f)
            status, u, g, D, E = result

            run_times[solver].append(run_time)
            if solver == solvers[0]: coefs.append(g)

            print(f"N={N}, solver={solver}: {status} in {run_time/60}min")
            print(f"{len(g)} tree nodes total, {N+len(g)} irredundant variables")
            print(f"{sum(len(Dn[0]) for Dn in D)} inequalities total")

        pt.plot(Ns, run_times[solver], 'k-'+marker, label=solver)

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
    pt.xlim(Ns[0]-.25, Ns[-1]+.25)
    pt.ylabel("$|\\gamma|$", rotation=0)
    pt.tight_layout()
    pt.savefig('gammas.pdf')
    pt.show()



