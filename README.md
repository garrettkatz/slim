This is the code for the paper

Anonymous et al. "On the Feasibility of Single-Pass Full-Capacity Learning in Linear Threshold Neurons with Binary Input Vectors."

This code has been tested on a workstation with 8-core Intel i7 CPU and 32GB of RAM, Fedora 39 Linux, Python 3.11.7, NumPy 1.26.3, SciPy 1.11.1, CVXPY 1.4.1, Gurobi 11.0.0, and MOSEK 10.1.  Some scripts use multiprocessing and will use most of your cores.

The core implementation of the method is in `check_span_rule.py`.  You do not run this script directly; it is used by other scripts.

You can regenerate the results with these steps:

1. Enumerate all the canonical regions for N=3 with `python enumerate_regions.py GUROBI 3`.  To use a different solver, replace `GUROBI` with another CVXPY-recognized solver such as MOSEK, ECOS, CBC, or SCIPY.  To generate for a higher dimension, replace `3` with the dimension you want.  On our workstation, N=8 takes about one minute. N=9 may take several hours.  The results are saved in a file `regions_{solver}_{N}.npz` which is required for subsequent scripts.

1. Identify all the boundary vectors for each canonical region with `python identify_boundaries.py GUROBI 3`.  Again, you can change the solver and input dimemsion.  N=8 takes about 20 minutes; N=9 may take more than one day and 8GB of RAM.  The results are saved in a file `boundaries_{solver}_{N}.npy` which is required for subsequent scripts.

1. Check full-capacity feasibility with `python full_capacity_experiment.py GUROBI 3` or similar.  All dimensions from 3 to the N that you provide will be checked.  Instead of one solver you can also specify multiple solvers in a comma-separated list (no spaces).  N=8 takes a matter of minutes with GUROBI, but mileage may vary with other solvers.  At N=8 scale, a commercial or academic Gurobi license is needed.

1. Check high-capacity feasibility with `python high_capacity_experiment.py GUROBI 3` or similar.  Since this performs many repetitions, N=8 may take over a day.  This will save one results file per repetition.

1. Find a size-2 subset of infeasible dichotomies with `python find_infeasible_pair.py GUROBI 8` or similar.  N=8 or 9 will take a few hours.

1. To visualize the infeasible sub-graph and its certificate, run `python counterexample_fig_rot.py`.

