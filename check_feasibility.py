"""
Feasibility checking for linear separability of training samples
"""
import numpy as np
import cvxpy as cp

def check_feasibility(X, y, ε, canonical, solver, verbose=False):
    """
    X[k]: kth training input
    y[k]: target label assigned to kth input
    ε: margin; (w @ X[k]) * y[k] must be greater than ε for every k
    canonical: True|False flag whether w must be canonical
    solver, verbose: options passed to cvxpy
    """

    # input dimension
    N = X.shape[1]

    # weight variable
    w = cp.Variable(N)

    # dots with each (signed) training sample
    wxy = w @ (X.T * y)

    # linear separability constraint
    constraints = [wxy >= ε]

    # canonical constraints
    if canonical:
        constraints += [
            w[-1] >= 0,      # non-negative
            w[:-1] >= w[1:]] # descending order

    # minimize slack to make problem bounded
    objective = cp.Minimize(cp.sum(wxy))

    # solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    # pack results
    feasible = (problem.status == 'optimal')
    w = w.value if feasible else np.empty(N)
    return feasible, w

def check_feasibility_pooled(args):
    """
    multiprocessing/starmap version, with arguments as tuple
    additional last argument is job tag for progress message
    """

    # unpack arguments
    args, tag = args[:-1], args[-1]

    # check feasibility
    feasible, w = check_feasibility(*args)

    # display progress message
    print(f"{tag} complete: feasible={feasible}")

    # return result
    return feasible, w

