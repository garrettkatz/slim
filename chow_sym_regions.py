import sys
import itertools as it
import numpy as np
import cvxpy as cp

if __name__ == "__main__":

    do_solve = True
    solver = sys.argv[1]
    N = int(sys.argv[2])
    ε = 1

    if do_solve:

        # load potentially non-chow-symmetric regions
        with np.load(f"regions_{N}_{solver}.npz") as regions:
            X, Y, B, W = (regions[key] for key in ("XYBW"))

        # and their boundary points        
        B = np.load(f"boundaries_{N}_{solver}.npy")
        
        # shifted/scaled chow parameters
        C = Y @ X
        
        # solve the regions with chow symmetry constraints
        for i, y in enumerate(Y):
            print(f"{i} of {len(Y)}")
        
            # symmetric weights
            u = cp.Variable(N)

            # fit the boundary points
            data_constraints = [u @ (X[B[i]].T * y[B[i]]) >= ε]

            # stay canonical
            canonical_constraints = [u[-1] >= 0, u[:-1] >= u[1:]]

            # enforce chow symmetries
            chow_constraints = []
            n = np.flatnonzero(C[i,:-1] == C[i,1:])
            if len(n) > 0: chow_constraints.append(u[n] == u[n+1])
            if C[i,-1] == 0: chow_constraints.append(u[-1] == 0)

            # solve the linear program            
            constraints = data_constraints + canonical_constraints + chow_constraints
            objective = cp.Minimize(cp.sum(cp.multiply(u,C[i]))) # chow params are also net slack
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=solver, verbose=False)

            # make sure no false negatives
            assert problem.status == "optimal"

            # record result
            W[i] = u.value

        # save chow-symmetric weights
        np.savez(f"chow_regions_{N}_{solver}.npz", X=X, Y=Y, B=B, W=W, C=C)

    with np.load(f"chow_regions_{N}_{solver}.npz") as regions:
        X, Y, B, W, C = (regions[key] for key in ("XYBWC"))
    
    print("C")
    print(C)
    print("W")
    print(W)
    
    
