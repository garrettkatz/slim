import pickle as pk
import sys
import numpy as np
import cvxpy as cp
import load_ltm_data as ld

# @profile
def main():

    do_opt = True

    # input dimension for optimization
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 8

    # solver = 'GLPK'
    solver = 'SCIPY'
    do_ab = True

    # load canonical regions and adjacencies
    Yc, _, X, Ac = ld.load_ltm_data(N)
    A = ld.organize_by_source(Ac)
    with open(f"adjs_{N}_c.npz", "rb") as f: (Yn, _) = pk.load(f)

    # number of regions
    R = Yc.shape[0]

    # set up boundary indices to remove redundant region constraints
    K = {}
    for i in Yn: K[i] = (Yc[i] != Yn[i]).argmax(axis=1)

    if do_opt:

        # weight variables for each region
        w = cp.Variable((R, N))
    
        # set up constraints and objective
        objective_vector = np.empty((R, N))
        region_constraints = []
        ab_constraints = []
        for i in range(R):
    
            # current region variable
            wiT = w[i:i+1]
    
            # absorb y into constraint normals
            Xy = X * Yc[i]
    
            # w is in its region (irredundant boundaries only)
            region_constraints.append( wiT @ Xy[:, K[i]] >= 1 )
    
            # minimize dot with boundary normal average
            objective_vector[i] = Xy[:, K[i]].mean(axis=1)
    
            # necessary conditions for alpha, beta
            for (j, k) in A[i]:
    
                # project other boundaries
                xb = Xy[:, k:k+1]
                xp = Xy[:, [kp for kp in K[i] if kp != k]]
                PXy = xp - xb @ (xb.T @ xp) / N
    
                # necessary constraint
                ab_constraints.append( wiT @ PXy >= 0 )
    
        if do_ab:
            constraints = region_constraints + ab_constraints
        else:
            constraints = region_constraints
    
        # minimize weight to stay bounded
        objective = cp.Minimize(cp.sum(cp.multiply(w, objective_vector)))

        # do the optimization
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver, verbose=True)
    
        fname = f"canon_lprog_ab_necessary_{N}_{solver}{'_ab' if do_ab else ''}.npy"
        np.save(fname, w.value)
    
        print(w.value.round(3))
        print(problem.status)
        print(problem.value)

    else:

        W_lp = np.load(f"canon_lprog_ab_necessary_{N}_{solver}.npy")
        W_ab = np.load(f"canon_lprog_ab_necessary_{N}_{solver}_ab.npy")

        # check region constraints
        for i in range(R): assert (W_ab[i] @ (X * Yc[i]) >= 1 - 1e-7).all()
    
        print('max abs diff', np.fabs(W_lp - W_ab).max())
        diff_rows = ~np.isclose(W_lp, W_ab).all(axis=1)
        print('num rows with diff', diff_rows.sum())
        if diff_rows.sum() > 0:

            print('lp')
            print(W_lp[diff_rows][:10].round(3))
            print('ab')
            print(W_ab[diff_rows][:10].round(3))


if __name__ == "__main__": main()

