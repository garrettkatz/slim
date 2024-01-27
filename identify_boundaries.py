import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from check_separability import check_separability_pooled

if __name__ == "__main__":

    do_bounds = True
    solver = sys.argv[1]
    N = int(sys.argv[2])
    ε = 1
    canonical = False # neighbors may not be canonical

    if do_bounds:

        # load region data
        with np.load(f"regions_{N}_{solver}.npz") as regions:
            X, Y, B, W = (regions[key] for key in ("XYBW"))
    
        # process each region
        for i, y in enumerate(Y):
    
            # set up candidate neighbors of y
            Yn = y * (-1)**np.eye(len(y))
    
            # prepare arguments for feasibility checks
            pool_args = [
                (X, yn, ε, canonical, solver, f"{i} of {len(Y)}, {j} of {len(Yn)}")
                for j, yn in enumerate(Yn)]
    
            # multiprocessing version (don't use all cores)
            num_procs = max(1, cpu_count()-1)
            with Pool(num_procs) as pool:
                results = pool.map(check_separability_pooled, pool_args)
    
            # overwrite B with adjacencies
            feasible, _ = map(np.array, zip(*results))
            B[i] = feasible

        np.save(f"boundaries_{N}_{solver}.npy", B)

    B = np.load(f"boundaries_{N}_{solver}.npy")
    nB = B.sum(axis=1)
    print(f"{len(Y)} feasible regions total")
    print(f"{nB.min()} <= ~{nB.mean()} <= {nB.max()} boundaries per region")


