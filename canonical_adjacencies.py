"""
Lists adjacencies (i,j,k) between canonical regions i and j
Non-symmetric, only stores one of (i,j,k) and (j,i,k)
Stores the one with positive dot at the destination: (i,j,k) where Y[j,k] > 0
"""
import sys
import itertools as it
import pickle as pk
import numpy as np

if __name__ == "__main__":

    do_solve = True
    solver = sys.argv[1]
    N = int(sys.argv[2])
    ε = 1

    # load canonical regions
    with np.load(f"regions_{N}_{solver}.npz") as regions:
        X, Y, B, W = (regions[key] for key in ("XYBW"))

    if do_solve:
        
        # extract adjacencies between them
        A = set()
        for (i,j) in it.combinations(range(len(Y)), r=2):
            print(f"pair {i},{j} of {len(Y)} regions")

            # get all disagreements between region pair
            k = np.flatnonzero(Y[i] != Y[j])

            # skip pairs with more than one disagreement, not adjacent
            if len(k) != 1: continue

            # store adjacency with the index of the single disagreement
            k = k[0]            
            if Y[j,k] > 0:
                A.add((i,j,k))
            else:
                A.add((j,i,k))

        # save sorted list of adjacencies
        A = sorted(A)
        with open(f"canon_adjacencies_{N}.pkl","wb") as f: pk.dump(A, f)

    # show adjacencies
    with open(f"canon_adjacencies_{N}.pkl","rb") as f: A = pk.load(f)
    
    print("i,j,k, W[i], W[j], Y[i,k], X[k], Y[j,k], W[j] - W[i]")
    for (i,j,k) in sorted(A):
        print(i,j,k, W[i], W[j], Y[i,k], X[k], Y[j,k], (W[j] - W[i]).round(1))
    
