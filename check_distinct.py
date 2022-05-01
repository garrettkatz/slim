# Can every vidx be solved while always keeping different columns of H distinct?

import pickle as pk
import itertools as it
import numpy as np

N = 3
M = 4

with open(f"hemis_{N}.npy","rb") as f: _, hemis = pk.load(f)
chots = np.unique(hemis[:,:M], axis=0)

with open(f"vmap_{N}_{M}.pkl", "rb") as f: vidx_map, _ = pk.load(f)

soln_counts = {}
anti_counts = {}
for vidx in it.product(range(M), repeat=M):
    soln_counts[vidx] = 0
    anti_counts[vidx] = 0
    for key in vidx_map[vidx]:
        if len(key) != N-1: continue
        for last in vidx_map[vidx][key]:

            rows = key + (last,)
            H = np.array([chots[r] for r in rows])
            hidx = tuple(2**np.arange(N-1,-1,-1) @ (H > 0))

            if len(hidx) == len(set(hidx)):
                soln_counts[vidx] += 1

                antipodes = set([2**N - h for h in hidx])
                if len(antipodes & set(hidx)) == 0:
                    anti_counts[vidx] += 1

    print(vidx, soln_counts[vidx], anti_counts[vidx])
print("restricted to distinct hidx columns:")
print(f"{sum(soln_counts.values())} distinct solns across all vidx, {min(soln_counts.values())}-{max(soln_counts.values())}")

print("restricted to distinct hidx columns and no antipodes:")
print(f"{sum(anti_counts.values())} distinct solns across all vidx, {min(anti_counts.values())}-{max(anti_counts.values())}")

