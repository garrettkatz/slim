import os
import random
import pickle as pk
import numpy as np
import itertools as it
from nondet_sized import NonDeterminator

N = 4
M = 5

recalc = True

C = np.array(tuple(it.product((-1,+1), repeat=N))).T

with open(f"hemis_{N}.npy","rb") as f: _, hemis = pk.load(f)
chots = np.unique(hemis[:,:M], axis=0)

if os.path.exists(f"vmap_{N}_{M}.pkl") and not recalc:
    with open(f"vmap_{N}_{M}.pkl","rb") as f: vidx_map, soln_counts = pk.load(f)
else:

    vidx_map = {}
    hchot_memo = {}
    
    nd = NonDeterminator()
    def fill_map():
        rows = [nd.choice(range(chots.shape[0])) for i in range(N)]
        H = np.array([chots[r] for r in rows])
        hidx = tuple(2**np.arange(N-1,-1,-1) @ (H > 0))

        if hidx not in hchot_memo:
            hchot_memo[hidx] = (np.unique(hemis[:,hidx], axis=0) > 0).astype(int)
        hchots = hchot_memo[hidx]

        vidx = np.zeros(M, dtype=int)
        for i in range(N):
            vidx_inc = hchots * 2**(N-i-1) + vidx
            choices = np.flatnonzero((vidx_inc < M).all(axis=1))
            if len(choices) == 0: return
            vidx = vidx_inc[nd.choice(choices)]

        vidx = tuple(vidx)

        if vidx not in vidx_map: vidx_map[vidx] = {}
    
        for i in range(N):
            lookup = tuple(rows[:i])
            if lookup not in vidx_map[vidx]: vidx_map[vidx][lookup] = []
            if rows[i] not in vidx_map[vidx][lookup]: vidx_map[vidx][lookup].append(rows[i])
    
    for i,_ in enumerate(nd.runs(fill_map)):
        if i % 1000 == 0: print(i, nd.counter_string())

    # num solns for each vidx
    soln_counts = {}
    for vidx in it.product(range(M), repeat=M):
        num_solns = sum(len(vidx_map[vidx][key]) for key in vidx_map[vidx] if len(key) == N-1)
        soln_counts[vidx] = num_solns

    with open(f"vmap_{N}_{M}.pkl","wb") as f: pk.dump((vidx_map, soln_counts), f)

print(f"{len(vidx_map)} vidxs = M**M = {M**M}")
print(f"{sum(map(len,vidx_map.values()))} total lookups")

for vidx in it.product(range(M), repeat=M):
    print(vidx, soln_counts[vidx])
print(f"{sum(soln_counts.values())} distinct solns across all vidx, {min(soln_counts.values())}-{max(soln_counts.values())}")

# check
for rep in range(10):
    vidx = tuple(random.choices(range(M), k=M))
    rows = ()
    while len(rows) < N:
        rows += (random.choice(vidx_map[vidx][rows]),)
    H = np.array([chots[r] for r in rows])
    hidx = tuple(2**np.arange(N-1,-1,-1) @ (H > 0))
    
    for i in range(N):
        assert (C[i,vidx] == hemis[:, hidx]).all(axis=1).any()
    
