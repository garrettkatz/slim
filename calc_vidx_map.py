import os
import random
import pickle as pk
import numpy as np
import itertools as it
from nondet_sized import NonDeterminator

N = 2
M = 2

C = np.array(tuple(it.product((-1,+1), repeat=N))).T

with open(f"hemis_{N}.npy","rb") as f: _, hemis = pk.load(f)
chots = np.unique(hemis[:,:M], axis=0)

if os.path.exists(f"vmap_{N}_{M}.pkl"):
    with open(f"vmap_{N}_{M}.pkl","rb") as f: vidx_map = pk.load(f)
else:

    vidx_map = {}
    
    nd = NonDeterminator()
    def fill_map():
        rows = [nd.choice(range(chots.shape[0])) for i in range(N)]
        H = np.array([chots[r] for r in rows])
        hidx = tuple(2**np.arange(N-1,-1,-1) @ (H > 0))
    
        hchots = np.unique(hemis[:,hidx], axis=0)
        V = np.array([hchots[nd.choice(range(hchots.shape[0]))] for i in range(N)])
        vidx = tuple(2**np.arange(N-1,-1,-1) @ (V > 0))
    
        if max(vidx) >= M: return
    
        for i in range(N):
            lookup = (vidx, tuple(rows[:i]))
            if lookup not in vidx_map: vidx_map[lookup] = []
            vidx_map[lookup].append(rows[i])
    
    for i,_ in enumerate(nd.runs(fill_map)):
        if i % 1000 == 0: print(i, nd.counter_string())

    with open(f"vmap_{N}_{M}.pkl","wb") as f: pk.dump(vidx_map, f)

print(f"{len(vidx_map)} lookups")

# check
for rep in range(10):
    vidx = tuple(random.choices(range(M), k=M))
    rows = ()
    while len(rows) < N:
        rows += (random.choice(vidx_map[vidx, rows]),)
    H = np.array([chots[r] for r in rows])
    hidx = tuple(2**np.arange(N-1,-1,-1) @ (H > 0))
    
    for i in range(N):
        assert (C[i,vidx] == hemis[:, hidx]).all(axis=1).any()
    
