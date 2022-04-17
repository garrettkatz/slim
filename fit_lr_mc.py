import pickle as pk
import random as rd
import itertools as it
import numpy as np
from nondet_sized import NonDeterminator

N = 2
M = 2

kidx = tuple(range(M))

# build first M hypercube vertices
C = np.empty((N, M), dtype=int)
for m,v in enumerate(it.product((-1, 1), repeat=N)):
    if m == M: break
    C[:,m] = v

with open(f"hemis_{N}.npy", "rb") as f: _, H = pk.load(f)
H_uni = np.unique(H[:,kidx], axis=0)

num_paths = 2
path_length = 2 # number of vidx nodes in path

print("Sizing")
print(f"V: num_paths*path_length*3*N*M = {num_paths}*{path_length}*3*{N}*{M} = {num_paths*path_length*3*N*M}")

def one_to_many(inp, out):
    mapping = {}
    for x, y in zip(inp, out):
        if mapping.get(x, y) != y: return True
        mapping[x] = y
    return False

# mc sample paths to a given depth
paths = []
for p in range(num_paths):
    vidx = kidx
    nodes = [vidx]
    edges = []
    for t in range(1,path_length):
        j, k = rd.choices(range(M), k=2) # new_vidx[j] <- kidx[k]
        vidx = vidx[:j] + (kidx[k],) + vidx[j+1:]
        edges.append((j, k))
        nodes.append(vidx)
    paths.append((nodes, edges))
        
# search combinatorial variable bindings
nd = NonDeterminator()
def combind():

    # Choose layer activations
    V = np.empty((num_paths, path_length, 3, N, M), dtype=int) # 3 = 1 hidden layer + 2 backward layers
    for p, (nodes, edges) in enumerate(paths):
        for n, vidx in enumerate(nodes):

            # choose hemichotomies for forward hidden layer
            for i in range(N):
                r = nd.choice(range(H_uni.shape[0]))
                V[p, n, 0, i] = H_uni[r]

            # abort if hidx -> vidx is one-to-many
            hidx = 2**np.arange(N-1,-1,-1) @ V[p, n, 0]
            if one_to_many(hidx, vidx): return False

            # abort if any hidx -> vidx row is a non-feasible hemichotomy
            for i in range(N):
                if not (H[:,hidx] == C[i,vidx]).all(axis=1).any(): return False

            # choose hemichotomies for backward hidden layer
            H_uni_vidx = np.unique(H_uni[:,vidx], axis=0)
            for i in range(N):
                r = nd.choice(range(H_uni_vidx.shape[0]))
                V[p, n, 1, i] = H_uni_vidx[r]

            # choose hemichotomies for backward input layer
            bidx = 2**np.arange(N-1,-1,-1) @ V[p, n, 1]
            H_uni_bidx = np.unique(H[:, bidx], axis=0)
            for i in range(N):
                r = nd.choice(range(H_uni_bidx.shape[0]))
                V[p, n, 2, i] = H_uni_bidx[r]

    # Form linprog data
    # TODO
    
# for _ in nd.runs(combind):
#     print(nd.counter_string())

