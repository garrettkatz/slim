import pickle as pk
import random as rd
import itertools as it
import numpy as np
import scipy.sparse as sp
import scipy.optimize as so
from nondet_sized import NonDeterminator

# rd.seed(20000)

N = 3
M = 4

kidx = tuple(range(M))

# build first M hypercube vertices
# C = np.empty((N, M), dtype=int)
# for m,v in enumerate(it.product((-1, 1), repeat=N)):
#     if m == M: break
#     C[:,m] = v

C = np.array(tuple(it.product((-1, +1), repeat=N))).T

with open(f"hemis_{N}.npy", "rb") as f: _, hemis = pk.load(f)
chots = np.unique(hemis[:,kidx], axis=0)

with open(f"vmap_{N}_{M}.pkl", "rb") as f: vidx_map = pk.load(f)

num_paths = 2
path_length = 3 # number of vidx nodes in path

# print("Sizing")
# print(f"V: num_paths*path_length*3*N*M = {num_paths}*{path_length}*3*{N}*{M} = {num_paths*path_length*3*N*M}")

# def one_to_many(inp, out):
#     mapping = {}
#     for x, y in zip(inp, out):
#         if mapping.get(x, y) != y: return True
#         mapping[x] = y
#     return False

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

# Layer activations
X = np.empty((num_paths, path_length, N, M), dtype=int) # forward input layer activities
H = np.empty((num_paths, path_length, N, M), dtype=int) # forward hidden layer activities
V = np.empty((num_paths, path_length, N, M), dtype=int) # forward output layer activities
Y = np.empty((num_paths, path_length-1, N), dtype=int) # backward output activity
Z = np.empty((num_paths, path_length-1, N), dtype=int) # backward hidden activity
U = np.empty((num_paths, path_length-1, N), dtype=int) # backward input activity

nd = NonDeterminator()
def combind():

    pn = []

    for p, (nodes, edges) in enumerate(paths):

        for n, vidx in enumerate(nodes):

            pn.append((p, n))

            # forward

            # set input activities
            X[p, n] = C[:, kidx]

            # choose hidden activities
            rows = ()
            for i in range(N):
                row = nd.choice(vidx_map[vidx, rows])
                rows += (row,)
                H[p, n, i] = chots[row]

            # set output activities
            V[p, n] = C[:, vidx]

            # backward
            if n < len(edges):
                _, k = edges[n]
    
                # set output activity
                Y[p, n] = C[:, kidx[k]]
    
                # choose hidden activity
                Z[p, n] = C[:, nd.choice(range(2**N))]
    
                # choose input activity
                U[p, n] = C[:, nd.choice(range(2**N))]

            # linprog
            if nd.depth < len(nd.counters): continue # don't redo linprog on same choices from last call
            # print(nd.depth, "linprogs")

            if n == 0: continue # collect at least one transition in each path before linprog

            forward = (X, H, V)
            backward = (U, Z, Y)

            # forward constraints
            for layer in range(2):
                fi, fo = forward[layer], forward[layer+1]
                bi, bo = backward[layer], backward[layer+1]
                for i in range(N):
                    blocks = [ -(fi[_p, _n] * fo[_p, _n, i]).T for (_p, _n) in pn] + [np.zeros((0, N*4))] # 4*N term columns
                    A_ub = sp.block_diag(blocks)
                    b_ub = -np.ones(A_ub.shape[0])
                    c = -A_ub.mean(axis=0)
    
                    blocks = [[None for _ in range(len(pn) + 4)] for  _ in range(len(pn))]
                    b_eq = np.zeros(len(pn) * N)
                    for b, (_p, _n) in enumerate(pn):
                        blocks[b][b] = sp.eye(N)
                        if _n == 0:
                            b_eq[b*N + i] = 1
                        else:
                            _, edges_p = paths[_p]
                            j,_ = edges_p[_n-1]
                            blocks[b][b-1] = -sp.eye(N)
                            blocks[b][-1] = sp.diags(fo[_p,_n-1,i,j] * fi[_p,_n-1,:,j])
                            blocks[b][-2] = sp.diags(fo[_p,_n-1,i,j] * bi[_p,_n-1])
                            blocks[b][-3] = sp.diags(bo[_p,_n-1,i] * fi[_p,_n-1,:,j])
                            blocks[b][-4] = sp.diags(bo[_p,_n-1,i] * bi[_p,_n-1])
                    A_eq = sp.bmat(blocks)
    
                    res = so.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(None, None))
                    if not res.success:
                        # print(p, n, i, f"fwd layer {layer} fail")
                        return False
                # print(p, n, f"fwd layer {layer} success")

            if n == 1: continue # collect at least two transitions before backward linprog

            # backward constraints
            for layer in range(2):
                fi, fo = forward[layer], forward[layer+1]
                bi, bo = backward[layer], backward[layer+1]
                for i in range(N):
                    blocks = [ -(bo[_p, _n] * bi[_p, _n, i]).reshape(1,-1) for (_p, _n) in pn if _n < path_length-1] + [np.zeros((0, N*4))] # 4*N term columns
                    A_ub = sp.block_diag(blocks)
                    b_ub = -np.ones(A_ub.shape[0])
                    c = -A_ub.mean(axis=0)
                    
                    nb = len(pn) - (p+1) # omit last node of every path
                    blocks = [[None for _ in range(nb + 4)] for  _ in range(nb)]
                    b_eq = np.zeros(nb * N)
                    b = 0
                    for (_p, _n) in pn:
                        if _n == path_length -1: continue
                        blocks[b][b] = sp.eye(N)
                        if _n == 0:
                            b_eq[b*N + i] = 1
                        else:
                            _, edges_p = paths[_p]
                            j,_ = edges_p[_n-1]
                            blocks[b][b-1] = -sp.eye(N)
                            blocks[b][-1] = sp.diags(fi[_p,_n-1,i,j] * fo[_p,_n-1,:,j])
                            blocks[b][-2] = sp.diags(fi[_p,_n-1,i,j] * bo[_p,_n-1])
                            blocks[b][-3] = sp.diags(bi[_p,_n-1,i] * fo[_p,_n-1,:,j])
                            blocks[b][-4] = sp.diags(bi[_p,_n-1,i] * bo[_p,_n-1])
                        b += 1
                    A_eq = sp.bmat(blocks)

                    res = so.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(None, None))
                    if not res.success:
                        # print(p, n, i, f"bkwd layer {layer} fail")
                        return False
                # print(p, n, f"bkwd layer {layer} success")

    return True


for i,success in enumerate(nd.runs(combind)):
    # if i % 1000 == 0: print(i, nd.counter_string())
    print(i, success, nd.counter_string())
    if success: break
    # if i == 300: break

