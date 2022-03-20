import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt

N = 2 # number of neurons
M = 2 # number of key-value pairs
save_depth = 0

V = np.array(tuple(it.product((-1, 1), repeat=N))).T

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, hemis = pk.load(f)
weights = np.array(weights).reshape((-1,N))

# load all in memory, not good for big N
kidx = tuple(range(M))
solns = {}
for lvidx in it.product(kidx, repeat=save_depth):
    vlead = "_".join(map(str, lvidx))
    fname = f"solns/N{N}M{M}_{vlead}"
    with open(fname, "rb") as f:
        solns.update(pk.load(f))

vidx = kidx
w10 = np.eye(N)
w20 = np.eye(N)

jk = tuple((j,k) for (j,k) in it.product(range(M), repeat=2) if j != k)
options = tuple(solns[vidx[:j] + (kidx[k],) + vidx[j+1:]] for (j,k) in jk)
for option in it.product(*options):
    for (j,k),t in zip(jk, option):
        x = V[:,vidx[j]].reshape(-1,1)
        y = V[:,kidx[k]].reshape(-1,1)
        h = np.sign(w1 @ x)
        v = np.sign(w2 @ h)
        z = np.sign(w2.T @ y)
        u = np.sign(w1.T @ z)

        # # first attempted LR:
        # θi.shape = (4, N, N) | (4, 1, 1)
        # for p,(a,b) in enumerate(it.product((x, u), (h, z))):
        #     w1 += b * θ1[p] * a.T
        # for p,(a,b) in enumerate(it.product((h, z), (v, y))):
        #     w2 += b * θ2[p] * a.T    
        # return w1, w2


# m1 = [np.isclose(np.eye(N)[i], weights).all(axis=1).argmax() for i in range(N)]
# m2 = list(m1)

# sign solves:

