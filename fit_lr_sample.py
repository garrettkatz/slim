import pickle as pk
import itertools as it
import numpy as np

N = 2 # number of neurons
M = 2 # number of key-value pairs
save_depth = 0

T = 10 # sequence length
num_iters = 1

V = np.array(tuple(it.product((-1, 1), repeat=N))).T

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, hemis = pk.load(f)
weights = np.array(weights)

# load all in memory, not good for big N
kidx = tuple(range(M))
solns = {}
for lvidx in it.product(kidx, repeat=save_depth):
    vlead = "_".join(map(str, lvidx))
    fname = f"solns/N{N}M{M}_{vlead}"
    with open(fname, "rb") as f:
        solns.update(pk.load(f))

print(len(solns))
print(M**M)

for itr in range(num_iters):
    
    vidx = kidx
    m1 = [np.isclose(np.eye(N)[i], weights).all(axis=1).argmax() for i in range(N)]
    m2 = list(m1)
    print(m1)
        
    for t in range(T):
        j, k = np.random.choice(M, size=2)
        new_vidx = vidx[:j] + (k,) + vidx[j+1:]
        _, mm1, mm2 = solns[new_vidx]

        x = V[:,kidx[j]].reshape((N,1))
        h = np.sign(weights[m1] @ x)
        v = np.sign(weights[m2] @ h)
        y = V[:, kidx[k]].reshape((N,1))

        w1, w2 = lr_fn(weights[m1], weights[m2], x, h, v, y)
        new_m1 = [mm1[((w1[i] - weights[mm1[i]])**2).sum(axis=1).argmin()] for i in range(N)]
        new_m2 = [mm2[((w2[i] - weights[mm2[i]])**2).sum(axis=1).argmin()] for i in range(N)]
        dist = ((w1 - weights[new_m1])**2).sum() + ((w2 - weights[new_m2])**2).sum()

        # backprop/interpolate solve

        vidx = new_vidx    
    



