import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt

N = 3 # number of neurons
M = 4 # number of key-value pairs
ortho = False
save_depth = 1

# N = 4 # number of neurons
# M = 4 # number of key-value pairs
# ortho = True
# save_depth = 1

orth = "orth" if ortho else ""

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, H = pk.load(f)
print("H.shape:", H.shape) # (num_dichotomies, num vertices = 2**N)
# weights = np.array(weights).round() # empirically always looks like integer-valued.  somehow important for checkfit to work properly! roundoff errors?
weights = np.concatenate(weights, axis=0).round() # empirically always looks like integer-valued.  somehow important for checkfit to work properly! roundoff errors?


# load all in memory, not good for big N
kidx = tuple(range(M))
if N == 4 and M == 4 and ortho: kidx = (0, 3, 5, 6)

solns = {}
for lvidx in it.product(kidx, repeat=save_depth):
    vlead = "_".join(map(str, lvidx))
    fname = f"solns/N{N}M{M}{orth}_{vlead}"
    with open(fname, "rb") as f:
        solns.update(pk.load(f))

V = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(V)
print(V[:,kidx])

print(len(solns))
print(M**M)

E = M**(M+2)
print(f"{E} edges...")

vidx = kidx
s = 0

pt.ion()
fig, axs = pt.subplots(nrows=N, ncols=4, sharex='col')

while True:

    hidx, m1, m2 = solns[vidx][s]
    for l, (idx,mm) in enumerate(zip((kidx, hidx), (m1, m2))):
        for i in range(N):
            axs[i,2*l].imshow(weights[mm[i]])
            axs[i,2*l].set_xticks(range(N))
            axs[i,2*l].set_yticks(range(len(mm[i])))
            axs[i,2*l].set_ylabel(f"i={i}")

            axs[i,2*l+1].imshow(H[mm[i], idx].reshape(-1, M))
            axs[i,2*l+1].set_xticks(range(M))
            axs[i,2*l+1].set_yticks(range(len(mm[i])))
            axs[i,2*l+1].set_ylabel(f"i={i}")

        axs[-1,2*l].set_xlabel("j")
        axs[-1,2*l+1].set_xlabel("m")

    print(f"vidx: {vidx}, soln {s} of {len(solns[vidx])}")
    cmd = input('[n]ext soln, [p]rev soln')
    if cmd == 'n': s = (s+1) % len(solns[vidx])
    if cmd == 'p': s = (s-1) % len(solns[vidx])

