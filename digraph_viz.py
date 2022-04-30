import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from nondet_sized import NonDeterminator

N = 2 # number of neurons
M = 2 # number of key-value pairs

with open(f"hemis_{N}.npy", "rb") as f: weights, hemis = pk.load(f)
print("hemis.shape:", hemis.shape) # (num_dichotomies, num vertices = 2**N)
weights = np.concatenate(weights, axis=0).round() # empirically always looks like integer-valued.  somehow important for checkfit to work properly! roundoff errors?
chots, uidx = np.unique(hemis[:,:M], axis=0, return_index=True)
whits = weights[uidx]

with open(f"vmap_{N}_{M}.pkl","rb") as f: vidx_map, soln_counts = pk.load(f)

kidx = tuple(range(M))

C = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(C)
print(C[:,kidx])

E = M**(M+2)
print(f"{E} edges...")

pt.ion()
# fig, axs = pt.subplots(nrows=N, ncols=4, sharex='col')
fig, axs = pt.subplots(nrows=1, ncols=2)

vidx = kidx

while True:

    V = C[:,vidx]

    nd = NonDeterminator()
    def soln_looper():
        rows = ()
        for i in range(N):
            rows += (nd.choice(vidx_map[vidx][rows]),)
        return rows

    for r,rows in enumerate(nd.runs(soln_looper)):
        H = np.array([chots[r] for r in rows])
        hidx = tuple(2**np.arange(N-1,-1,-1) @ (H > 0))
        hchots, uidx = np.unique(hemis[:,hidx], axis=0, return_index=True)
        hwhits = weights[uidx]
        
        outs = [(hchots == V[i]).all(axis=1).argmax() for i in range(N)]

        KWH = np.nan * np.ones((N, M + 1 + N + 1 + M))
        KWH[:,:M] = C[:,kidx]
        for i in range(N):
            KWH[i,M+1:M+N+1] = whits[rows[i]]
            KWH[i,-M:] = chots[rows[i]]

        WVV = np.nan * np.ones((N, N + 1 + M + 1 + M))
        WVV[:,-N:] = C[:,vidx]
        for i in range(N):
            WVV[i,:N] = hwhits[outs[i]]
            WVV[i,N+1:N+M+1] = hchots[outs[i]]

        axs[0].clear()
        axs[1].clear()

        axs[0].imshow(KWH)
        axs[0].set_title("K, W0, H")

        for (i,j) in it.product(range(N), repeat=2):
            axs[0].text(M+j+.75,i+.25,int(KWH[i,M+1+j]), color='red')

        axs[1].imshow(WVV)
        axs[1].set_title("W1, V, V")

        for (i,j) in it.product(range(N), repeat=2):
            axs[1].text(j-.25,i+.25,int(WVV[i,j]), color='red')

        print(f"vidx: {vidx}, soln {r} of {soln_counts[vidx]}")
        cmd = input('next [s]oln, next [v]idx: ')
        cmd = 's'
        # pt.pause(0.1)

        if cmd == 's': continue
        if cmd == 'v': break

    j = int(input(f"Key to remap (0-{M-1}): ")) % M
    k = int(input(f"New value (0-{M-1}): ")) % M

    vidx = vidx[:j] + (k,) + vidx[j+1:]

