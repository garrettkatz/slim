import os
import numpy as np
from scipy.special import comb
import itertools as it
from sign_solve import solve

N = 3
V = np.array(tuple(it.product((-1, 1), repeat=N))).T

# backtrack perceptron
def leading_solve(V, out, feas_outs, Ws):
    status, W = solve(V[:,:len(out)], np.array([out]))
    if status == False: return 0
    if len(out) == 2**(N-1): # and status == True here # other half determined by W(-v) = -Wv
        tuo = [-s for s in reversed(out)]
        feas_outs.append(out + tuo)
        Ws.append(W)
        return 1
    feas = 0
    for s in (-1, 1):
        feas_s = leading_solve(V, out + [s], feas_outs, Ws)
        if feas_s > 0: print(f"  {feas_s} feas in", out + [s])
        feas += feas_s
    return feas

feas = 0
feas_outs, Ws = [], []
for s in (-1,1):
    feas += leading_solve(V, [s], feas_outs, Ws)
    print(s, feas)
feas_outs = np.array(feas_outs)
print(f"{feas} of {2**(2**N)}")
print(f"{feas**N} of {(2**N)**(2**N)}")
print(feas_outs.shape)
print(feas_outs) # (num dichotomies, num vertices = 2**N)

for W, out in zip(Ws, feas_outs):
    assert (np.sign(W @ V) == out).all()

# num_feas**N possible output arrays
nxts = []
last_first = 0
for hemis in it.product(range(feas_outs.shape[0]), repeat=N):
    if hemis[0] != last_first:
        print(f"hemi {hemis[0]} of {feas_outs.shape[0]}...")
        last_first = hemis[0]
    
    # memory intensive, build idxs in here?
    nxts.append(np.array([feas_outs[h] for h in hemis]))
nxts = np.array(nxts)
print(nxts.shape) # num multichotomies, N, 2**N
idxs = ((nxts > 0) * 2**np.arange(N-1,-1,-1).reshape(1, N, 1)).sum(axis=1).astype(int)
print(idxs.shape) # num multichotomies, num vertices = 2**N
print(idxs)

fname = f"twostep_{N}.npy"
if os.path.exists(fname):
    twostep = np.load(fname)
else:
    # num_multi**2 possible 2-step nxts (maybe with duplicates)
    twostep = set()
    for m0 in range(idxs.shape[0]):
        print(f" multichotomy {m0} of {idxs.shape[0]}")
        i1 = idxs[m0]
        for m1 in range(idxs.shape[0]):
            twostep.add(tuple(idxs[m1][i1]))

    twostep = np.array([i for i in twostep])
    np.save(fname, twostep)

print(twostep.shape) # num multis, 2**N
print(idxs.shape[0]**2)

# brute check all keysets with size M = N+1 for shattering of twostep
# probably don't need to check symmetric M-sets (those related by an invertible NxN matrix)

# M = N
M = 2
num_shatters = 0
for k,keys in enumerate(map(list, it.combinations(range(2**N), M))):
    print(f"{k} of {int(comb(2**N, M))}: {keys}")
    feas_maps = set()
    keyset = set(keys)
    for idx in twostep:
        vals = idx[keys]
        if set(vals) == keyset:
            feas_maps.add(tuple(vals))
    # input(feas_maps)
    if len(feas_maps) == M**M:
        num_shatters += 1
print(f"{num_shatters} keysets shatter")

