import numpy as np
import itertools as it
from sign_solve import solve

N = 2
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
for hemis in it.product(range(feas_outs.shape[0]), repeat=N):
    nxts.append(np.array([feas_outs[h] for h in hemis]))
nxts = np.array(nxts)
print(nxts.shape) # num multichotomies, N, 2**N
idxs = ((nxts > 0) * 2**np.arange(N-1,-1,-1).reshape(1, N, 1)).sum(axis=1).astype(int)
print(idxs.shape) # num multichotomies, num vertices = 2**N
print(idxs)

# # num_multi**2 possible 2-step nxts (maybe with duplicates)
# idxs_2 = set()
# for m0 in range(idxs.shape[0]):
#     print(f" multichotomy {m0} of {idxs.shape[0]}")
#     i1 = idxs[m0]
#     for m1 in range(idxs.shape[0]):
#         idxs_2.add(tuple(idxs[m1][i1]))

# idxs_2 = np.array([i for i in idxs_2])
# np.save("idxs_2.npy", idxs_2)

idxs_2 = np.load("idxs_2.npy")
print(idxs_2.shape)
print(idxs.shape[0]**2)

# brute check all keysets with size M = N+1 for shattering of idxs_2
# probably don't need to check symmetric M-sets (those related by an invertible NxN matrix)

