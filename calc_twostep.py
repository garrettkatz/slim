from scipy.special import comb
import pickle as pk
import numpy as np
import itertools as it

N = 3
V = np.array(tuple(it.product((-1, 1), repeat=N))).T
print("V.shape:", V.shape) # (num neurons N, num verticies 2**N)

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, hemis = pk.load(f)
print("hemis.shape:", hemis.shape) # (num_dichotomies, num vertices = 2**N)

def check(kidx, vidx, hidx):
    if len(vidx) == 0: return True
    m1, m2 = [], []
    for n in range(N):
        matches = (V[n,hidx] == hemis[:,kidx]).all(axis=1)
        if not matches.any(): return False
        m1.append(matches.argmax())

        matches = (V[n,vidx] == hemis[:,hidx]).all(axis=1)
        if not matches.any(): return False
        m2.append(matches.argmax())
    return m1, m2

def leading_shatter(kidx, vidx, hidx, solns, justone=False):
    sat = check(kidx[:len(vidx)], vidx, hidx)
    if sat == False: return False
    if len(vidx) < len(kidx):
        for v in kidx:
            any_h = False
            for h in range(2**N):
                any_h = any_h or leading_shatter(kidx, vidx + [v], hidx + [h], solns, justone)
                if justone and any_h: break
            if not any_h: return False
    else: # sat == True and len(vidx) == len(kidx)
        if tuple(vidx) not in solns[kidx]: solns[kidx][tuple(vidx)] = []
        solns[kidx][tuple(vidx)].append(sat)
    return True

# M = N
M = 4
kidxs = []
solns = {}
for k,kidx in enumerate(it.combinations(range(2**N), M)):
    solns[kidx] = {}
    shattered = leading_shatter(kidx, [], [], solns, justone=True)
    print(f"{k} of {comb(V.shape[1], M, exact=True)} kidxs: shattered={shattered}...")

print("shattered kidxs:")
num_shattered = 0
for kidx in sorted(solns.keys()):
    if len(solns[kidx]) != M**M: continue
    num_shattered += 1
    for vidx in sorted(solns[kidx].keys()):
        print(kidx, vidx, f"{len(solns[kidx][vidx])} solns")
        for m1, m2 in solns[kidx][vidx]:
            W1 = np.concatenate([weights[i] for i in m1], axis=0)
            W2 = np.concatenate([weights[i] for i in m2], axis=0)
            Y = np.sign(W2 @ np.sign(W1 @ V[:,kidx]))
            assert np.allclose(Y, V[:,vidx])
print(f"{num_shattered} of {comb(V.shape[1], M, exact=True)} shattered kidxs")

