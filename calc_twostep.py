from scipy.special import comb
import pickle as pk
import numpy as np
import itertools as it

N = 4
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

def leading_hidden(kidx, vidx, hidx, solns, justone=False):
    sat = check(kidx[:len(hidx)], vidx[:len(hidx)], hidx)
    if sat == False: return False
    if len(hidx) < len(kidx):
        any_h = False
        for h in range(2**N):
            any_h = any_h or leading_hidden(kidx, vidx, hidx + [h], solns, justone)
            if justone and any_h: break
        if not any_h: return False
    else: # sat == True and len(hidx) == len(kidx)
        solns[kidx][vidx].append(sat)
    return True

def check_shatter(kidx, solns, justone=False):
    for v,vidx in enumerate(it.product(kidx, repeat=len(kidx))):
        print(kidx, v, vidx)
        solns[kidx][vidx] = []
        if not leading_hidden(kidx, vidx, [], solns, justone): return False
    return True

# justone doesn't work as expected:
# even though it terminates current hidx[depth] early, a different hidx[depth-1] is still tried with the same vidx[depth-1:depth+1]
# would fix this if all v in kidx loops happened shallower than all h in range(2**N) loops
# but then it would not be able to abort unshatterable kidxs early
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

# # this might fix justone
# # but doesn't work because different vidx[depth] will require different hidx[depth]
# def leading_shatter(kidx, vidx, hidx, solns, justone=False):
#     sat = check(kidx[:len(vidx)], vidx, hidx)
#     if sat == False: return False
#     if len(vidx) < len(kidx):
#         for h in range(2**N):
#             all_v = True
#             for v in kidx:
#                 all_v = all_v and leading_shatter(kidx, vidx + [v], hidx + [h], solns, justone)
#                 if not all_v: break
#             if all_v and justone: break
#     else: # sat == True and len(vidx) == len(kidx)
#         if tuple(vidx) not in solns[kidx]: solns[kidx][tuple(vidx)] = []
#         solns[kidx][tuple(vidx)].append(sat)
#     return True

# M = N
M = 7
kidxs = []
solns = {}
for k,kidx in enumerate(it.combinations(range(2**N), M)):
    # kidx is unshatterable if it contains both +/- v
    if any((2**N - j - 1) in kidx for j in kidx): continue
    solns[kidx] = {}
    # shattered = leading_shatter(kidx, [], [], solns, justone=False)
    shattered = check_shatter(kidx, solns, justone=True)
    print(f"{k} of {comb(V.shape[1], M, exact=True)} kidxs: {kidx} shattered={shattered}...")
    if shattered: break # only find one shattered kidx

print("shattered kidxs:")
num_shattered = 0
largest_solns = 0
for kidx in sorted(solns.keys()):
    if len(solns[kidx]) != M**M: continue
    num_shattered += 1
    for vidx in sorted(solns[kidx].keys()):
        # print(kidx, vidx, f"{len(solns[kidx][vidx])} solns")
        largest_solns = max(largest_solns, len(solns[kidx][vidx]))
        for m1, m2 in solns[kidx][vidx]:
            W1 = np.concatenate([weights[i] for i in m1], axis=0)
            W2 = np.concatenate([weights[i] for i in m2], axis=0)
            Y = np.sign(W2 @ np.sign(W1 @ V[:,kidx]))
            assert np.allclose(Y, V[:,vidx])
    print(kidx, " shatterable")
# print(f"{num_shattered} of {comb(V.shape[1], M, exact=True)} shattered kidxs")
print(f"{num_shattered} of {len(solns)} shattered kidxs")
print(f"{largest_solns} solns at most")
