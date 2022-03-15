import os
import numpy as np
import pickle as pk
from scipy.special import comb
import itertools as it
from sign_solve import solve

np.set_printoptions(threshold=np.inf)

N = 6
V = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(V.shape) # (num neurons N, num verticies 2**N)

fname = f"hemis_{N}.npy"
if os.path.exists(fname):
    with open(fname,"rb") as f: weights, hemis = pk.load(f)
else:

    # backtrack recursion
    def leading_solve(V, out, hemis, weights):
        status, w = solve(V[:,:len(out)], np.array([out]))
        if status == False: return 0
        if len(out) == 2**(N-1): # and status == True here # other half determined by w(-v) = -wv
            tuo = [-s for s in reversed(out)]
            hemis.append(out + tuo)
            weights.append(w)
            return 1
        feas = 0
        for s in (-1, 1):
            feas_s = leading_solve(V, out + [s], hemis, weights)
            if feas_s > 0: print(f"  {feas_s} feas in", out + [s])
            feas += feas_s
        return feas
    
    feas = 0
    hemis, weights = [], []
    for s in (-1,1):
        feas += leading_solve(V, [s], hemis, weights)
        print(s, feas)
    hemis = np.array(hemis)
    print(f"{feas} of {2**(2**N)}")
    print(f"{feas**N} of {(2**N)**(2**N)}")
    for w, out in zip(weights, hemis):
        assert (np.sign(w @ V) == out).all()

    with open(fname,"wb") as f: pk.dump((weights, hemis), f)

print(hemis.shape)
print(hemis) # (num_dichotomies, num vertices = 2**N)

# all columns have both signs?
print("cols have both signs:")
print((hemis < 0).any(axis=0) & (hemis > 0).any(axis=0))

# all rows have both signs?
# hemichotomies necessarily have an equal number of +/-1
print("rows have both signs (count of each):")
print((hemis < 0).any(axis=1) & (hemis > 0).any(axis=1))
print((hemis < 0).sum(axis=1))
print((hemis > 0).sum(axis=1))

import matplotlib.pyplot as pt
pt.subplot(1,3,1)
pt.imshow(hemis)
pt.subplot(1,3,2)
pt.imshow(np.concatenate(weights, axis=0))
pt.subplot(1,3,3)
pt.plot(np.unique(weights))
pt.show()

# # exp case M ~ 2**(N-1), first kidx: is every binary integer 0 ... M-1 found in the rows of H[:,kidx]?
# # although first kidx does not shatter N4M8
# print(np.unique(hemis[:,:2**(N-1)]))

