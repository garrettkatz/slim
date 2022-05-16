import os
import numpy as np
import pickle as pk
from scipy.special import comb
import itertools as it
from sign_solve import solve
import matplotlib.pyplot as pt

N = 3
X = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(X.shape) # (num neurons N, num verticies 2**N)

Z = np.array(tuple(it.product((-1,0,1), repeat=N)))
Z = Z[np.fabs(Z).astype(int).sum(axis=1) % 2 == 1, :].T

S = np.array([
    [1, 1,  1,  1, 1],
    [1, 0, -1,  0, 1],
    [0, 1,  0, -1, 0],
])

w = np.array([1,1,1])
x = np.array([1,1,-1])

w_p = w - (w*x).sum() * x / N
w_r = w - 2*(w*x).sum() * x / N


def descend(w_old, x_flip, w0):
    w_seq = [w0]
    for t in range(2*3):
        wt = w_seq[-1]
        cands = wt - np.diag(np.sign(wt))
        devs = cands - w_old
        
        valid = (cands @ x_flip <= -1) & (devs @ w_old >= 1 - np.sum(w_old**2))
        if not valid.any():
            print(f"term at {t / 2}")
            break
    
        c = (cands[valid]**2).sum(axis=1).argmin()
        w_seq.append(cands[c])
    return w_seq

w_feas = N*(N-2)*w - (N-1)*x
w_seq = descend(w, x, w_feas)

w_old = np.array((0,0,1))
w_feas2 = N*(N-2)*w_old + (N-1)*x
w_seq2 = descend(w_old, -x, w_feas2)

# w_old = np.array((0,0,1))
# w_seq2 = [w_feas2]
# for t in range(2*3):
#     wt = w_seq2[-1]
#     cands = wt - np.diag(np.sign(wt))
#     devs = cands - w_old
    
#     valid = (cands @ x <= -1) & (devs @ w_old >= 1 - np.sum(w_old**2))
#     if not valid.any():
#         print(f"term at {t / 2}")
#         print("wt", wt)
#         print("cands:")
#         print(cands)
#         print("x:")
#         print(x)
#         print("cands @ x:")
#         print(cands @ x)
#         print("devs @ w:")
#         print(devs @ w_old)
#         print("lobound:", 1 - np.sum(w_old**2))
#         break

#     c = (cands[valid]**2).sum(axis=1).argmin()
#     w_seq2.append(cands[c])

fig = pt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.plot( Z[0], Z[1], Z[2], 'k+')

ax.plot( S[0], S[1], S[2], 'ko-')
ax.plot(-S[0], S[1], S[2], 'ko-')
ax.plot( S[1], S[2], S[0], 'ko-')
ax.plot( S[1], S[2],-S[0], 'ko-')
ax.plot( S[2], S[0], S[1],'ko-')
ax.plot( S[2],-S[0], S[1],'ko-')

ax.plot([0, w[0]], [0, w[1]], [0, w[2]], 'r-')
# ax.plot([0, 3**-.5], [0, 3**-.5], [0, 3**-.5], 'r-')
ax.plot([0, 0], [0, 0], [0, 1], 'g-')
ax.plot([-x[0], x[0]], [-x[1], x[1]], [-x[2], x[2]], 'b-')
# ax.plot([0, 0.5], [0, 0.5], [0, 1], 'm-')
ax.plot([0, w_p[0]], [0, w_p[1]], [0, w_p[2]], 'm-')
ax.plot([0, w_r[0]], [0, w_r[1]], [0, w_r[2]], 'co')
ax.plot([0, w_feas[0]], [0, w_feas[1]], [0, w_feas[2]], 'c-')
ax.plot([0, w_feas2[0]], [0, w_feas2[1]], [0, w_feas2[2]], 'y-')

ax.plot(*np.stack(w_seq).T, marker='d', color='c', linestyle='--')
ax.plot(*np.stack(w_seq2).T, marker='d', color='y', linestyle='--')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
pt.show()

