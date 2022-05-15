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

fig = pt.figure()
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
ax.plot([0, w_r[0]], [0, w_r[1]], [0, w_r[2]], 'o-')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
pt.show()

