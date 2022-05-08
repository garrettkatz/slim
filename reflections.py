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

S = np.array([
    [1, 1,  1,  1, 1],
    [1, 0, -1,  0, 1],
    [0, 1,  0, -1, 0],
])

fig = pt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot( S[0], S[1], S[2], 'ko-')
ax.plot(-S[0], S[1], S[2], 'ko-')
ax.plot( S[1], S[2], S[0], 'ko-')
ax.plot( S[1], S[2],-S[0], 'ko-')
ax.plot( S[2], S[0], S[1],'ko-')
ax.plot( S[2],-S[0], S[1],'ko-')
ax.plot([0, 1], [0, 1], [0, 1], 'r-')
# ax.plot([0, 3**-.5], [0, 3**-.5], [0, 3**-.5], 'r-')
ax.plot([0, 0], [0, 0], [0, 1], 'g-')
ax.plot([-1, 1], [-1, 1], [1, -1], 'b-')
ax.plot([0, 0.5], [0, 0.5], [0, 1], 'm-')
pt.show()

