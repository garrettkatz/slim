import os
import numpy as np
import pickle as pk
from scipy.special import comb
import itertools as it
from sign_solve import solve
import matplotlib.pyplot as pt

N = 4
X = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(X.shape) # (num neurons N, num verticies 2**N)

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, hemis = pk.load(f)

print(hemis.shape)
print(hemis) # (num_dichotomies, num vertices = 2**N)

weights = np.concatenate(weights, axis=0).round().astype(int)
print(weights)

# project weights onto vertices
dists = weights @ X / N**0.5
scdists = weights @ X
boundaries = np.zeros(dists.shape, dtype=bool)

for m,w in enumerate(weights):
    print(f"{m} of {len(weights)}")
    # wX = w @ X
    wX = hemis[m]

    # # wp = w - ((w @ X) * X).T / N
    # # wpX = wp @ X
    # wpX = wX - (w @ X) * (X.T @ X) / N

    # # boundaries[m] = (wpX * wX >= 0).all(axis=1) & (wX >= 0) & ((wp**2).sum(axis=1) > 0)
    # boundaries[m] = (wpX * wX >= 0).all(axis=1) & (wX >= 0)
    # if w.min() == w.max():
    #     boundaries[m][(w * X.T > 0).all(axis=1)] = False

    for j in range(X.shape[1]):
        if (w * X[:,j] > 0).all() and w.min() == w.max(): continue
        if wX[j] < 0: continue

        # wp = w - ((w @ X[:,j]) * X[:,j]).T / N
        # wpX = wp @ X
        wpX = wX - (w @ X[:,j]) * ((X[:,j].T @ X) / N) # more numerically stable somehow?

        # if wpX[j] < 0: input('..')
        if wpX[j] < 0: wpX[j] = 0

        # boundaries[m,j] = (wpX * wX >= 0).all() & (wX[j] > 0) & ((wp**2).sum() > 0)
        boundaries[m,j] = (wpX * wX >= 0).all()

    # try to solve equal dot of w with every boundary
    # w = np.linalg.lstsw(X[:,boundaries[m]].T, 

for m in range(len(weights)):
    pt.plot([m]*boundaries[m].sum(), scdists[m][boundaries[m]], 'ko')
    # pt.plot([m], [len(np.unique(scdists[m][boundaries[m]]))], 'ko')
    # if len(np.unique(dists[m][boundaries[m]])) == 0:
    #     print(boundaries[m])
    #     print(dists[m])
pt.xlabel("region")
pt.ylabel("distances to boundary planes")
# pt.ylabel("num distinct distances to boundary planes")
pt.show()

# strata over all regions/planes
pt.scatter(boundaries.flatten(), scdists.flatten())
# pt.scatter(boundaries.flatten(), dists.flatten())
pt.xlabel("x plane boundary of w region")
pt.ylabel("distance from w to x plane")
pt.show()

# dist/boundary heatmaps
# pt.subplot(3,1,1)
# pt.imshow(dists.T)
# pt.subplot(3,1,2)
# pt.imshow(boundaries.T)
# pt.subplot(3,1,3)
# pt.imshow(hemis.T)
# pt.show()
pt.subplot(1,4,1)
pt.imshow(dists)
pt.subplot(1,4,2)
pt.imshow(boundaries)
pt.title("Boundaries")
pt.xlabel("Vertex plane")
pt.ylabel("Weight region (hemi)")
pt.subplot(1,4,3)
pt.imshow(hemis)
pt.title("Hemis")
pt.xlabel("Vertex x")
pt.ylabel("s(wx)")
pt.subplot(1,4,4)
pt.imshow(weights)
pt.title("Weights")
pt.xlabel("i")
pt.ylabel("hemi")
pt.tight_layout()
pt.show()

