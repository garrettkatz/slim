import os
import numpy as np
import pickle as pk
from scipy.special import comb
import itertools as it
from sign_solve import solve
import matplotlib.pyplot as pt

# np.set_printoptions(sign="+")
np.set_printoptions(formatter={"int": lambda x: "%+d" % x})

N = 5
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
        # if wX[j] < 0: continue # antipodal planes

        # wp = w - ((w @ X[:,j]) * X[:,j]).T / N
        # wpX = wp @ X
        wpX = wX - (w @ X[:,j]) * ((X[:,j].T @ X) / N) # more numerically stable somehow?

        # if wpX[j] < 0: input('..')
        if wpX[j] < 0: wpX[j] = 0

        # boundaries[m,j] = (wpX * wX >= 0).all() & (wX[j] > 0) & ((wp**2).sum() > 0)
        boundaries[m,j] = (wpX * wX >= 0).all()

    # try to solve equal dot of w with every boundary
    # w = np.linalg.lstsw(X[:,boundaries[m]].T, 

# # distances to boundaries
# for m in range(len(weights)):
#     pt.plot([m]*boundaries[m].sum(), scdists[m][boundaries[m]], 'ko')
#     # pt.plot([m], [len(np.unique(scdists[m][boundaries[m]]))], 'ko')
#     # if len(np.unique(dists[m][boundaries[m]])) == 0:
#     #     print(boundaries[m])
#     #     print(dists[m])
# pt.xlabel("region")
# pt.ylabel("distances to boundary planes")
# # pt.ylabel("num distinct distances to boundary planes")
# pt.show()

# # strata over all regions/planes
# pt.scatter(boundaries.flatten(), scdists.flatten())
# # pt.scatter(boundaries.flatten(), dists.flatten())
# pt.xlabel("x plane boundary of w region")
# pt.ylabel("distance from w to x plane")
# pt.show()

# is every bit flip between feasible hemi regions also a flip over a boundary plane?
feasflip = np.zeros(boundaries.shape, dtype=bool)
for m in range(hemis.shape[0]): 

    for b in range(2**(N-1)): # only first half for fixed bias
        flip = hemis[m, :2**(N-1)].copy()
        flip[b] *= -1
        if not (flip == hemis[:, :2**(N-1)]).all(axis=1).any():
            # print(flip)
            # print(hemis[:, :2**(N-1)])
            # print(flip == hemis[:, :2**(N-1)])
            # input('.')
            continue # flip to infeasible hemi
        feasflip[m,b] = True
        # if not (boundaries[m,b] or boundaries[m, 2**N - 1 - b]): print(f"{m,b}: feas flip not across boundary")
        if not boundaries[m,b]: print(f"{m,b}: feas flip not across boundary")

if (feasflip == boundaries)[:, :2**(N-1)].all():
    print("ALL feasible flips are across boundaries")
else:
    print("Some feasible flips are NOT across boundaries")

# distribution of boundary plane counts
bcounts = boundaries.sum(axis=1) // 2 # halve for antipodes
classes = {}
for m,count in enumerate(bcounts):
    if count not in classes: classes[count] = set()
    wclass = tuple(np.sort(np.fabs(weights[m])))
    classes[count].add(wclass)

print("\nregion boundary counts and weight magnitudes")
for count in sorted(classes.keys()):
    print(count, classes[count])

# flip transitions
for m in range(hemis.shape[0]):
    for b in np.flatnonzero(boundaries[m,:2**(N-1)]):
        flip = hemis[m].copy()
        flip[[b, 2**N - 1 - b]] *= -1
        n = (flip == hemis).all(axis=1).argmax()
        s = f"{m: 4d},{b: 4d}: w{weights[m]} x{X[:,b]} = {hemis[m,b]:+d} |{bcounts[m]: 4d}| -> {n: 4d}: w{weights[n]} |{bcounts[n]: 4d}|"
        input(s)


# pt.plot(bcounts)
# pt.show()


# dist/boundary heatmaps
tp = (N > 4) #False
rows,cols = (6, 1) if tp else (1, 6)
lab1, lab2 = (pt.ylabel, pt.xlabel) if tp else (pt.xlabel, pt.ylabel)
sp = 0

pt.subplot(rows, cols, sp := sp + 1)
pt.imshow(dists.T if tp else dists)

pt.subplot(rows, cols, sp := sp + 1)
pt.imshow(boundaries.T if tp else boundaries)
pt.title("Boundaries")
lab1("Vertex plane")
lab2("Weight region (hemi)")

pt.subplot(rows, cols, sp := sp + 1)
pt.imshow(feasflip.T if tp else feasflip)
pt.title("Feasible bit flips")
lab1("Vertex plane")
lab2("Weight region (hemi)")

pt.subplot(rows, cols, sp := sp + 1)
pt.imshow(hemis.T if tp else hemis)
pt.title("Hemis")
lab1("Vertex x")
lab2("s(wx)")

pt.subplot(rows, cols, sp := sp + 1)
pt.imshow(weights.T if tp else weights)
pt.title("Weights")
lab1("i")
lab2("hemi")

pt.subplot(rows, cols, sp := sp + 1)
bcounts = bcounts.reshape(-1,1)
pt.imshow(bcounts.T if tp else bcounts)
pt.title("Boundary counts")
lab2("hemi")

pt.tight_layout()
pt.show()

