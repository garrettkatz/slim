import os, sys
import numpy as np
import pickle as pk
from scipy.special import comb
import itertools as it
from sign_solve import solve
import matplotlib.pyplot as pt
import scipy.optimize as so

# np.set_printoptions(sign="+")
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

# N = 3
N = int(sys.argv[1])

X = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(X.shape) # (num neurons N, num verticies 2**N)

# calculate all column perms of X for signed row permutations of neurons
pows = 2**np.arange(N-1,-1,-1)
perms = []
for p in map(list, it.permutations(range(N))):
    for s in it.product((+1,-1), repeat=N):
        perms.append( pows @ ((np.array(s).reshape(N,1) * X[p,:]) > 0) )
print(f"{len(perms)} perms")
print(perms[0]) # identity with p,s ordering


fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, hemis = pk.load(f)

print(hemis.shape)
print(hemis) # (num_dichotomies, num vertices = 2**N)

weights = np.concatenate(weights, axis=0).round().astype(int)
print(weights)

# # only keep one representative from every signed row permutation equivalence class
# keep = np.ones(hemis.shape[0], dtype=bool)
# for m in range(hemis.shape[0]):
#     if not keep[m]: continue
#     for p,perm in enumerate(perms[1:]):
#         mp = (hemis[m+1:] == hemis[m,perm]).all(axis=1).argmax()
#         keep[mp] = False
#         print(f"perming {m},{p} of {hemis.shape[0]},{len(perms)-1}..")
# hemis = hemis[keep,:]
# weights = weights[keep,:]

# only keep one representative from every weight multiset equivalent class
_, uidx, uinv = np.unique(np.sort(np.fabs(weights).astype(int), axis=1), axis=0, return_index=True, return_inverse=True)
uweights = weights[uidx,:]
uhemis = hemis[uidx,:]
print(uweights)
print(uhemis)
print(uhemis.shape)
input('.')

uboundaries = np.zeros((uhemis.shape[0], X.shape[1]), dtype=bool)
dists = np.zeros((uhemis.shape[0], X.shape[1]))
orthcond = np.zeros((uhemis.shape[0], X.shape[1]), dtype=bool)
reflcond = np.zeros((uhemis.shape[0], X.shape[1]), dtype=bool)
spancond = np.zeros((uhemis.shape[0], X.shape[1]), dtype=bool)
rks = np.empty(uhemis.shape[0], dtype=int)

for m in range(uhemis.shape[0]):
    print(f"linproging {m} of {uhemis.shape[0]}")
    pos = list(np.flatnonzero(uhemis[m] > 0))
    for b in range(len(pos)):
        others = pos[:b] + pos[b+1:]
        result = so.linprog(
            c = X[:,others].mean(axis=1),
            A_eq = X[:,pos[b]].reshape(1,-1),
            b_eq = np.zeros(1),
            A_ub = -X[:,others].T,
            b_ub = -np.ones(len(pos)-1),
            bounds=(None, None))
        # check result in case of warnings
        w = result.x
        bcheck = (w @ X[:,others] >= .99).all() and np.fabs(w @ X[:,pos[b]]) <= 0.01
        
        if (result.status in (0, 3)) != bcheck:
            input('unreliable!')

        if bcheck:
            uboundaries[m, pos[b]] = True
            dists[m, pos[b]] = (uweights[m] @ X[:,pos[b]]).round()
            if dists[m, pos[b]] != 1:
                input('wx not 1!')

            # wp = uweights[m] - (uweights[m] * X[:,pos[b]]).sum() * X[:,pos[b]] / N
            wp = N * uweights[m] - (uweights[m] * X[:,pos[b]]).sum() * X[:,pos[b]] # scale to maintain integer values
            orthcond[m,pos[b]] = (wp @ X[:,others] >= .99).all()
            if not orthcond[m,pos[b]]:
                print(uweights[m])
                print(w)
                print(w @ X[:,others])
                print(wp)
                print(wp @ X[:,others])
                input("!")

            wr = N * uweights[m] - 2 * (uweights[m] * X[:,pos[b]]).sum() * X[:,pos[b]] # scale to maintain integer values
            hr = uhemis[m].copy()
            hr[[pos[b], 2**N - 1 - pos[b]]] *= -1
            n = (hr == hemis).all(axis=1).argmax()
            # reflcond[m,pos[b]] = (wr @ X * hr >= .99).all()
            reflcond[m,pos[b]] = (wr @ X * hr >= 0).all()
            # if not reflcond[m,pos[b]]:
            #     print("old h", uhemis[m])
            #     print("new h", hr)
            #     print("wr X ", wr @ X)
            #     print("wr   ", wr)
            #     print("wp   ", wp)
            #     print("new w", weights[n])
            #     print("x fl ", X[:,pos[b]])
            #     # input("!r!")

            # A = np.stack((uweights[m], X[:,pos[b]], np.sign(uweights[m]), np.sign(uweights[m] * X[:,pos[b]]))).T
            A = np.stack((uweights[m], X[:,pos[b]])).T
            coefs = np.linalg.lstsq(A, weights[n], rcond=None)[0]
            ws = A @ coefs
            spancond[m,pos[b]] = (ws.round() == weights[n].round()).all()
            # if not spancond[m,pos[b]]:
            #     print(X[:,pos[b]])
            #     print(uweights[m])
            #     print(weights[n] - ws)
            #     input("!s!")

    # check uniqueness of canonical vector w Xb = 1 (rank condition)
    B = uboundaries[m].sum()
    rks[m] = np.linalg.matrix_rank(np.concatenate((X[:,uboundaries[m]].T, np.ones((B, 1))), axis=1))

print("Max normal dots within region:")
for m in range(uhemis.shape[0]):
    dots = X[:,uboundaries[m]].T @ X[:,uboundaries[m]]
    dots -= N*np.eye(uboundaries[m].sum(), dtype=int) # exclude self-dots
    print(f"{m}: {dots.max()} of {N}")

print("Min > 1 dot ratios:")
for m in range(uhemis.shape[0]):
    numers = N * uweights[m] @ X
    for b,bj in enumerate(np.flatnonzero(uboundaries[m])):
        denoms = X[:,bj] @ X
        sorter = np.argsort(numers / denoms)
        print(f"{m:02},{b:02}: " + ", ".join([f"{n}/{d}" for n,d in zip(numers[sorter], denoms[sorter]) if (n > d > 0)]))

if ((dists == 1) == uboundaries).all():
    print("all boundary distances 1")
else:
    print("not all 1")

print(rks)
if (rks == N).all():
    print("all hemis unique canonical weights")
else:
    print("not all hemis unique canonical weights")

if (orthcond == uboundaries).all():
    print("all boundaries reachable by ortho projection")
else:
    print(uboundaries)
    print(orthcond)
    print("not all reachable by orthoproj")

if (reflcond == uboundaries).all():
    print("all boundary reflections stay in new region (maybe on new boundary)")
else:
    print("some boundary reflections leave new region")

if (spancond == uboundaries).all():
    print("all flipped weights are in span")
else:
    print("some flipped weights are not in span")

# distances to all planes in each region
scdists = uweights @ X
for m in range(len(uweights)):
    pt.plot([m,m],[scdists[m].min(), scdists[m].max()], '-', color=(.6,.6,.6))
    pt.plot([m+.25]*uboundaries[m].sum(), scdists[m][uboundaries[m]], 'k.')
    pt.plot([m]*(1-uboundaries)[m].sum(), scdists[m][~uboundaries[m]], 'b.')
pt.xlabel("region")
pt.ylabel("distances to planes")
pt.show()

rows, cols = (2,1) if N > 10 else (1, 2)
pt.subplot(3,1,1)
pt.imshow(dists.T if N > 10 else dists)
pt.subplot(3,1,2)
pt.imshow(uboundaries.T if N > 10 else uboundaries)
pt.subplot(3,1,3)
pt.imshow((spancond != uboundaries).T if N > 10 else spancond != uboundaries)
pt.show()

