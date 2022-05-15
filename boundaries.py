import os, sys
import numpy as np
import pickle as pk
from scipy.special import comb
import itertools as it
from sign_solve import solve
import matplotlib.pyplot as pt
import scipy.optimize as so
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=so.OptimizeWarning)

# np.set_printoptions(sign="+")
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

# N = 3
N = int(sys.argv[1])

X = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(X.shape) # (num neurons N, num verticies 2**N)

# fname = f"hemis_{N}.npy"
# with open(fname,"rb") as f: weights, hemis = pk.load(f)
# weights = np.concatenate(weights, axis=0).round().astype(int)

npz = np.load(f"hemitree_{N}.npz")
weights, hemis = npz["weights"], npz["hemis"]

print(hemis.shape)
print(hemis) # (num_dichotomies, num vertices = 2**N)
print(weights)
print("w sums", weights.sum(axis=1))
print(((weights.sum(axis=1) % 2) == 1).all())
input("all odd ^^..")

# # calculate all column perms of X for signed row permutations of neurons
# pows = 2**np.arange(N-1,-1,-1)
# perms = []
# for p in map(list, it.permutations(range(N))):
#     for s in it.product((+1,-1), repeat=N):
#         perms.append( pows @ ((np.array(s).reshape(N,1) * X[p,:]) > 0) )
# print(f"{len(perms)} perms")
# print(perms[0]) # identity with p,s ordering

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
reprs, uidx, uinv = np.unique(np.sort(np.fabs(weights).astype(int), axis=1), axis=0, return_index=True, return_inverse=True)
uweights = weights[uidx,:]
uhemis = hemis[uidx,:]
print(uweights)
print(uhemis)
print(uhemis.shape)
print(uidx)

reprs = set(map(tuple, reprs))

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

            # # lookup when using hemis_{N}.npy
            # n = (hr == hemis).all(axis=1).argmax()
            # wn = weights[n]
            # recalculate wn online when using reduced hemitree data
            result = so.linprog(
                # c = X @ hr, # numerically ill-behaved
                # A_ub = -(X * hr).T,
                # b_ub = -np.ones(2**N),
                # bounds = (None, None),
                c = X[:,:2**(N-1)] @ hr[:2**(N-1)],
                A_ub = -(X * hr)[:,:2**(N-1)].T,
                b_ub = -np.ones(2**(N-1)),
                bounds = (None, None),
            )
            wn = result.x
            print("lookup flip region: flipj, hr, c, wn, wn round")
            print(pos[b])
            print(hr)
            print(X @ hr)
            print(wn)
            assert (np.sign(wn @ X) == hr).all()
            wn = result.x.round().astype(int)
            print(wn)
            assert (np.sign(wn @ X) == hr).all()
            assert tuple(np.sort(np.fabs(wn))) in reprs

            # A = np.stack((uweights[m], X[:,pos[b]], np.sign(uweights[m]), np.sign(uweights[m] * X[:,pos[b]]))).T
            A = np.stack((uweights[m], X[:,pos[b]])).T
            coefs = np.linalg.lstsq(A, wn, rcond=None)[0]
            ws = A @ coefs
            spancond[m,pos[b]] = (ws.round() == wn.round()).all()
            if not spancond[m,pos[b]]:
                print("out of span:")
                print("w, wX, X")
                print(uweights[m])
                print(np.arange(2**(N-1)) % 10)
                print(uweights[m] @ X[:,:2**(N-1)])
                print(X[:,:2**(N-1)])
                print("boundary")
                print((np.fabs(uweights[m] @ X[:,:2**(N-1)]) == 1).astype(int))
                # print((uboundaries[m] + uboundaries[m,::-1]).astype(int))
                # print(uboundaries[m].astype(int))
                print(f"flip {pos[b]}: x*")
                print(X[:,pos[b]])
                print("wf, wfX")
                print(wn)
                print(np.arange(2**(N-1)) % 10)
                print(wn @ X[:,:2**(N-1)])
                print("residual")
                print(wn - ws)
                # input("!s!")

            # rounding from w eps
            we = w - X[:,pos[b]]*(N-1)/(N-2)/N
            assert (np.sign(we @ X) == hr).all()
            wflo = np.floor(we).astype(int)
            wcei = np.ceil(we).astype(int)
            wrou = np.array(tuple(it.product(*np.stack((wflo, wcei)).T)))
            wrou = wrou[np.fabs(wrou).sum(axis=1) % 2 == 1]
            ham = np.fabs(wrou - we).sum(axis=1)
            # ham = ((wrou - we)**2).sum(axis=1)
            opt = ham.argmin()
            if True: #(ham == ham[opt]).sum() > 1 or not (wrou[opt] == wn).all():
                print("canonical wm:")
                print(uweights[m])
                print("flip x*:")
                print(X[:,pos[b]])
                print("non-canonical w eps:")
                print(we)
                print("neighbors:")
                print(wrou.T)
                print("hams:")
                print(ham)
                print("dots with x:")
                print(X[:,pos[b]] @ wrou.T)
                print("nearest odd ZN:")
                print(wrou[opt])
                print("canonical wm:")
                print(uweights[m])
                print("flip x*:")
                print(X[:,pos[b]])
                print("canonical wn:")
                print(wn)
                print("match:", (wrou[opt] == wn).all())
                input("!e!")

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

