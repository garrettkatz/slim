import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import scipy.optimize as so

# N = 2 # number of neurons
# M = 2 # number of key-value pairs
# save_depth = 0

N = 3 # number of neurons
M = 4 # number of key-value pairs
save_depth = 1

V = np.array(tuple(it.product((-1, 1), repeat=N))).T

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, hemis = pk.load(f)
weights = np.array(weights).reshape((-1,N))

# load all in memory, not good for big N
kidx = tuple(range(M))
solns = {}
for lvidx in it.product(kidx, repeat=save_depth):
    vlead = "_".join(map(str, lvidx))
    fname = f"solns/N{N}M{M}_{vlead}"
    with open(fname, "rb") as f:
        solns.update(pk.load(f))

vidx = kidx
w1 = np.eye(N)
w2 = np.eye(N)

jk = tuple((j,k) for (j,k) in it.product(range(M), repeat=2) if j != k)
options = tuple(solns[vidx[:j] + (kidx[k],) + vidx[j+1:]] for (j,k) in jk)
numopt = 1
for opt in options: numopt *= len(opt)
for lp,option in enumerate(it.product(*options)):

    # init linprog mats
    A_ub = ()
    b_ub = ()

    for (j,k), (hidx, m1, m2) in zip(jk, option):
        
        x = V[:,kidx[j]].reshape(-1,1)
        y = V[:,kidx[k]].reshape(-1,1)
        h = np.sign(w1 @ x)
        v = np.sign(w2 @ h)
        z = np.sign(w2.T @ y)
        u = np.sign(w1.T @ z)

        # # first attempted LR:
        # θi.shape = (4, 1, 1) (first, and then try (4, N, N))
        # for p,(a,b) in enumerate(it.product((x, u), (h, z))):
        #     w1 += b * θ1[p] * a.T
        # for p,(a,b) in enumerate(it.product((h, z), (v, y))):
        #     w2 += b * θ2[p] * a.T    
        # return w1, w2

        # accumulate the sign-solve data for the foregoing
        for (i,m) in it.product(range(N), range(M)):
            A_ub_row = np.zeros((1, 4))
            # input -> hidden
            for p,(a,b) in enumerate(it.product((x, u), (h, z))):
                A_ub_row[0,p] = - (V[i,hidx[m]] * b[i]) * (a.T @ V[:,kidx[m]].reshape(-1,1))
            b_ub_row = V[i,hidx[m]] * (w1[i,:] * V[:,kidx[m]]).sum() - 1
            A_ub += (A_ub_row,)
            b_ub += (b_ub_row,)

    # run the gigantic signsolve for current option
    A_ub = np.concatenate(A_ub, axis=0)
    b_ub = np.array(b_ub)
    c = -A_ub.mean(axis=0)
    result = so.linprog(c, A_ub, b_ub, bounds=(None, None))#, method='simplex')

    print(f"{lp} of {numopt}: {result.status}")
    if result.success: break

if result.success:
    print(result.x)

