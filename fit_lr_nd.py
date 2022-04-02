import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import scipy.optimize as so
# from nondet_unsized import NonDeterminator
from nondet_sized import NonDeterminator

rerun = True

# N = 2 # number of neurons
# M = 2 # number of key-value pairs
# save_depth = 0

N = 3 # number of neurons
M = 4 # number of key-value pairs
save_depth = 1

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, H = pk.load(f)
print("H.shape:", H.shape) # (num_dichotomies, num vertices = 2**N)

# load all in memory, not good for big N
kidx = tuple(range(M))
solns = {}
for lvidx in it.product(kidx, repeat=save_depth):
    vlead = "_".join(map(str, lvidx))
    fname = f"solns/N{N}M{M}_{vlead}"
    with open(fname, "rb") as f:
        solns.update(pk.load(f))

V = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(V)
print(V[:,kidx])

print(len(solns))
print(M**M)

E = M**(M+2)
print(f"{E} edges...")

nd = NonDeterminator()

def lrterms_fser(W, x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    h = np.sign(W[0] @ x)
    z = np.sign(W[1] @ h)
    terms1 = W[0][np.newaxis]
    terms2 = np.stack((W[1], y * h.T, z * h.T))
    return terms1, terms2
T0, T1 = 1, 3

def checkfit():
    W = {}
    # choose net for each node
    for vidx in it.product(kidx, repeat=M):
        hidx, m1, m2 = nd.choice(solns[vidx])
        W[vidx] = np.empty((2,N,N))
        for l,mm in enumerate((m1, m2)):
            for i in range(N):
                mi = nd.choice(mm[i])
                W[vidx][l,i] = weights[mi]
    # build term data
    terms = [np.empty((E, T0, N, N)), np.empty((E, T1, N, N))]
    targ_vidxs = []
    e = 0
    for vidx in it.product(kidx, repeat=M):        
        for (j,k) in it.product(range(M), repeat=2):
            terms[0][e], terms[1][e] = lrterms_fser(W[vidx], V[:,kidx[j]], V[:,kidx[k]])
            targ_vidxs.append(vidx[:j] + (kidx[k],) + vidx[j+1:])
            e += 1
    # solve theta
    θ = [np.empty((T0, N, N)), np.empty((T1, N, N))]
    for l in range(2):
        for i,j in it.product(range(N), repeat=2):
            A = terms[l][:,:,i,j]
            b = np.array([W[vidx][l,i,j] for vidx in targ_vidxs])
            x = np.linalg.lstsq(A, b, rcond=None)[0]

            if np.fabs(b - (A @ x)).max() > .4: return False, θ, W

            θ[l][:,i,j] = x

    return True, θ, W

def checkfit_progressive():
    # queue root net with identity weights
    m1 = [np.isclose(np.eye(N)[i], weights).all(axis=1).argmax() for i in range(N)]
    m2 = list(m1)
    W = {kidx: np.stack((np.eye(N),)*2)}
    queue = [kidx]

    # init term data
    terms = [np.empty((E, T0, N, N)), np.empty((E, T1, N, N))]
    targ_vidxs = []
    e = 0
    while len(queue) > 0:
        # choose next node to fit
        vidx = queue.pop()

        # enumerate edges
        for (j,k) in it.product(range(M), repeat=2):
            terms[0][e], terms[1][e] = lrterms_fser(W[vidx], V[:,kidx[j]], V[:,kidx[k]])
            new_vidx = vidx[:j] + (kidx[k],) + vidx[j+1:]
            targ_vidxs.append(new_vidx)
            
            # handle new nodes
            if new_vidx not in W:
                # choose net for new node
                hidx, m1, m2 = nd.choice(solns[new_vidx])
                W[new_vidx] = np.empty((2,N,N))
                for l,mm in enumerate((m1, m2)):
                    for i in range(N):
                        mi = nd.choice(mm[i])
                        W[new_vidx][l,i] = weights[mi]
                
                # add to queue
                queue.append(new_vidx)

            # advance edge counter
            e += 1

            # resolve theta
            θ = [np.empty((T0, N, N)), np.empty((T1, N, N))]
            for l in range(2):
                for i,j in it.product(range(N), repeat=2):
                    A = terms[l][:e,:,i,j]
                    b = np.array([W[vidx][l,i,j] for vidx in targ_vidxs[:e]])
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
        
                    if np.fabs(b - (A @ x)).max() > .4: return False, θ, W
        
                    θ[l][:,i,j] = x

    return True, θ, W

if rerun:

    # for r, (success, θ, W) in enumerate(nd.runs(checkfit)):
    for r, (success, θ, W) in enumerate(nd.runs(checkfit_progressive)):

        
        ctr = " ".join(f"{co}/{len(ch)}" for co,ch in zip(nd.counters, nd.choices) if len(ch) > 1)
        print(f"run {r}: {ctr}")
        if success:
            with open(f"fitlr_{N}_{M}.pkl", "wb") as f: pk.dump((θ, W), f)

            print("θ:")
            print(θ[0])
            print(θ[1].round(3))
            print("root W:")
            print(W[kidx][0].round(3))
            print(W[kidx][1].round(3))

            break

else:
    with open(f"fitlr_{N}_{M}.pkl", "rb") as f: (θ, W) = pk.load(f)

# doublecheck
vidx = kidx
w = W[vidx]
for t in range(100):
    j,k = np.random.choice(M, size=2)
    x = V[:,kidx[j]]
    y = V[:,kidx[k]]
    terms = lrterms_fser(w, x, y)
    w = tuple((terms[l] * θ[l]).sum(axis=0) for l in (0,1))
    vidx = vidx[:j] + (kidx[k],) + vidx[j+1:]

    print(t)
    assert np.allclose(np.sign(w[1] @ np.sign(w[0] @ V[:,kidx])), V[:,vidx])

print("Checked out")

