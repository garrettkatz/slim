import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import scipy.optimize as so
from nondet_unsized import NonDeterminator

rerun = False

N = 2 # number of neurons
M = 2 # number of key-value pairs
save_depth = 0

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

if rerun:

    for r, (success, θ, W) in enumerate(nd.runs(checkfit)):
    
        print(f"run {r}")
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

# # LR(W, x, y):
# # lrterms(W, x, y) -> x, W1x, sign(W2 sign(W1x)), etc
# # make each term shape (N,N) (outer products, broadcast, etc)
# # W_new[i,j] <- (stack(lrterms(W, x, y))[:,i,j] * theta[:,i,j]).sum(axis=0)
# # W_new[i] @ x = ((stack(lrterms(W, x, y))[:,i,:] * theta[:,i,:]).sum(axis=0) * x).sum()

# # e is edges in graph
# # for all e,i,j:
# # - out[e,i,j] * (W_new[e,i,k] * inp[e,k,j]).sum(k) <= -1
# # - out[e,i,j] * ((theta[e,t,i,k] * terms[e,t,i,k]).sum(t) * inp[e,k,j]).sum(k) <= -1
# # (- out[e,i,j] * inp[e,k,j] * terms[e,t,i,k] * theta[t,i,k]).sum(t,k) <= -1
# # (- out[e,i,j] * inp[e,k,j] * terms[e,i,t,k] * theta[i,t,k]).sum(t,k) <= -1

# # (- out[i,e,j,0,0] * inp[0,e,j,0,k] * terms[i,e,0,t,k] * theta[i,0,0,t,k]).sum(t,k) <= -1

# def fitlr(terms, inp, out):

#     # out[e,i,j] -> [i,e,j,0,0]
#     out = out.transpose(axes=(1,0,2))[:,:,:,np.newaxis,np.newaxis]
#     # inp[e,k,j] -> inp[0,e,j,0,k]
#     inp = inp.transpose(axes=(0,2,1))[np.newaxis,:,:,np.newaxis,:]
#     # terms[e,t,i,k] -> terms[i,e,0,t,k]
#     terms = terms.transpose(axes=(2,0,1,3))[:,:,np.newaxis,:,:] # put t after i

#     T, K = terms.shape[3:]
    
#     θ = []
#     for i in range(inp.shape[0]):

#         A_ub = -out * inp * terms
#         A_ub = A_ub.reshape(-1, T*K)
#         b_ub = -np.ones(A_ub.shape[0])
#         c = -A_ub.mean(axis=0)
#         result = so.linprog(c, A_ub, b_ub, bounds=(None, None))#, method='simplex')

#         if not result.status in (0, 3): return False, θ
#         θ.append(result.x.reshape((T,K)))

#     # theta[i,t,k] -> theta[t,i,k]
#     θ = np.stack(θ).tranpose(axes=(1,0,2))
#     return True, θ

# T_fser = 3
# def lrterms_fser(W, x, y):
#     z = np.sign(W @ x)
#     terms = np.stack((W, y * x.T, z * x.T))
#     return terms

# N = 2 # number of neurons
# M = 2 # number of key-value pairs

# V = np.array(tuple(it.product((-1, 1), repeat=N))).T

# w1 = np.eye(N)
# w2 = np.eye(N)
# # kidx = tuple(range(M))
# kidx = {
#     2: (0, 1),
#     4: (0, 3, 5, 6)
# }[N]

# print(V)
# print(V[:,kidx])

# nd = NonDeterminator()


# E = M**(M+2)
# print(f"{E} edges...")
# input('.')

# def checkfit():

#     # out[e,i,j] -> [i,e,j,0,0]
#     # inp[e,k,j] -> inp[0,e,j,0,k]
#     # terms[e,t,i,k] -> terms[i,e,0,t,k]
#     # 2*E for two layers
#     inp = np.zeros((2*E, N, M))
#     out = np.zeros((2*E, N, M))
#     terms = np.zeros((2*E, T_fser, N, N))

#     # identity net vidx = hidx = kidx
#     hidxs = {kidx: kidx}
#     weights = {kidx: (np.eye(N), np.eye(N))}
#     queue = [kidx]
    
#     e = 0
#     while len(queue) > 0:
#         vidx = queue.pop()
#         hidx = hidxs[vidx]

#         for (j, k) in it.product(range(M), repeat=2):
#             new_vidx = vidx[:j] + (kidx[k],) + vidx[j+1:]
#             if new_vidx not in hidxs:
#                 # hidxs[new_vidx] = np.choice(it.product(range(2**N), repeat=M))
#                 hidxs[new_vidx] = kidx # first layer identity
#             new_hidx = hidxs[new_vidx]

#             # # layer 1
#             # inp[e] = V[:, kidx]
#             # out[e] = V[:, new_hidx]
#             # terms[e] = lrterms_fser(weights[vidx][0], V[:,kidx[j]], ?])
#             # e += 1

#             # layer 2
#             inp[e] = V[:, new_hidx]
#             out[e] = V[:, new_vidx]
#             terms[e] = lrterms_fser(weights[vidx][1], V[:,new_hidx[j]], V[:,new_vidx[j]]])
#             e += 1
