import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import scipy.optimize as so
from nondet_unsized import NonDeterminator

# LR(W, x, y):
# lrterms(W, x, y) -> x, W1x, sign(W2 sign(W1x)), etc
# make each term shape (N,N) (outer products, broadcast, etc)
# W_new[i,j] <- (stack(lrterms(W, x, y))[:,i,j] * theta[:,i,j]).sum(axis=0)
# W_new[i] @ x = ((stack(lrterms(W, x, y))[:,i,:] * theta[:,i,:]).sum(axis=0) * x).sum()

# e is edges in graph
# for all e,i,j:
# - out[e,i,j] * (W_new[e,i,k] * inp[e,k,j]).sum(k) <= -1
# - out[e,i,j] * ((theta[e,t,i,k] * terms[e,t,i,k]).sum(t) * inp[e,k,j]).sum(k) <= -1
# (- out[e,i,j] * inp[e,k,j] * terms[e,t,i,k] * theta[t,i,k]).sum(t,k) <= -1
# (- out[e,i,j] * inp[e,k,j] * terms[e,i,t,k] * theta[i,t,k]).sum(t,k) <= -1

# (- out[i,e,j,0,0] * inp[0,e,j,0,k] * terms[i,e,0,t,k] * theta[i,0,0,t,k]).sum(t,k) <= -1

def fitlr(terms, inp, out):

    # out[e,i,j] -> [i,e,j,0,0]
    out = out.transpose(axes=(1,0,2))[:,:,:,np.newaxis,np.newaxis]
    # inp[e,k,j] -> inp[0,e,j,0,k]
    inp = inp.transpose(axes=(0,2,1))[np.newaxis,:,:,np.newaxis,:]
    # terms[e,t,i,k] -> terms[i,e,0,t,k]
    terms = terms.transpose(axes=(2,0,1,3))[:,:,np.newaxis,:,:] # put t after i

    T, K = terms.shape[3:]
    
    θ = []
    for i in range(inp.shape[0]):

        A_ub = -out * inp * terms
        A_ub = A_ub.reshape(-1, T*K)
        b_ub = -np.ones(A_ub.shape[0])
        c = -A_ub.mean(axis=0)
        result = so.linprog(c, A_ub, b_ub, bounds=(None, None))#, method='simplex')

        if not result.status in (0, 3): return False, θ
        θ.append(result.x.reshape((T,K)))

    # theta[i,t,k] -> theta[t,i,k]
    θ = np.stack(θ).tranpose(axes=(1,0,2))
    return True, θ

T_fser = 3
def lrterms_fser(W, x, y):
    z = np.sign(W @ x)
    terms = np.stack((W, y * x.T, z * x.T))
    return terms

N = 2 # number of neurons
M = 2 # number of key-value pairs

V = np.array(tuple(it.product((-1, 1), repeat=N))).T

w1 = np.eye(N)
w2 = np.eye(N)
# kidx = tuple(range(M))
kidx = {
    2: (0, 1),
    4: (0, 3, 5, 6)
}[N]

print(V)
print(V[:,kidx])

nd = NonDeterminator()


E = M**(M+2)
print(f"{E} edges...")
input('.')

def checkfit():

    # out[e,i,j] -> [i,e,j,0,0]
    # inp[e,k,j] -> inp[0,e,j,0,k]
    # terms[e,t,i,k] -> terms[i,e,0,t,k]
    # 2*E for two layers
    inp = np.zeros((2*E, N, M))
    out = np.zeros((2*E, N, M))
    terms = np.zeros((2*E, T_fser, N, N))

    # identity net vidx = hidx = kidx
    hidxs = {kidx: kidx}
    weights = {kidx: (np.eye(N), np.eye(N))}
    queue = [kidx]
    
    e = 0
    while len(queue) > 0:
        vidx = queue.pop()
        hidx = hidxs[vidx]

        for (j, k) in it.product(range(M), repeat=2):
            new_vidx = vidx[:j] + (kidx[k],) + vidx[j+1:]
            if new_vidx not in hidxs:
                # hidxs[new_vidx] = np.choice(it.product(range(2**N), repeat=M))
                hidxs[new_vidx] = kidx # first layer identity
            new_hidx = hidxs[new_vidx]

            # # layer 1
            # inp[e] = V[:, kidx]
            # out[e] = V[:, new_hidx]
            # terms[e] = lrterms_fser(weights[vidx][0], V[:,kidx[j]], ?])
            # e += 1

            # layer 2
            inp[e] = V[:, new_hidx]
            out[e] = V[:, new_vidx]
            terms[e] = lrterms_fser(weights[vidx][1], V[:,new_hidx[j]], V[:,new_vidx[j]]])
            e += 1
