import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import scipy.optimize as so
from nondet import NonDeterminator


# LR(W, x, y):
# lrterms(W, x, y) -> x, W1x, sign(W2 sign(W1x)), etc
# make each term shape (N,N) (outer products, broadcast, etc)
# W_new[i,j] <- (stack(lrterms(W, x, y))[:,i,j] * theta[:,i,j]).sum(axis=0)
# W_new[i] @ x = ((stack(lrterms(W, x, y))[:,i,:] * theta[:,i,:]).sum(axis=0) * x).sum()

# W_new[i] @ inp[n] = (theta[t,i,j] * lrterms(W,x,y)[t,i,j] * inp[_,_,j]).sum(axes=(0,2))
# coeff[n,t,i,j] = lrterms(W,x,y)[_,t,i,j] * inp[n,_,_,j]

def fitlr(terms, inp, out)
    # terms[t,i,j]
    A_ub
    
    A_ub = np.stack([
        -out[i](terms * inp[n]).flatten()
    
    b_ub = -np.ones(A_ub.shape[0])

def lrterms1(W, x, y):
    return [W, 

N = 4 # number of neurons
M = 4 # number of key-value pairs

V = np.array(tuple(it.product((-1, 1), repeat=N))).T

w1 = np.eye(N)
w2 = np.eye(N)
# kidx = tuple(range(M))
kidx = (0, 3, 5, 6)

print(V)
print(V[:,kidx])
input('.')

nd = NonDeterminator()

def fit_search():
    A_ub, b_ub = (), ()
    for mv, k in it.permutations(range(M), 2):
        vidx = kidx[:mv] + (kidx[k],) + kidx[mv+1:]

        x = V[:,kidx[mv]].reshape(-1,1)
        y = V[:,vidx[mv]].reshape(-1,1)
        h = np.sign(w1 @ x)
        v = np.sign(w2 @ h)
        z = np.sign(w2.T @ y)
        u = np.sign(w1.T @ z)

        hidx = ()
        for mh in enumerate(range(M)):
            j = nd.choice(range(2**N))
            hidx += (j,)

            # # first attempted LR:
            # θi.shape = (4, 1, 1) (first, and then try (4, N, N))
            # for p,(a,b) in enumerate(it.product((x, u), (h, z))):
            #     w1 += b * θ1[p] * a.T
            # for p,(a,b) in enumerate(it.product((h, z), (v, y))):
            #     w2 += b * θ2[p] * a.T    
            # return w1, w2

            # for each (w,idx,odx), i < N, and m < len(hidx)
            # (w_new[i,c] * V[c,idx[m]]).sum(c) * V[i,odx[m]] > 1
            # (dw[i,c] * V[c,idx[m]]).sum(c) * V[i,odx[m]] > 1 - (w[i,c] * V[c,idx[m]]).sum(c) * V[i,odx[m]]
            # ((b[p][i] * θ[p,i,c] * a[p][c]).sum(p,a,b) * V[c,idx[m]]).sum(c) * V[i,odx[m]] > b_lb[i,m]
            # (b[p][i] * θ[p,i,c] * a[p][c] * V[c,idx[m]] * V[i,odx[m]]).sum(c,p) > b_lb[i,m]
            # (θ[p,i,c] * A_lb[c,p,i,m]).sum(c,p) > b_lb[i,m]
            # (θ[p] * A_lb[c,p,i,m]).sum(c,p) > b_lb[i,m] # simpler θ shape 411 version

            for (w, idx, odx) in ((w1,kidx,hidx), (w2, hidx, vidx)):
                b_lb = 1 - (w @ V[:,idx[:len(hidx)]]) * V[:,odx[:len(hidx)]]
                for i,m in it.product(range(N), range(len(hidx))):
                    A_lb = np.zeros((1,4))
                    for p,(a,b) in enumerate(it.product((x, u), (h, z))):
                        A_lb[0,p] = b[i] * a.T @ V[:,idx[m]] * V[i,odx[m]]
                    A_ub += (-A_lb,)
                    b_ub += (-b_lb[i,m],)
                
            # rerun gigantic signsolve for current hidx
            A_ub_mat = np.concatenate(A_ub, axis=0)
            b_ub_mat = np.array(b_ub)
            c = -A_ub_mat.mean(axis=0)
            result = so.linprog(c, A_ub_mat, b_ub_mat, bounds=(None, None))#, method='simplex')

            if not result.success:
                return False, (mv, k, hidx)

    # success at this point
    return True, result.x

for r,ret in enumerate(nd.runs(fit_search)):
    success, result = ret
    if success:
        print(f"rep {r} success! result = {result}")
        break
    else:
        mv, k, hidx = result
        print(f"rep {r}: failed at mv={mv}, k={k}, len(hidx)={len(hidx)}")

