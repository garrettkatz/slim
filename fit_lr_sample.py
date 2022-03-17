import pickle as pk
import itertools as it
import numpy as np
import torch as tr
import matplotlib.pyplot as pt

N = 2 # number of neurons
M = 2 # number of key-value pairs
save_depth = 0

T = 10 # sequence length
num_iters = 1000
verb = 50

V = tr.tensor(tuple(it.product((-1, 1), repeat=N))).T.float()

fname = f"hemis_{N}.npy"
with open(fname,"rb") as f: weights, hemis = pk.load(f)
weights = tr.tensor(np.array(weights)).squeeze().float()

# load all in memory, not good for big N
kidx = tuple(range(M))
solns = {}
for lvidx in it.product(kidx, repeat=save_depth):
    vlead = "_".join(map(str, lvidx))
    fname = f"solns/N{N}M{M}_{vlead}"
    with open(fname, "rb") as f:
        solns.update(pk.load(f))

# θ1 = (tr.randn(4, N, N)*0.01).detach().requires_grad_()
# θ2 = (tr.randn(4, N, N)*0.01).detach().requires_grad_()
θ1 = (tr.randn(4,1,1)*0.001).detach().requires_grad_()
θ2 = (tr.randn(4,1,1)*0.001).detach().requires_grad_()
def lr1(w1, w2, x, y):
    h = tr.sign(w1 @ x)
    v = tr.sign(w2 @ h)
    z = tr.sign(w2.T @ y)
    u = tr.sign(w1.T @ z)

    w1 = w1.detach().clone()
    w2 = w1.detach().clone()

    for p,(a,b) in enumerate(it.product((x, u), (h, z))):
        w1 += b * θ1[p] * a.T
    for p,(a,b) in enumerate(it.product((h, z), (v, y))):
        w2 += b * θ2[p] * a.T

    return w1, w2

lr_fn = lr1

opt = tr.optim.SGD((θ1, θ2), lr=0.001)
loss = []

for itr in range(num_iters):
    
    vidx = kidx
    m1 = [tr.isclose(tr.eye(N)[i], weights).all(dim=1).int().argmax().item() for i in range(N)]
    m2 = list(m1)

    itr_loss = []

    opt.zero_grad()

    for t in range(T):
        j, k = np.random.choice(M, size=2)
        new_vidx = vidx[:j] + (k,) + vidx[j+1:]

        x = V[:,kidx[j]].reshape((N,1))
        y = V[:,kidx[k]].reshape((N,1))
        w1, w2 = lr_fn(weights[m1], weights[m2], x, y)

        best_dist = np.inf
        best_m1, best_m2 = None, None
        for (_, mm1, mm2) in solns[new_vidx]:
            new_m1 = [mm1[i][((w1[i] - weights[mm1[i]])**2).sum(dim=1).argmin()] for i in range(N)]
            new_m2 = [mm2[i][((w2[i] - weights[mm2[i]])**2).sum(dim=1).argmin()] for i in range(N)]
            dist = ((w1 - weights[new_m1])**2).sum() + ((w2 - weights[new_m2])**2).sum() / (T * N**2)
            if dist < best_dist:
                best_dist = dist
                best_m1, best_m2 = new_m1, new_m2

        itr_loss.append(best_dist.item())

        # backprop/interpolate solve
        best_dist.backward()
        
        # update for next timestep
        vidx = new_vidx
        m1, m2 = best_m1, best_m2

    opt.step()

    loss.append(itr_loss)
    if itr % verb == 0:
        print(f"{itr}: {np.mean(itr_loss)}")

loss = np.array(loss)

pt.figure()
pt.plot(loss.mean(axis=1), 'b')
pt.plot(loss.mean(axis=1) + loss.std(axis=1), 'r')

pt.figure()
for p in range(θ1.shape[0]):
    pt.subplot(2, θ1.shape[0], p+1)
    pt.imshow(θ1[p].detach().numpy())
    pt.subplot(2, θ1.shape[0], θ1.shape[0]+p+1)
    pt.imshow(θ2[p].detach().numpy())

pt.show()
