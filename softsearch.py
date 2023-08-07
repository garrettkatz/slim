import itertools as it
import numpy as np
import torch as tr
from softform import form_str, dot, inv, square
from svm_data import random_transitions
from softfit import SpanRule
import heapq

if __name__ == "__main__":
    
    X, Y = {}, {}
    for N in range(3, 6):
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X[N], Y[N] = ltms["X"], ltms["Y"]

    ops = {
        0: ['w', 'x', 'y', 'N', '1'],
        1: [tr.neg, tr.sign, square],
        2: [tr.add, tr.mul, tr.maximum, tr.minimum, dot],
    }

    N = 5
    B = 16
    max_depth = 8
    num_itrs = 10

    use_ham = False

    if use_ham:  ops[0].extend([f"x{i}" for i in range(N)])

    model = SpanRule(ops, max_depth, B, N)

    def loss_grad(alpha_attn, beta_attn):
        model.alpha.attention.data = alpha_attn
        model.beta.attention.data = beta_attn

        # training batch
        w_new, w_old, x, y = random_transitions(X, Y, N, B)
        w_new = tr.nn.functional.normalize(tr.tensor(w_new, dtype=tr.float32))
    
        inputs = {
            'w': tr.nn.functional.normalize(tr.tensor(w_old, dtype=tr.float32)),
            'x': tr.tensor(x, dtype=tr.float32),
            'y': tr.tensor(y, dtype=tr.float32).view(-1,1).expand(B, N), # broadcast
            'N': N*tr.ones(B,N), # broadcast
            '1': tr.ones(B,N), # broadcast
        }
        if use_ham:
            for i in range(N):
                inputs[f"x{i}"] = inputs['x'] * (-1)**tr.eye(N)[i]

        # backprop
        pred = tr.nn.functional.normalize(model(inputs))
        loss = -(w_new*pred).sum(dim=1).mean()
        loss.backward()

        alpha_grad =  model.alpha.attention.grad.clone().detach()
        beta_grad =  model.beta.attention.grad.clone().detach()
        return loss.item(), (alpha_grad, beta_grad)

    # simplest formula
    attn = tr.zeros(model.alpha.attention.shape)
    attn[0, :] = 1

    def to_key(attns):
        key = tuple(attns[0].flatten().tolist()) + tuple(attns[1].flatten().tolist())
        return key
    def to_attn(key):
        attns = tr.tensor(key).reshape(2, *attn.shape)
        return attns[0], attns[1]

    heap = [(0, to_key((attn, attn)))]
    explored = {}

    best_loss = 1.
    best_attns = None

    for itr in it.count():

        # stop when priority queue exhausted or max itrs reached
        if len(heap) == 0: break
        if itr == num_itrs: break

        # pop highest priority (lowest linproxed loss) so far
        _, key = heapq.heappop(heap)
        if key in explored: continue
        attns = to_attn(key)

        # evaluate popped loss and gradient
        loss, grads = loss_grad(*attns)
        explored[key] = loss

        msg = f"{itr} of {num_itrs}: loss={loss:.3f}"

        # track best so far
        if loss < best_loss:
            best_loss = loss
            best_attns = attns
            msg += " ************** new best"

        print(msg)

        # iterate over all neighbors
        for ab in range(2):
            for n in range(attns[ab].shape[0]):
                print(ab, n)
                for f in range(attns[ab].shape[1]):

                    # don't include popped as its own neighbor
                    if attns[ab][n,f] == 1: continue

                    # calculate the neighbor priority
                    new_attns = (attns[0].clone().detach(), attns[1].clone().detach())
                    new_attns[ab][n,:] = 0
                    new_attns[ab][n,f] = 1

                    # gradient linprox with sparse delta vector
                    p = loss + ((new_attns[ab][n] - attns[ab][n])*grads[ab][n]).sum().item()

                    # insert neighbor into queue
                    heapq.heappush(heap, (p, to_key(new_attns)))

