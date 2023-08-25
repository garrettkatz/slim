import itertools as it
import numpy as np
import torch as tr
import hardform as hf
import softform as sf
import svm_data as sd
from svm_eval import svm_eval
from form_softclimb import get_loss_grad
from form_hillclimb import get_loss

if __name__ == "__main__":

    import matplotlib.pyplot as pt

    transitions = []
    for N in range(3,5):
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        W, X, Y = ltms["W"], {N: ltms["X"]}, {N: ltms["Y"]}
        print(N)
        print(W)
        w_new, w_old, x, y, margins = sd.all_transitions(X, Y, N)

        w_new = tr.nn.functional.normalize(tr.tensor(np.concatenate(w_new, axis=0), dtype=tr.float32))
        w_old = tr.nn.functional.normalize(tr.tensor(np.concatenate(w_old, axis=0), dtype=tr.float32))
        x = tr.tensor(np.stack(x), dtype=tr.float32)
        y = tr.tensor(y, dtype=tr.float32).view(-1,1).expand(*x.shape) # broadcast
        _1 = tr.ones(*x.shape)
        transitions.append((w_new, w_old, x, y, _1))

    max_depth = 5
    num_nodes = 2**(max_depth+1) - 1

    ops = {
        0: ['w', 'x', 'y', '1'],
        1: [tr.neg, tr.sign, sf.square, sf.min, sf.max, sf.mean],
        2: [tr.add, tr.sub, tr.mul, tr.maximum, tr.minimum, sf.dotmean, sf.project, sf.reject],
    }
    num_ops = sum(map(len, ops.values()))

    num_reps = 5

    for rep in range(num_reps):

        child_losses, child_approxes = [], []

        # random formula
        idx = np.random.choice(num_ops, size=num_nodes)
        idx[2**max_depth - 1:] = np.random.choice(len(ops[0]), size=2**max_depth)
        loss, grad = get_loss_grad(idx, ops, transitions)

        print("rep form, grad max:")
        print(sf.form_str(hf.to_tree(idx, ops)))
        print(grad.abs().max().item())

        for i in range(len(idx)):
            print(f"{rep}|{num_reps}, {i} of {len(idx)}, |grad[i]| <= {grad[i].abs().max().item()}")
            # leaves must be 0-ary
            k_bound = num_ops if i < 2**max_depth - 1 else len(ops[0])
            for k in range(k_bound):
                if k == idx[i]: continue
    
                child_idx = tuple(k if i == j else idx[j] for j in range(len(idx)))
                child_loss = get_loss(child_idx, ops, transitions)
                child_approx = loss + (grad[i, k] - grad[i, idx[i]]) # f(x) + g(x).dot(x'-x)
    
                child_losses.append(child_loss)
                child_approxes.append(child_approx)
    
        pt.subplot(1, num_reps, rep+1)
        pt.scatter(child_losses, child_approxes)
        pt.xlabel("Actual loss")
        pt.ylabel("First order approx")

    pt.show()
