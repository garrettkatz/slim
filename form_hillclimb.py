import itertools as it
import numpy as np
import torch as tr
import hardform as hf
import softform as sf
import svm_data as sd
from svm_eval import svm_eval

def get_loss(idx, ops, transitions):

    formula = hf.to_tree(idx, ops)

    loss = 0.
    for (w_new, w_old, x, y, _1) in transitions:
        inputs = {'w': w_old, 'x': x, 'y': y, '1': _1}
        w_pred = sf.form_eval(formula, inputs)
        w_pred = tr.nn.functional.normalize(w_pred)
        loss += -(w_new*w_pred).sum(dim=1).mean().item() # already normalized

    return loss / len(transitions)

if __name__ == "__main__":

    transitions = []
    for N in range(3,5):
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        W, X, Y = ltms["W"], {N: ltms["X"]}, {N: ltms["Y"]}
        print(N)
        print(W)
        w_new, w_old, x, y = sd.all_transitions(X, Y, N)

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

    # random start, tends to converge to f = w
    idx = np.random.choice(num_ops, size=num_nodes)
    idx[2**max_depth - 1:] = np.random.choice(len(ops[0]), size=2**max_depth)

    # perceptron rule:
    # w <- w + (y - sign(dot(w,x)))*x
    idx = [0] * (2**(max_depth + 1) - 1) # comment to start with random in masked subtrees
    idx[ 0] = 10 # add(
    idx[ 1] =  0 #     w,
    idx[ 2] = 12 #     mul(
    idx[ 5] = 11 #         sub(
    idx[11] =  2 #             y,
    idx[12] =  5 #             sign(
    idx[25] = 15 #                 dot(
    idx[51] =  0 #                     w,
    idx[52] =  1 #                     x)))),
    idx[ 6] =  1 #         x))

    idx = tuple(idx)
    explored = {idx: get_loss(idx, ops, transitions)}

    print(f"starting with [{explored[idx]}] {sf.form_str(hf.to_tree(idx, ops))}")

    for itr in it.count():

        best_loss = explored[idx]
        best_idx = idx

        for i in range(len(idx)):
            # print(f"{itr}: {i} of {len(idx)}")
            # leaves must be 0-ary
            k_bound = num_ops if i < 2**max_depth - 1 else len(ops[0])

            for k in range(k_bound):
                if k == idx[i]: continue
    
                child_idx = tuple(k if i == j else idx[j] for j in range(len(idx)))
    
                if child_idx not in explored:
                    explored[child_idx] = get_loss(child_idx, ops, transitions)
                    # print(f"  {itr},{i},{k}: {explored[child_idx]} {sf.form_str(hf.to_tree(child_idx, ops))}")
    
                if explored[child_idx] < best_loss:
                    best_idx = child_idx
                    best_loss = explored[child_idx]

        if best_idx != idx:
            idx = best_idx
            tree = hf.to_tree(best_idx, ops)
            print(f"{itr} new best [{explored[best_idx]}]: {sf.form_str(tree)}")

        else:

            break

    print(f"Local opt in {itr} iters. eval...")

    formula = hf.to_tree(best_idx, ops)

    num_runs = 30

    N = 7
    ltms = np.load(f"ltms_{N}_c.npz")
    X, Y = {N: ltms["X"]}, {N: ltms["Y"]}

    def update_rule(w, x, y, N):
        inputs = {
            'w': tr.nn.functional.normalize(tr.tensor(w, dtype=tr.float32).view(1,N)),
            'x': tr.tensor(x, dtype=tr.float32).view(1,N),
            'y': tr.tensor(y, dtype=tr.float32).view(1,1).expand(1, N), # broadcast
            '1': tr.ones(1,N), # broadcast
        }
        w_pred = sf.form_eval(formula, inputs)
        w_pred = tr.nn.functional.normalize(w_pred)
        return w_pred.clone().squeeze().numpy()

    loss, accu = svm_eval(X, Y, update_rule, num_runs)
    accu = np.mean(accu)
    loss = np.mean(loss * 180/np.pi) # convert to degrees
    print(f"svm loss={loss}, accu={accu}")

