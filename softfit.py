import pickle as pk
import numpy as np
import torch as tr
from softform import SoftForm, form_str, dot, inv, square
from svm_data import random_transitions
from svm_eval import svm_eval

class SpanRule(tr.nn.Module):
    def __init__(self, ops, max_depth, B, N):
        super().__init__()

        self.alpha = SoftForm(ops, max_depth, B, N)
        self.beta = SoftForm(ops, max_depth, B, N)

    def forward(self, inputs):
        # allows vector, element-wise coefficients
        alpha = self.alpha(inputs)
        beta = self.beta(inputs)
        return alpha * inputs['w'] + beta * inputs['x']

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

    N = 4
    B = 16
    max_depth = 8

    lr = 0.0001

    # # quick test
    # num_itrs = 20
    # num_runs = 10

    # big run
    num_itrs = 10000
    num_runs = 1000

    use_ham = False

    do_train = False

    if use_ham: ops[0].extend([f"x{i}" for i in range(N)])

    # model = SoftForm(ops, max_depth, B, N)
    model = SpanRule(ops, max_depth, B, N)
    cossim = tr.nn.CosineSimilarity()
    opt = tr.optim.Adam(model.parameters(), lr=lr)

    init_attn = model.beta.attention.detach().clone().numpy()

    if do_train:

        losses, gns = [], []
        for itr in range(num_itrs):
    
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
            pred = model(inputs)
    
            # loss = -cossim(w_new, pred).mean()
    
            pred = tr.nn.functional.normalize(pred)
            loss = -(w_new*pred).sum(dim=1).mean()
    
            losses.append(loss.item())
    
            loss.backward()
            gn = tr.linalg.vector_norm(tr.nn.utils.parameters_to_vector(model.parameters()))
            gns.append(gn.item())
            
            opt.step()
            opt.zero_grad()
    
            if itr % 100 == 0 or itr+1 == num_itrs:
                print(f"{itr} of {num_itrs}", losses[-1], gns[-1], form_str(model.alpha.harden()), form_str(model.beta.harden()))
                tr.save(model, 'softfit.pt')
                with open('softfit.pkl', 'wb') as f:
                    pk.dump((losses, gns, init_attn), f)

    model = tr.load('softfit.pt')
    with open('softfit.pkl', 'rb') as f:
        (losses, gns, init_attn) = pk.load(f)

    opt_attn = model.beta.attention.detach().clone().numpy()

    print(f"attn {init_attn.min():.3f}~{init_attn.mean():.3f}~{init_attn.max():.3f} to {opt_attn.min():.3f}~{opt_attn.mean():.3f}~{opt_attn.max():.3f}")

    print("form", form_str(model.alpha.harden()), form_str(model.beta.harden()))

    def update_rule(w, x, y, N):

        model.alpha.reset_dims(1, N)
        model.beta.reset_dims(1, N)

        inputs = {
            'w': tr.nn.functional.normalize(tr.tensor(w, dtype=tr.float32).view(1,N)),
            'x': tr.tensor(x, dtype=tr.float32).view(1,N),
            'y': tr.tensor(y, dtype=tr.float32).view(1,1).expand(1, N), # broadcast
            'N': N*tr.ones(1,N), # broadcast
            '1': tr.ones(1,N), # broadcast
        }
        if use_ham:
            for i in range(N):
                inputs[f"x{i}"] = inputs['x'] * (-1)**tr.eye(N)[i]
        with tr.no_grad():
            w_new = model(inputs)
        return w_new.clone().squeeze().numpy()

    print('svm eval...')    
    # loss, accu = svm_eval({N: X[N]}, {N: Y[N]}, update_rule, num_runs=100)
    loss, accu = svm_eval(X, Y, update_rule, num_runs)
    loss = loss * 180/np.pi # convert to degrees
    print('mean loss, accuracy', loss.mean(), accu.mean())

    import matplotlib.pyplot as pt
    pt.subplot(1,4,1)
    pt.plot(losses)

    pt.subplot(1,4,2)
    pt.plot(gns)

    pt.subplot(1,4,3)
    pt.imshow(init_attn)

    pt.subplot(1,4,4)
    pt.imshow(opt_attn)

    pt.show()
