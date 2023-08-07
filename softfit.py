import pickle as pk
import numpy as np
import torch as tr
from softform import SoftForm, form_str, dot, inv, square
from svm_data import random_transitions
from svm_eval import svm_eval

class SpanRule(tr.nn.Module):
    def __init__(self, ops, max_depth, B, N, use_ham=False):
        super().__init__()

        self.use_ham = use_ham
        if use_ham:
            ops = dict(ops)
            ops[0].extend([f"x{i}" for i in range(N)])

        self.alpha = SoftForm(ops, max_depth, B, N)
        self.beta = SoftForm(ops, max_depth, B, N)

    def forward(self, inputs):
        # add nearby input vector with hamming distance = 1
        if self.use_ham:
            inputs = dict(inputs)
            for i in range(N):
                inputs[f"x{i}"] = inputs['x'] * (-1)**tr.eye(N)[i]

        # resize formulas for current inputs
        self.alpha.reset_dims(*inputs['w'].shape)
        self.beta.reset_dims(*inputs['w'].shape)

        # apply soft formulas, span rule and normalize
        alpha = self.alpha(inputs)
        beta = self.beta(inputs)
        w_pred = alpha * inputs['w'] + beta * inputs['x'] # allows vector, element-wise coefficients
        w_pred = tr.nn.functional.normalize(w_pred)

        return w_pred

if __name__ == "__main__":

    Ns = list(range(3,6))
    X, Y = {}, {}
    for N in Ns:
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X[N], Y[N] = ltms["X"], ltms["Y"]
        print(N, Y[N].shape[0])

    ops = {
        0: ['w', 'x', 'y', '1'], # large N makes overflows, normalizing and no inv anyway
        1: [tr.neg, tr.sign, square],
        2: [tr.add, tr.mul, tr.maximum, tr.minimum, dot],
    }

    B = 16
    max_depth = 8
    use_ham = False

    do_train = True
    lr = 0.0001

    # # quick test
    # num_itrs = 20
    # num_runs = 10

    # medium run
    num_itrs = 500
    num_runs = 30

    # # big run
    # num_itrs = 10000
    # num_runs = 1000

    model = SpanRule(ops, max_depth, B, Ns[0], use_ham)
    cossim = tr.nn.CosineSimilarity()
    opt = tr.optim.Adam(model.parameters(), lr=lr)

    init_attn = model.beta.attention.detach().clone().numpy()

    if do_train:

        losses, gns = [], []
        for itr in range(num_itrs):
    
            # training batch
            N = np.random.choice(Ns)
            w_new, w_old, x, y = random_transitions(X, Y, N, B)
            w_new = tr.nn.functional.normalize(tr.tensor(w_new, dtype=tr.float32))
        
            inputs = {
                'w': tr.nn.functional.normalize(tr.tensor(w_old, dtype=tr.float32)),
                'x': tr.tensor(x, dtype=tr.float32),
                'y': tr.tensor(y, dtype=tr.float32).view(-1,1).expand(B, N), # broadcast
                '1': tr.ones(B,N), # broadcast
            }
    
            # backprop on negative cosine similarity
            w_pred = model(inputs)
            loss = -(w_new*w_pred).sum(dim=1).mean() # already normalized
            loss.backward()
            losses.append(loss.item())    

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

        inputs = {
            'w': tr.nn.functional.normalize(tr.tensor(w, dtype=tr.float32).view(1,N)),
            'x': tr.tensor(x, dtype=tr.float32).view(1,N),
            'y': tr.tensor(y, dtype=tr.float32).view(1,1).expand(1, N), # broadcast
            '1': tr.ones(1,N), # broadcast
        }
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
