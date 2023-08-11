import matplotlib.pyplot as pt
import pickle as pk
from multiprocessing import Pool
import numpy as np
import torch as tr
import softform
from svm_data import random_transitions
from svm_eval import svm_eval

class SpanRule(tr.nn.Module):
    def __init__(self, ops, max_depth, B, N, use_ham=False):
        super().__init__()

        self.use_ham = use_ham
        if use_ham:
            ops = dict(ops)
            ops[0].extend([f"x{i}" for i in range(N)])

        self.alpha = softform.SoftForm(ops, max_depth, B, N)
        self.beta = softform.SoftForm(ops, max_depth, B, N)

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
        if tr.isnan(w_pred).any():
            if tr.isnan(alpha).any(): print('alpha')
            elif tr.isnan(beta).any(): print('beta')
            elif tr.isnan(inputs["w"]).any(): print('inputs["w"]')
            elif tr.isnan(inputs["x"]).any(): print('inputs["x"]')
            else:
                print("alpha", alpha)
                print("inputs['w']", inputs['w'])
                print("beta", beta)
                print("inputs['x']", inputs['x'])
            input('um.')
        w_pred = tr.nn.functional.normalize(w_pred)
        if tr.isnan(w_pred).any():
            input('arg.')

        return w_pred

class VecRule(tr.nn.Module):
    def __init__(self, ops, max_depth, B, N, use_ham=False, logits=True):
        super().__init__()

        self.use_ham = use_ham
        if use_ham:
            ops = dict(ops)
            ops[0].extend([f"x{i}" for i in range(N)])

        self.sf = softform.SoftForm(ops, max_depth, B, N, logits)

    def forward(self, inputs):
        # add nearby input vector with hamming distance = 1
        if self.use_ham:
            inputs = dict(inputs)
            for i in range(N):
                inputs[f"x{i}"] = inputs['x'] * (-1)**tr.eye(N)[i]

        # resize formulas for current inputs
        self.sf.reset_dims(*inputs['w'].shape)

        # apply soft formula and normalize
        w_pred = self.sf(inputs)
        w_pred = tr.nn.functional.normalize(w_pred)
        return w_pred

def do_training_run(rep):

    # make sure different runs have different random numbers
    tr.manual_seed(tr.initial_seed() + 100*rep)

    Ns = list(range(3,8))

    B = 16
    max_depth = 5
    use_ham = False

    # lr = 0.0001 # no logits
    lr = 0.00025 # logits

    # # quick test
    # num_itrs = 20

    # medium run
    num_itrs = 2000

    # # big run
    # num_itrs = 20000

    X, Y = {}, {}
    for N in Ns:
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X[N], Y[N] = ltms["X"], ltms["Y"]

    ops = {
        0: ['w', 'x', 'y', '1'], # large N makes overflows, normalizing and no inv anyway
        1: [tr.neg, tr.sign, softform.square, softform.min, softform.max, softform.mean],
        2: [tr.add, tr.sub, tr.mul, tr.maximum, tr.minimum, softform.dotmean, softform.project, softform.reject],
    }

    model = VecRule(ops, max_depth, B, Ns[0], use_ham, logits=True)
    opt = tr.optim.Adam(model.parameters(), lr=lr)

    init_attn = model.sf.attention.detach().clone().numpy()

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
            opt_attn = model.sf.attention.detach().clone().numpy()

            # print(f"{itr} of {num_itrs}", losses[-1], gns[-1], softform.form_str(model.alpha.harden()), softform.form_str(model.beta.harden()))
            print(f"{rep}: {itr} of {num_itrs}", losses[-1], gns[-1], np.fabs(opt_attn).max().item(), softform.form_str(model.sf.harden()))
            tr.save(model, f'softfit_{rep}.pt')
            with open(f'softfit_{rep}_train.pkl', 'wb') as f:
                pk.dump((losses, gns, init_attn, opt_attn), f)

def do_evaluation(rep):

    num_runs = 100

    Ns = list(range(3,8))

    X, Y = {"train": {}, "test": {}}, {"train": {}, "test": {}}
    for N in Ns:
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X['train'][N], Y['train'][N] = ltms["X"], ltms["Y"]

    ltms = np.load("ltms_8_c.npz")
    X['test'][8], Y['test'][8] = ltms["X"], ltms["Y"]

    model = tr.load(f'softfit_{rep}.pt')

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

    loss, accu = {}, {}
    for key in ('train', 'test'):
        loss[key], accu[key] = svm_eval(X[key], Y[key], update_rule, num_runs)
        accu[key] = np.mean(accu[key])
        loss[key] = np.mean(loss[key] * 180/np.pi) # convert to degrees
        print(f"{rep} svm {key} loss={loss[key]}, accu={accu[key]}")

    with open(f'softfit_{rep}_eval.pkl', 'wb') as f:
        pk.dump((loss, accu), f)

if __name__ == "__main__":

    do_train = False
    do_eval = False
    do_show = True
    num_proc = 3
    num_reps = 30

    if do_train:
        with Pool(processes=num_proc) as pool:
            pool.map(do_training_run, range(num_reps))

    if do_eval:
        with Pool(processes=num_proc) as pool:
            pool.map(do_evaluation, range(num_reps))

    if do_show:

        ops = {
            0: ['w', 'x', 'y', '1'], # large N makes overflows, normalizing and no inv anyway
            1: [tr.neg, tr.sign, softform.square, softform.min, softform.max, softform.mean],
            2: [tr.add, tr.sub, tr.mul, tr.maximum, tr.minimum, softform.dotmean, softform.project, softform.reject],
        }

        rep = 0

        all_losses, all_gns = [], []
        for rep in range(num_reps):
            with open(f'nologit/softfit_{rep}_train.pkl', 'rb') as f:
                (losses, gns, init_attn, opt_attn) = pk.load(f)
                all_losses.append(losses)
                all_gns.append(gns)

        all_losses = np.stack(all_losses)
        all_gns = np.stack(all_gns)

        pt.figure(figsize=(4,4))

        pt.subplot(2,1,1)
        pt.plot(all_losses.T, '-', color=(.75,)*3)
        pt.plot(all_losses.mean(axis=0), '-', color='k', label='mean')
        pt.ylabel("Cosine Loss")
        pt.legend()

        pt.subplot(2,1,2)
        pt.plot(all_gns.T, '-', color=(.75,)*3)
        pt.plot(all_gns.mean(axis=0), '-', color='k')
        pt.ylabel("Gradient Norm")
        pt.xlabel("Optimization step")
        
        pt.tight_layout()
        pt.savefig('optimization.pdf')
        pt.show()
        
        # pt.subplot(1,4,1)
        # pt.plot(losses)
    
        # pt.subplot(1,4,2)
        # pt.plot(gns)
    
        # pt.subplot(1,4,3)
        # pt.imshow(init_attn)
        # pt.xticks(range(sum(map(len, ops.values()))), [op if type(op) == str else op.__name__ for n in ops for op in ops[n]], rotation=90)
    
        # pt.subplot(1,4,4)
        # pt.imshow(opt_attn)
        # pt.xticks(range(sum(map(len, ops.values()))), [op if type(op) == str else op.__name__ for n in ops for op in ops[n]], rotation=90)
    
        # pt.show()

        # eval metrics
        accus = {'train': np.empty(num_reps), 'test': np.empty(num_reps)}
        for rep in range(num_reps):
            with open(f'nologit/softfit_{rep}_eval.pkl', 'rb') as f:
                (loss, accu)= pk.load(f)
            for key in ('train', 'test'):
                accus[key][rep] = accu[key]

        print(f"best rep: {np.argmax(accus['test'])}")

        # pt.plot([0]*num_reps, accus['train'], 'r.')
        # pt.plot([1]*num_reps, accus['test'], 'b.')
        # pt.scatter(0, np.mean(accus['train']), 50, color='r')
        # pt.scatter(1, np.mean(accus['test']), 50, color='b')
        # pt.scatter(0, np.mean(accus['train']) - np.std(accus['train']), 50, color='r')
        # pt.scatter(1, np.mean(accus['test']) - np.std(accus['test']), 50, color='b')
        # pt.xticks([0,1], ['train','test'], rotation=90)
        # pt.show()

        pt.figure(figsize=(4,3))
        pt.hist(accus['train'], bins = np.linspace(0, 1.0, 100), align='left', rwidth=0.5, label="N < 8")
        pt.hist(accus['test'], bins = np.linspace(0, 1.0, 100), align='right', rwidth=0.5, label="N = 8")
        pt.xlim([.8, 1.0])
        pt.xlabel("Accuracy")
        pt.ylabel("Frequency")
        pt.legend()
        pt.tight_layout()
        pt.savefig("acc.pdf")
        pt.show()

