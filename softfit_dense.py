import matplotlib.pyplot as pt
import pickle as pk
from multiprocessing import Pool
import numpy as np
import torch as tr
import softform_dense as sfd
from svm_data import random_transitions, all_transitions
from svm_eval import svm_eval
from noam import NoamScheduler, NoScheduler

class VecRule(tr.nn.Module):
    def __init__(self, max_depth, logits=True, init_scale=None):
        super().__init__()
        self.sf = sfd.SoftFormDense(max_depth, logits, init_scale)

    def forward(self, inputs):
        # apply soft formula and normalize
        w_pred = self.sf(inputs)
        w_pred = tr.nn.functional.normalize(w_pred)
        return w_pred

def do_training_run(rep):

    # make sure different runs have different random numbers
    tr.manual_seed(tr.initial_seed() + 100*rep)
    rng = np.random.default_rng()

    use_softmax = True
    use_noam = True

    Ns = list(range(3,8))

    B = 16
    max_depth = 5

    lr = 0.0005 # no schedule

    if use_noam:
        sched = NoamScheduler(base_lr=0.03, warmup=5000)
    else:
        sched = NoScheduler()

    # # quick test
    # num_itrs = 100

    # # medium run
    # num_itrs = 5000

    # big run
    num_itrs = 50000

    examples = []
    for N in range(3,5):
        # get all transitions
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        W, X, Y = ltms["W"], {N: ltms["X"]}, {N: ltms["Y"]}
        w_new, w_old, x, y, margins = all_transitions(X, Y, N)

        # package as tensors
        w_new = tr.tensor(np.concatenate(w_new, axis=0), dtype=tr.float32)
        w_old = tr.tensor(np.concatenate(w_old, axis=0), dtype=tr.float32)
        x = tr.tensor(np.stack(x), dtype=tr.float32)
        y = tr.tensor(y, dtype=tr.float32).view(-1,1).expand(*x.shape) # broadcast
        _1 = tr.ones(*x.shape)
        margins = tr.tensor(np.stack(margins))
        mdenoms = 0.5*tr.pi - tr.acos(tr.clamp(margins, -1., 1.))

        inputs = tr.stack([w_old, x, y, _1])
        examples.append((inputs, (w_new, mdenoms)))

    if use_noam:
        model = VecRule(max_depth, logits=use_softmax)
    else:
        model = VecRule(max_depth, logits=use_softmax, init_scale = 0.01)
    opt = tr.optim.Adam(model.parameters(), lr=lr)

    init_attn = model.sf.inners_attn.detach().clone().numpy()

    losses, gns = [], []
    for itr in range(num_itrs):

        # collect loss over transitions
        total_loss = 0.
        total_perfection = 0.
        for (inputs, (w_new, mdenom)) in examples:
            w_pred = model(inputs)

            # # orig
            # loss = -(w_new*w_pred).sum(dim=1).mean() / len(examples) # already normalized

            # # angles, make nan grads maybe arccos(1)?
            # gamma = tr.acos(tr.clamp((w_new*w_pred).sum(dim=1), -1., 1.)) # already normalized
            # loss = (gamma / mdenom).mean() / len(examples)

            # trigs
            cosgam = (w_new*w_pred).sum(dim=1) # already normalized
            sinthe = tr.cos(mdenom)
            loss = -(cosgam / sinthe).mean() / len(examples)
            perfection = -(1.0 / sinthe).mean() / len(examples)

            loss.backward()
            total_loss += loss.item()
            total_perfection += perfection.item()

        losses.append(total_loss)

        # if itr % 5000 == 0:
        #     pt.subplot(1,3,1)
        #     pt.imshow(init_attn)
        #     pt.subplot(1,3,2)
        #     pt.imshow(model.sf.inners_attn.detach().numpy())
        #     pt.subplot(1,3,3)
        #     pt.imshow(model.sf.leaves_attn.detach().numpy())
        #     pt.show()

        gn = tr.linalg.vector_norm(tr.nn.utils.parameters_to_vector(model.parameters()))
        gns.append(gn.item())

        sched.apply_lr(opt)
        sched.step()
        opt.step()
        opt.zero_grad()

        if itr % 100 == 0 or itr+1 == num_itrs:
            if use_softmax:
                opt_attn = tr.softmax(model.sf.inners_attn.detach().clone(), dim=1).numpy()
            else:
                opt_attn = model.sf.inners_attn.detach().clone().numpy()

            print(f"{rep}: {itr} of {num_itrs} loss={total_loss} vs {total_perfection},", gns[-1], np.fabs(opt_attn).max(axis=1).mean(), sfd.form_str(model.sf.harden()))
            tr.save(model, f'sfd_{rep}.pt')
            with open(f'sfd_{rep}_train.pkl', 'wb') as f:
                pk.dump((losses, gns, init_attn, opt_attn), f)

def do_evaluation(rep):

    num_runs = 100

    Ns = list(range(3,9))
    X, Y = {}, {}
    for N in Ns:
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X[N], Y[N] = ltms["X"], ltms["Y"]

    model = tr.load(f'sfd_{rep}.pt')

    def update_rule(w, x, y, N):

        inputs = tr.stack([
            tr.nn.functional.normalize(tr.tensor(w, dtype=tr.float32).view(1,N)), # w
            tr.tensor(x, dtype=tr.float32).view(1,N), # x
            tr.tensor(y, dtype=tr.float32).view(1,1).expand(1, N), # y
            tr.ones(1,N), # 1
        ])
        with tr.no_grad():
            w_new = model(inputs)
        return w_new.clone().squeeze().numpy()

    loss, accu = {}, {}
    for N in Ns:
        loss[N], accu[N] = svm_eval({N: X[N]}, {N: Y[N]}, update_rule, num_runs)
        accu[N] = np.mean(accu[N])
        loss[N] = np.mean(loss[N] * 180/np.pi) # convert to degrees
        print(f"{rep} svm {N} loss={loss[N]}, accu={accu[N]}")

    with open(f'sfd_{rep}_eval.pkl', 'wb') as f:
        pk.dump((loss, accu), f)

if __name__ == "__main__":

    do_train = False
    do_eval = False
    do_show = True
    num_proc = 2
    num_reps = 2

    if do_train:
        with Pool(processes=num_proc) as pool:
            pool.map(do_training_run, range(num_reps))

    if do_eval:
        with Pool(processes=num_proc) as pool:
            pool.map(do_evaluation, range(num_reps))

    if do_show:

        rep = 0

        all_losses, all_gns = [], []
        for rep in range(num_reps):
            with open(f'sfd_{rep}_train.pkl', 'rb') as f:
                (losses, gns, init_attn, opt_attn) = pk.load(f)
                all_losses.append(losses)
                all_gns.append(gns)

        all_losses = np.stack(all_losses)
        all_gns = np.stack(all_gns)

        pt.figure(figsize=(4,4))

        pt.subplot(2,1,1)
        all_losses -= all_losses.min() # if log scale
        pt.plot(all_losses.T, '-', color=(.75,)*3)
        pt.plot(all_losses.mean(axis=0), '-', color='k', label='mean')
        pt.ylabel("Loss")
        pt.yscale('log')
        pt.legend()

        pt.subplot(2,1,2)
        pt.plot(all_gns.T, '-', color=(.75,)*3)
        pt.plot(all_gns.mean(axis=0), '-', color='k')
        pt.ylabel("Gradient Norm")
        pt.xlabel("Optimization step")
        
        pt.tight_layout()
        pt.savefig('sfd_optimization.pdf')
        pt.show()
        
        # pt.subplot(1,4,1)
        # pt.plot(losses)
    
        # pt.subplot(1,4,2)
        # pt.plot(gns)
    
        # pt.subplot(1,4,3)
        # pt.imshow(init_attn)
        # pt.xticks(range(sum(map(len, sfd.OPS.values()))), [op if type(op) == str else op.__name__ for n in ops for op in ops[n]], rotation=90)
    
        # pt.subplot(1,4,4)
        # pt.imshow(opt_attn)
        # pt.xticks(range(sum(map(len, sfd.OPS.values()))), [op if type(op) == str else op.__name__ for n in ops for op in ops[n]], rotation=90)
    
        # pt.show()

        # eval metrics
        # accus = {'train': np.empty(num_reps), 'test': np.empty(num_reps)}
        accus = {}
        for rep in range(num_reps):
            with open(f'sfd_{rep}_eval.pkl', 'rb') as f:
                (loss, accu) = pk.load(f)
            for key in accu.keys():
                if len(accus) == 0: accus[key] = np.empty(num_reps)
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
        # pt.hist(accus['train'], bins = np.linspace(0, 1.0, 100), align='left', rwidth=0.5, label="N < 8")
        # pt.hist(accus['test'], bins = np.linspace(0, 1.0, 100), align='right', rwidth=0.5, label="N = 8")
        # pt.xlim([.8, 1.0])
        for key, vals in accus.items():
            pt.hist(vals, label=f"N = {key}")
        pt.xlabel("Accuracy")
        pt.ylabel("Frequency")
        pt.legend()
        pt.tight_layout()
        pt.savefig("acc.pdf")
        pt.show()

