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

def load_examples(Ns):

    examples = []
    optimal_loss = 0.
    for N in Ns:
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

        # calculate optimal loss when all cosine sims are 1.0
        sinmarg = tr.cos(mdenoms)
        optimal_loss += -(1.0 / sinmarg).mean().item()

        inputs = tr.stack([w_old, x, y, _1])
        examples.append((inputs, (w_new, mdenoms)))

    optimal_loss /= len(examples)

    return examples, optimal_loss

def do_training_run(rep):

    # make sure different runs have different random numbers
    tr.manual_seed(tr.initial_seed() + 100*rep)
    rng = np.random.default_rng()

    use_softmax = False
    use_noam = True
    use_simplex_clipping = True # don't use with softmax

    B = 16
    max_depth = 5

    lr = 0.1 # no schedule

    if use_noam:
        # sched = NoamScheduler(base_lr=0.005, warmup=5000) # softmax
        sched = NoamScheduler(base_lr=.5, warmup=500) # clipping
    else:
        sched = NoScheduler()

    # # quick test
    # num_itrs = 100

    # medium run
    num_itrs = 1000

    # # big run
    # num_itrs = 30000

    examples, optimal_loss = load_examples(list(range(3,5)))

    if use_noam:
        model = VecRule(max_depth, logits=use_softmax)
    else:
        model = VecRule(max_depth, logits=use_softmax, init_scale = 0.01)

    if use_simplex_clipping:
        # initialize weights within the attention simplices
        for attn in (model.sf.inners_attn, model.sf.leaves_attn):
            attn.data = attn.data.abs()
            attn.data /= attn.data.sum(dim=1, keepdim=True)

    # opt = tr.optim.Adam(model.parameters(), lr=lr)
    opt = tr.optim.SGD(model.parameters(), lr=lr)

    init_attn = model.sf.inners_attn.detach().clone().numpy()

    losses, gns = [], []
    last_gvec = None
    for itr in range(num_itrs):

        # collect loss over transitions
        total_loss = 0.
        for (inputs, (w_new, mdenom)) in examples:
            w_pred = model(inputs)

            # # orig
            # loss = -(w_new*w_pred).sum(dim=1).mean() / len(examples) # already normalized

            # # angles, make nan grads maybe arccos(1)?
            # gamma = tr.acos(tr.clamp((w_new*w_pred).sum(dim=1), -1., 1.)) # already normalized
            # loss = (gamma / mdenom).mean() / len(examples)

            # trigs
            cosgam = (w_new*w_pred).sum(dim=1) # already normalized
            sinmarg = tr.cos(mdenom)
            loss = -(cosgam / sinmarg).mean() / len(examples)

            loss.backward()
            total_loss += loss.item()

        losses.append(total_loss)

        # if itr % 5000 == 0:
        #     pt.subplot(1,3,1)
        #     pt.imshow(init_attn)
        #     pt.subplot(1,3,2)
        #     pt.imshow(model.sf.inners_attn.detach().numpy())
        #     pt.subplot(1,3,3)
        #     pt.imshow(model.sf.leaves_attn.detach().numpy())
        #     pt.show()

        if use_simplex_clipping:
            clipped = []
            gradscales = []
            for attn in (model.sf.inners_attn, model.sf.leaves_attn):
                # project away grad components orthogonal to attention simplices
                attn.grad -= attn.grad.mean(dim=1, keepdim=True)
                # clip negative grads to limits of attention simplices
                # # these were clipping in positive grad direction!
                # minpos = tr.min(tr.where(attn.grad > 0, (1 - attn.data) / attn.grad, 1), dim=1, keepdim=True).values
                # minneg = tr.min(tr.where(attn.grad < 0, (  - attn.data) / attn.grad, 1), dim=1, keepdim=True).values
                minpos = tr.min(tr.where(attn.grad > 0, (attn.data    ) / attn.grad, 1), dim=1, keepdim=True).values
                minneg = tr.min(tr.where(attn.grad < 0, (attn.data - 1) / attn.grad, 1), dim=1, keepdim=True).values
                grad_scale = tr.minimum(tr.minimum(minpos, minneg), tr.ones(minpos.shape))
                attn.grad *= grad_scale
                clipped += (grad_scale != 1.).tolist()
                gradscales += grad_scale.tolist()
            clipped = np.mean(clipped)
            gradscale = np.mean(gradscales)

        gvec = tr.cat(tuple(attn.grad.data.flatten() for attn in (model.sf.inners_attn, model.sf.leaves_attn)), dim=0)
        gn = ((gvec**2).mean()**.5).item()
        gns.append(gn)

        if last_gvec is None:
            pang = 0.
        else:
            pang = tr.acos(tr.clamp(tr.nn.functional.cosine_similarity(gvec, last_gvec, dim=0), -1., 1.)).item() * 180 / tr.pi
        last_gvec = gvec.clone().detach()

        sched.apply_lr(opt)
        sched.step()
        opt.step()
        opt.zero_grad()

        if use_simplex_clipping:
            # remove small round-off errors outside the simplex
            for attn in (model.sf.inners_attn, model.sf.leaves_attn):
                # if itr % 100 == 0:
                #     print(attn.min(), attn.max(), attn.sum(dim=-1).min(), attn.sum(dim=-1).max())
                attn.data = tr.clamp(attn.data, 0., 1.)
                attn.data /= attn.data.sum(dim=1, keepdim=True)

        if itr % 100 == 0 or itr+1 == num_itrs:
            # if use_softmax:
            #     opt_attn = tr.softmax(model.sf.inners_attn.detach().clone(), dim=1).numpy()
            # else:
            #     opt_attn = model.sf.inners_attn.detach().clone().numpy()
            opt_attn = model.sf.inners_attn.detach().clone().numpy()

            print(f"{rep}: {itr} of {num_itrs} loss={total_loss:.3f} vs {optimal_loss:.3f},",
                  f"grad=clipped {clipped:.3f}, scaled~{gradscale:.3f} -> rms={gn:.3f}, ang={pang:.3f},",
                  f"attn~{np.fabs(opt_attn).max(axis=1).mean():.3f}", sfd.form_str(model.sf.harden()))
            tr.save(model, f'sfd_{rep}.pt')
            with open(f'sfd_{rep}_train.pkl', 'wb') as f:
                pk.dump((losses, gns, init_attn, opt_attn, optimal_loss), f)

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

    do_train = True
    do_eval = True
    do_show = True
    num_proc = 4
    num_reps = 4

    if do_train:
        with Pool(processes=num_proc) as pool:
            pool.map(do_training_run, range(num_reps))

    if do_eval:
        with Pool(processes=num_proc) as pool:
            pool.map(do_evaluation, range(num_reps))

    if do_show:

        all_losses, all_gns = [], []
        for rep in range(num_reps):
            with open(f'sfd_{rep}_train.pkl', 'rb') as f:
                (losses, gns, init_attn, opt_attn, optimal_loss) = pk.load(f)
                all_losses.append(losses)
                all_gns.append(gns)

        all_losses = np.stack(all_losses)
        all_gns = np.stack(all_gns)

        pt.figure(figsize=(12,8))

        pt.subplot(3,1,1)
        pt.plot(all_losses.T, '-', color=(.75,)*3)
        pt.plot(all_losses.mean(axis=0), '-', color='k', label='mean')
        pt.plot([0, all_losses.shape[1]], [optimal_loss]*2, 'k:', label='optimal')
        pt.ylabel("Loss")
        pt.legend()

        pt.subplot(3,1,2)
        pt.plot(all_losses.T - optimal_loss, '-', color=(.75,)*3)
        pt.plot(all_losses.mean(axis=0) - optimal_loss, '-', color='k', label='mean')
        pt.ylabel("Loss")
        pt.yscale('log')
        pt.legend()

        pt.subplot(3,1,3)
        pt.plot(all_gns.T, '-', color=(.75,)*3)
        pt.plot(all_gns.mean(axis=0), '-', color='k')
        pt.ylabel("Gradient Norm")
        pt.xlabel("Optimization step")
        
        pt.tight_layout()
        pt.savefig('sfd_optimization.pdf')
        pt.show()

        pt.figure(figsize=(8,6))

        pt.subplot(1,2,1)
        pt.imshow(init_attn)
        pt.xticks(range(len(sfd.OPS)), [op if type(op) == str else op.__name__ for op in sfd.OPS], rotation=90)
        pt.colorbar()
    
        pt.subplot(1,2,2)
        pt.imshow(opt_attn)
        pt.xticks(range(len(sfd.OPS)), [op if type(op) == str else op.__name__ for op in sfd.OPS], rotation=90)
        pt.colorbar()

        pt.tight_layout()
        pt.savefig('sfd_attn.pdf')
        pt.show()

        # eval metrics
        # accus = {'train': np.empty(num_reps), 'test': np.empty(num_reps)}
        accus = {}
        for rep in range(num_reps):
            with open(f'sfd_{rep}_eval.pkl', 'rb') as f:
                (loss, accu) = pk.load(f)
            for key in accu.keys():
                if key not in accus: accus[key] = np.empty(num_reps)
                accus[key][rep] = accu[key]

        print(f"best rep: {np.argmax(accus[8])}")

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
        # for key, vals in accus.items():
        #     pt.hist(vals, label=f"N = {key}")
        # pt.xlabel("Accuracy")
        # pt.ylabel("Frequency")
        # pt.legend()
        for key, vals in accus.items():
            pt.plot([key]*len(vals), vals, 'k.')
        pt.xlabel("N")
        pt.ylabel("Accuracy")
        pt.tight_layout()
        pt.savefig("sfd_acc.pdf")
        pt.show()

