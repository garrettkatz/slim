from multiprocessing import Pool
import pickle as pk
import numpy as np
import torch as tr
import softform_dense as sfd
import hardform_dense as hfd
from svm_eval import svm_eval
from softfit_dense import load_examples

# constrain formulas to be span rules
# idx[i, :] for individual i
def spanify(idx):
    num_nodes = idx.shape[-1]
    max_depth = round(np.log2(num_nodes + 1)) - 1
    num_inner = 2**max_depth - 1

    idx[:,         0] = 14 # add(
    idx[:,         1] = 16 #     mul(
    idx[:, num_inner] =  0 #         w,
    idx[:,         4] = 12 #         mean(_))
    idx[:,         2] = 16 #     mul(
    idx[:,         5] = 12 #         mean(_),
    idx[:,        -1] =  1 #         x)

    left_path = 2**tr.arange(max_depth+1) - 1
    idx[:, left_path[2:-1]] = 0 # idleft
    idx[:, left_path[3:]-1] = 1 # idright

    return idx

def do_evaluation(rep):

    # make sure different runs have different random numbers
    seed = tr.initial_seed() + 100*rep
    tr.manual_seed(seed)
    np.random.seed(seed % (2**32))

    num_runs = 100

    Ns = list(range(3,9))
    X, Y = {}, {}
    for N in Ns:
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X[N], Y[N] = ltms["X"], ltms["Y"]

    with open("genprog.pkl", "rb") as f:
        (tree, fitness) = pk.load(f)

    def update_rule(w, x, y, N):

        inputs = tr.stack([
            tr.nn.functional.normalize(tr.tensor(w, dtype=tr.float32).view(1,N)), # w
            tr.tensor(x, dtype=tr.float32).view(1,N), # x
            tr.tensor(y, dtype=tr.float32).view(1,1).expand(1, N), # y
            tr.ones(1,N), # 1
        ])
        w_new = hfd.tree_eval(tree, inputs)
        w_new = tr.nn.functional.normalize(w_new)
        return w_new.clone().squeeze().numpy()

    loss, accu = {}, {}
    for N in Ns:
        loss[N], accu[N] = svm_eval({N: X[N]}, {N: Y[N]}, update_rule, num_runs)
        accu[N] = np.mean(accu[N])
        loss[N] = np.mean(loss[N] * 180/np.pi) # convert to degrees
        print(f"{rep} svm {N} loss={loss[N]}, accu={accu[N]}")

    with open(f'gp_eval_{rep}.pkl', 'wb') as f:
        pk.dump((loss, accu), f)

if __name__ == "__main__":

    do_evo = True
    do_eval = True
    do_show = True

    do_span = True

    # just for svm eval
    num_proc = 6
    num_reps = 30

    examples, optimal_loss = load_examples(list(range(3,5)))
    ideal_fitness = -optimal_loss

    num_gens = 1000
    pop_size = 200
    top_size = 40
    mutation_rate = 0.1
    max_depth = 6

    num_inner = 2**max_depth - 1
    num_leaves = 2**max_depth
    num_nodes = num_inner + num_leaves

    # initialize coefficients for n in node n's descendent tree
    n_coef = 2**tr.log2(tr.arange(num_nodes) + 1).to(int)

    if do_evo:

        # initialize population
        inners_idx = tr.randint(len(sfd.OPS), size=(pop_size, num_inner))
        leaves_idx = tr.randint(4, size=(pop_size, num_leaves))
        idx = tr.cat((inners_idx, leaves_idx), dim=1)
        if do_span: idx = spanify(idx)

        fitness = tr.zeros(num_gens, pop_size)
        for gen in range(num_gens):
    
            # fitness evaluation
            for ind in range(pop_size):
                inners, leaves = idx[ind, :num_inner], idx[ind, num_inner:]
                tree = hfd.to_tree(inners, leaves, sfd.OPS)
    
                # collect loss over transitions
                for (inputs, (w_new, mdenom)) in examples:
                    w_pred = hfd.tree_eval(tree, inputs)
                    w_pred = tr.nn.functional.normalize(w_pred)
    
                    cosgam = (w_new*w_pred).sum(dim=1) # already normalized
                    sinmarg = tr.cos(mdenom)
                    fitness[gen, ind] += (cosgam / sinmarg).mean() / len(examples)
    
            # identify parent pool
            top = tr.topk(fitness[gen], top_size).indices
    
            # sample parents and crossover points
            par_idx = tr.randint(top_size, size=(pop_size, 2))
            parents = top[par_idx]
            crossover = tr.randint(num_nodes, size=(pop_size,))
    
            # build new population
            new_idx = tr.empty_like(idx)
            for ind in range(pop_size):
    
                # get crossover point descendents
                descendents = n_coef * crossover[ind] + tr.arange(len(n_coef))
                descendents = descendents[descendents < num_nodes]
    
                # crossover parents
                new_idx[ind] = idx[parents[ind, 0]]
                new_idx[ind, descendents] = idx[parents[ind, 1], descendents]
    
                # mutate
                mutations = (tr.rand(num_nodes) < mutation_rate)
                new_idx[ind, mutations] = tr.randint(len(sfd.OPS), size=(mutations.sum(),))
                new_idx[ind, num_inner:] %= 4 # leaf nodes
    
            # keep the best, replace the rest
            new_idx[top[0]] = idx[top[0]]
            idx = new_idx
            if do_span: idx = spanify(idx)
    
            best_idx = idx[top[0]]
            inners, leaves = best_idx[:num_inner], best_idx[num_inner:]
            tree = hfd.to_tree(inners, leaves, sfd.OPS)
    
            print(f"gen {gen} of {num_gens}: fitness ~ {fitness[gen].mean().item():.3f} <= {fitness[gen].max().item():.3f} vs {ideal_fitness:.3f},",
                  f"best: {sfd.form_str(tree)}")
    
            with open("genprog.pkl", "wb") as f:
                pk.dump((tree, fitness), f)

    if do_eval:

        with Pool(processes=num_proc) as pool:
            pool.map(do_evaluation, range(num_reps))

    if do_show:

        import matplotlib.pyplot as pt

        with open("genprog.pkl", "rb") as f:
            (tree, fitness) = pk.load(f)

        print(f"best: {sfd.form_str(tree)}")

        idx = np.arange(0, fitness.shape[0], 20)
        pt.figure(figsize=(8,3))
        pt.plot(idx, fitness.numpy()[idx], '.', color=(0,0,0,.1))
        pt.plot(idx, fitness.numpy()[idx].mean(axis=1), 'k-')
        pt.ylabel("Fitness")
        pt.xlabel("Generation")
        pt.tight_layout()
        pt.savefig("gp_opt.pdf")
        pt.savefig("gp_opt.png")
        pt.show()

        accus = {}
        for rep in range(num_reps):
            with open(f'gp_eval_{rep}.pkl', 'rb') as f:
                (loss, accu) = pk.load(f)
            for key in accu.keys():
                if key not in accus: accus[key] = np.empty(num_reps)
                accus[key][rep] = accu[key]

        pt.figure(figsize=(4,3))
        for key, vals in accus.items():
            pt.plot([key]*len(vals), vals, 'k.')
        pt.xlabel("N")
        pt.ylabel("Accuracy")
        pt.tight_layout()
        pt.savefig("gp_acc.pdf")
        pt.show()
