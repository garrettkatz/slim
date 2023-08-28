from multiprocessing import Pool
import pickle as pk
import numpy as np
import torch as tr
import softform_dense as sfd
import hardform_dense as hfd
from svm_eval import svm_eval
from softfit_dense import load_examples
from genprog import spanify

if __name__ == "__main__":

    do_rand = True
    do_eval = True
    do_show = True

    do_span = False

    # just for svm eval
    num_proc = 6
    num_reps = 30

    examples, optimal_loss = load_examples(list(range(3,5)))
    ideal_fitness = -optimal_loss

    num_gens = 1000
    pop_size = 200
    top = 
    max_depth = 6

    num_inner = 2**max_depth - 1
    num_leaves = 2**max_depth
    num_nodes = num_inner + num_leaves

    # sample random population
    inners_idx = tr.randint(len(sfd.OPS), size=(pop_size, num_inner))
    leaves_idx = tr.randint(4, size=(pop_size, num_leaves))
    idx = tr.cat((inners_idx, leaves_idx), dim=1)
    if do_span: idx = spanify(idx)

    if do_rand:

        best_fit = None
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

            # keep the best
            best = fitness[gen].argmax()
            if best_fit is None or fitness[gen, best] > best_fit:
                best_fit = fitness[gen, best]
                best_idx = idx[best]
                inners, leaves = best_idx[:num_inner], best_idx[num_inner:]
                best_tree = hfd.to_tree(inners, leaves, sfd.OPS)

            print(f"gen {gen} of {num_gens}: fitness ~ {fitness[gen].max().item():.3f} <= {best_fit:.3f} vs {ideal_fitness:.3f},",
                  f"best: {sfd.form_str(best_tree)}")
    
            with open("randform.pkl", "wb") as f:
                pk.dump((best_tree, fitness), f)


