import pickle as pk
from geneng_array import *
from graph_fitness import load_ltm_data, organize_by_source, graph_fitness

if __name__ == "__main__":

    Ns = [3,4,5]

    do_span = False # whether to constrain learning rule to be span rule
    do_vert = True # whether to constrain learning rule to be vertex rule

    do_opt = True # whether to run optimization
    do_check = True # whether to check result of optimization
    eps = .01 # minimal slack in region constraints

    # ops imported from geneng_ab
    productions = [Constant, Dimension, Variable, Add, Sub, Mul, Div, Power, Maximum, Minimum, Dot, Sign, Sqrt, Log2, Sum, Min, Max]
    
    if do_span:
        grammar = extract_grammar(productions, SpanRule)
    elif do_vert:
        grammar = extract_grammar(productions, VertexRule)
    else:
        grammar = extract_grammar(productions, VecRule)

    print("Grammar: {}.".format(repr(grammar)))

    Yc, W, X, Ac = load_ltm_data(Ns)

    # set up initial weights for graph search as [0,0,...,0,1]
    # so happens to always be index 1 in the enumerate_ltms canonical dichotomies
    I0 = {N: 1 for N in Ns}
    W0 = {N: W[N][I0[N]] for N in Ns}

    def learning_rule_factory(n: Array):

        def learning_rule(w, x, y):
            # geneng_array line of dataset expects 2D arrays, one row per example, but graph_fitness inputs 1D w and x and scalar y
            line = [w.reshape(1,-1), x.reshape(1,-1), np.array([[y]])]
            w_new = n.eval(line)[0]
            return w_new

        return learning_rule

    def fit_fun(n: Array):

        learning_rule = learning_rule_factory(n)
        region_loss, match_loss, W = graph_fitness(learning_rule, I0, W0, X, Yc, Ac, eps, verbose=False)
        # print(region_loss)

        # region loss should take precedence
        # tanh(matchloss) * eps < eps, so if region loss >= eps (outside of region) it takes precedence
        total_loss = region_loss + eps * np.tanh(match_loss)

        # NaNs might mess up genetic engine, convert to large finite loss
        if not np.isfinite(total_loss): total_loss = 10**6

        return total_loss
    
    prob = SingleObjectiveProblem(
        minimize=True, # smaller loss is better
        fitness_function=fit_fun,
    )

    if do_opt:
        alg = SimpleGP(
            grammar,
            problem=prob,
            # probability_crossover=0.4,
            # probability_mutation=0.4,
            number_of_generations=10000,
            max_depth=8,
            population_size=6000,
            selection_method=("tournament", 2),
            n_elites=1000,
            n_novelties=2000,
            seed=None, #np.random.randint(123456), # 123 for reproducible convergence on perceptron
            target_fitness=-1e-10, # allow small round-off, much less than epsilon
            # favor_less_complex_trees=True,
        )

        best = alg.evolve()


        # fitness, formula = best.get_fitness(prob), best.genotype
        fitness, formula = best.get_fitness(prob), best.genotype

    #     with open(f"graph_best_{'_'.join(map(str, Ns))}.pkl", "wb") as f: pk.dump((fitness, formula), f)

    # with open(f"graph_best_{'_'.join(map(str, Ns))}.pkl", "rb") as f: fitness, formula = pk.load(f)

    print(
        f"Fitness of {fitness} by genotype: {formula}",
    )

    if do_check:

        learning_rule = learning_rule_factory(formula)
        region_loss, match_loss, W = graph_fitness(learning_rule, I0, W0, X, Yc, Ac, eps, verbose=False)
        print(f"eps={eps}, region loss = {region_loss}, match loss = {match_loss}")
        for N in Ns:
            print(f"N={N}, W[N]:")
            print(W[N])

            A = organize_by_source(Ac[N])

            # repeatedly run a bunch of random transitions
            R = 10 # repetitions
            T = 100 # transitions
            for r in range(R):

                # start in random dichotomy
                i = np.random.randint(W[N].shape[0])
                w = W[N][i]

                # stream of transitions
                for t in range(T):
    
                    # check current region
                    assert ((w * (X[N] * Yc[N][i]).T).sum(axis=1) > 0).all()
    
                    # transition to random new region
                    j, k = A[i][np.random.randint(len(A[i]))]
                    w = learning_rule(w, X[N][:,k], Yc[N][j,k])
                    i = j

        print("check passed.")

