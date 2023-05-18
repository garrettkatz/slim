import pickle as pk
import itertools as it
import numpy as np
from scipy.special import factorial

if __name__ == "__main__":

    do_exp = True # whether to run the perceptron training or just load the results

    Ns = np.arange(3, 9)

    if do_exp:

        # track number of iterations and mistakes
        all_num_iters = []
        all_mistakes = []

        # process one N at a time    
        for N in Ns:
            print(f"N = {N}")
    
            # load all canonical hemichotomies
            ltms = np.load(f"ltms_{N}_c.npz")
            Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

            # default learning rate of 1
            # for vanilla perceptron, constant learning rate should not change convergence time, only final weight norm
            #     https://datascience.stackexchange.com/questions/16843/perceptron-learning-rate
            eta = 1

            # track iterations and mistakes on each hemichotomy        
            num_iters = np.empty(Y.shape[0], dtype=int)
            num_mistake_presentations = np.zeros(Y.shape, dtype=int)

            # separate training runs on one hemichotomy at a time
            for i, y in enumerate(Y):

                # initial zero weight vector
                w = np.zeros(N)        

                # keep updating to convergence
                for k in it.count():
            
                    # stop as soon as every example is fit correctly
                    if (np.sign(w @ X) == y).all(): break
                    
                    # randomize order of training examples when every epoch begins
                    if k % X.shape[1] == 0:
                        samples = np.random.permutation(np.arange(X.shape[1]))

                    # get the next training example in randomized order
                    j = samples[k % X.shape[1]]

                    # no weight update if current example fit correctly
                    if np.sign(w @ X[:,j]) == y[j]: continue
            
                    # track number of times mistakes were made on each sample
                    num_mistake_presentations[i, j] += 1

                    # apply weight update by perceptron learning rule
                    w = w + eta * y[j] * X[:,j]

                # track total number of iterations until convergence
                num_iters[i] = k
                print(f"hemi {i} of {len(Y)}: {k} vs {X.shape[1]} iterations, {num_mistake_presentations[i].sum()} mistakes")

            # print maximum iterations over all hemichotomies, store results before next N
            print(f"max iters = {num_iters.max()}")
            all_num_iters.append(num_iters)
            all_mistakes.append(num_mistake_presentations.sum(axis=1))
        
        # save all results
        with open("perbase.pkl", "wb") as f:
            pk.dump((all_num_iters, all_mistakes), f)

    # calculate weighted average weights
    syms = []
    for N in Ns:
        print(f"N = {N}")

        ltms = np.load(f"ltms_{N}_c.npz")
        W = ltms["W"]
        sym = np.zeros(len(W))
        for i, w in enumerate(W.round()):
            # get multiset coefficient
            uni = {}
            for n in range(N): uni[w[n]] = uni.get(w[n], 0) + 1
            sym[i] = factorial(sum(uni.values()))
            for v in uni.values(): sym[i] /= factorial(v)
            sym[i] *= 2**np.count_nonzero(w)        

        sym /= sym.sum()
        syms.append(sym)

    with open("perbase.pkl", "rb") as f:
        (all_num_iters, all_mistakes) = pk.load(f)

    all_num_epochs = [num_iters / 2.**(N-1) for (N, num_iters) in zip(Ns, all_num_iters)]

    import matplotlib.pyplot as pt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['text.usetex'] = True

    pt.figure(figsize=(2.25,1.8))

    # pt.subplot(1,2,1)
    avg_iters = np.array([np.average(num_epochs, weights=sym) for sym, num_epochs in zip(syms, all_num_epochs)])
    # std_iters = np.array([num_epochs.std() for num_epochs in all_num_epochs])
    std_iters = np.array([np.sqrt(np.cov(num_epochs, aweights=sym)) for sym, num_epochs in zip(syms, all_num_epochs)])
    pt.fill_between(Ns, avg_iters-std_iters, avg_iters+std_iters, color=(.8,)*3)

    for n, N in enumerate(Ns):
        num_epochs = all_num_epochs[n]
        # pt.plot([N]*len(num_epochs), num_epochs, '.', color=(.4,)*3)
        pt.plot(N +  + np.random.randn(len(num_epochs))*0.05, num_epochs, '.', color=(.4,)*3)

    pt.plot(Ns, avg_iters, 'ko-')

    # pt.ylabel("Num Iters")
    pt.ylabel("Epochs to convergence")
    pt.xlabel("Input dimension $N$")
    pt.xticks(Ns, Ns)
    pt.yscale('log')
    pt.yticks(10.**np.arange(3), ["$10^{%d}$" % k for k in range(3)])
    # pt.gca().yaxis.get_major_locator().set_params(numticks=3)#, subs=[.2, .4, .6, .8])
    pt.gca().yaxis.get_minor_locator().set_params(numticks=10)#, subs=[.2, .4, .6, .8])

    # pt.subplot(1,2,2)
    # for n, N in enumerate(Ns):
    #     # num_mistakes = all_mistakes[n]
    #     num_mistakes = all_mistakes[n] / 2**(N-1) # per sample
    #     pt.plot([N]*len(num_mistakes), num_mistakes, '.', color=(.5,)*3)

    # avg_mistakes = [num_mistakes.mean() for num_mistakes in all_mistakes]
    # std_mistakes = [num_mistakes.std() for num_mistakes in all_mistakes]
    # pt.plot(Ns, avg_mistakes, 'ko-')

    # pt.ylabel("Num Mistakes")
    # pt.xlabel("N")
    # pt.yscale('log', base=2)

    pt.tight_layout()
    pt.savefig('perbase.pdf')
    pt.show()

