import sys
import pickle as pk
import itertools as it
import numpy as np
from scipy.special import factorial
from enumerate_ltms import get_equivalence_class_size

if __name__ == "__main__":

    do_exp = True # whether to run the perceptron training or just load the results

    # experiment up to dimension N_max
    if len(sys.argv) > 1:
        N_max = int(sys.argv[1])
    else:
        N_max = 8
    Ns = list(range(3, N_max + 1))

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
                    w = w + eta * (y[j] - np.sign(w @ X[:,j])) * X[:,j] # Perceptron learning

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

    # load all results
    with open("perbase.pkl", "rb") as f:
        (all_num_iters, all_mistakes) = pk.load(f)

    # convert iters to epochs, 2**(N-1) iters per
    all_num_epochs = [num_iters / 2**(N-1) for (num_iters, N) in zip(all_num_iters, Ns)]

    # get average/stdev metrics for each N, weighted by equivalence class sizes
    avg_iters, std_iters = np.empty(len(Ns)), np.empty(len(Ns))
    for n, N in enumerate(Ns):

        # load all canonical hemichotomies
        ltms = np.load(f"ltms_{N}_c.npz")
        W = ltms["W"]

        # calculate weights for each equivalence class
        ecw = np.array(list(map(get_equivalence_class_size, W)))
        ecw /= ecw.sum()

        # calculate stats
        num_epochs = all_num_epochs[n]
        avg_iters[n] = np.average(num_epochs, weights=ecw)
        std_iters[n] = np.sqrt(np.cov(num_epochs, aweights=ecw))

    # plot results
    import matplotlib.pyplot as pt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['text.usetex'] = True

    pt.figure(figsize=(2.25,1.8))

    # stdev envelope background
    pt.fill_between(Ns, avg_iters-std_iters, avg_iters+std_iters, color=(.8,)*3)

    # individual hemichotomy points
    for (num_epochs, N) in zip(all_num_epochs, Ns):
        pt.plot(N +  + np.random.randn(len(num_epochs))*0.05, num_epochs, '.', color=(.4,)*3)

    # averages
    pt.plot(Ns, avg_iters, 'ko-')

    # format axes
    pt.ylabel("Epochs to convergence")
    pt.xlabel("Input dimension $N$")
    pt.xticks(Ns, Ns)
    pt.yscale('log')
    pt.yticks(10.**np.arange(3), ["$10^{%d}$" % k for k in range(3)])
    pt.gca().yaxis.get_minor_locator().set_params(numticks=10)

    # save and show
    pt.savefig('perbase.pdf')
    pt.show()

