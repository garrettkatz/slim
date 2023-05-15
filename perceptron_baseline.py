import pickle as pk
import itertools as it
import numpy as np
from scipy.special import factorial

if __name__ == "__main__":

    do_exp = False

    Ns = np.arange(3, 9)

    if do_exp:

        all_num_iters = []
        all_mistakes = []
    
        for N in Ns:
            print(f"N = {N}")
    
            ltms = np.load(f"ltms_{N}_c.npz")
            Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
        
            eta = 1
        
            num_iters = np.empty(Y.shape[0], dtype=int)
            num_presentations = np.zeros(Y.shape, dtype=int)
            num_mistake_presentations = np.zeros(Y.shape, dtype=int)
        
            for i in range(Y.shape[0]):
        
                w = np.zeros(N)
        
                for k in it.count():
            
                    if (np.sign(w @ X) == Y[i]).all(): break
            
                    # print(f"{k}: {(np.sign(w @ X) == Y[i]).sum()} of {X.shape[1]} correct")
            
                    # j = np.random.randint(X.shape[1]) # random sampling with replacement
        
                    # j = np.argmax(np.sign(w @ X) != Y[i]) # incorrect sample oracle
        
                    # j = k % X.shape[1] # cyclic over samples
        
                    # cyclic over shuffled samples
                    if k % X.shape[1] == 0:
                        samples = np.random.permutation(np.arange(X.shape[1]))
                    j = samples[k % X.shape[1]]
            
                    num_presentations[i, j] += 1
            
                    if np.sign(w @ X[:,j]) == Y[i,j]: continue
            
                    num_mistake_presentations[i, j] += 1
            
                    w = w + eta * Y[i,j] * X[:,j]
        
                num_iters[i] = k
                print(f"ltm {i} of {Y.shape[0]}: {k} vs {X.shape[1]} iterations, {num_mistake_presentations[i].sum()} mistakes")
            
                # print(f"ltm {i}: {k} iterations total.  num presentations (sums to {num_presentations.sum()}):")
                # print(num_presentations)
                # print(f"num mistake presentations (sums to {num_mistake_presentations.sum()}):")
                # print(num_mistake_presentations)
        
            print(f"max iters = {num_iters.max()}")
            all_num_iters.append(num_iters)
            all_mistakes.append(num_mistake_presentations.sum(axis=1))
        
            # import matplotlib.pyplot as pt
            # pt.subplot(1,2,1)
            # pt.hist(num_iters, bins = np.arange(num_iters.max()+1), ec='k', fc=(.5,)*3)
            # pt.ylabel("Frequency")
            # pt.xlabel("Iterations to convergence")
            # pt.subplot(1,2,2)
            # pt.hist(num_mistake_presentations.sum(axis=1), bins = np.arange(num_mistake_presentations.sum(axis=1).max()+1), ec='k', fc=(.5,)*3)
            # pt.xlabel("Number of mistakes")
            # pt.suptitle(f"Perceptron Learning of LTMs (N = {N})")
            # pt.tight_layout()
            # pt.show()

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

