import sys
import pickle as pk
import itertools as it
import numpy as np
from time import perf_counter
from scipy.special import factorial
from sklearn.svm import LinearSVC
from enumerate_ltms import get_equivalence_class_size

if __name__ == "__main__":

    do_exp = False # whether to run the svm optimization or just load the results

    # experiment up to dimension N_max
    if len(sys.argv) > 1:
        N_max = int(sys.argv[1])
    else:
        N_max = 8
    Ns = list(range(3, N_max + 1))

    if do_exp:

        # track number of iterations and mistakes
        all_num_iters = []
        all_run_times = []

        # process one N at a time    
        for N in Ns:
            print(f"N = {N}")
    
            # load all canonical hemichotomies
            ltms = np.load(f"ltms_{N}_c.npz")
            Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

            # track iterations and runtime on each hemichotomy        
            num_iters = np.empty(Y.shape[0], dtype=int)
            run_time = np.zeros(Y.shape[0], dtype=float)

            # separate svms on one hemichotomy at a time
            for i, y in enumerate(Y):

                svc = LinearSVC(fit_intercept=False, max_iter=50000, tol=1e-8, verbose=0, C=100.)

                start = perf_counter()
                svc.fit(X.T, y)
                duration = perf_counter() - start

                acc = svc.score(X.T, y)
                assert acc == 1.0
                # print(acc)
                assert (np.sign(svc.coef_ @ X) == y).all()

                # track total number of iterations until convergence
                num_iters[i] = svc.n_iter_
                run_time[i] = duration
                print(f"hemi {i} of {len(Y)}: {svc.n_iter_} iterations, {duration} seconds")

            # print maximum iterations over all hemichotomies, store results before next N
            print(f"max iters = {num_iters.max()}")
            all_num_iters.append(num_iters)
            all_run_times.append(run_time)
        
        # save all results
        with open("svmbase.pkl", "wb") as f:
            pk.dump((all_num_iters, all_run_times), f)

    # load all results
    with open("svmbase.pkl", "rb") as f:
        (all_num_iters, all_run_times) = pk.load(f)

    # convert to processing per sample, 2**(N-1) samples
    all_num_iters = [num_iters / 2**(N-1) for (num_iters, N) in zip(all_num_iters, Ns)]
    all_run_times = [run_times / 2**(N-1) for (run_times, N) in zip(all_run_times, Ns)]

    # get average/stdev metrics for each N, weighted by equivalence class sizes
    avg_iters, std_iters = np.empty(len(Ns)), np.empty(len(Ns))
    avg_times, std_times = np.empty(len(Ns)), np.empty(len(Ns))
    for n, N in enumerate(Ns):

        # load all canonical hemichotomies
        ltms = np.load(f"ltms_{N}_c.npz")
        W = ltms["W"]

        # calculate weights for each equivalence class
        ecw = np.array(list(map(get_equivalence_class_size, W)))
        ecw /= ecw.sum()

        # calculate stats
        num_iters = all_num_iters[n]
        avg_iters[n] = np.average(num_iters, weights=ecw)
        std_iters[n] = np.sqrt(np.cov(num_iters, aweights=ecw))

        run_times = all_run_times[n]
        avg_times[n] = np.average(run_times, weights=ecw)
        std_times[n] = np.sqrt(np.cov(run_times, aweights=ecw))

    # plot results
    import matplotlib.pyplot as pt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['text.usetex'] = True

    pt.figure(figsize=(5,1.8))
    for sp, (lab, dats, avg, std) in enumerate([("iterations", all_num_iters, avg_iters, std_iters), ("run time", all_run_times, avg_times, std_times)]):
        pt.subplot(1, 2, sp+1)

        # stdev envelope background
        pt.fill_between(Ns, avg-std, avg+std, color=(.8,)*3)
    
        # individual hemichotomy points
        for (dat, N) in zip(dats, Ns):
            pt.plot(N +  + np.random.randn(len(dat))*0.05, dat, '.', color=(.4,)*3)
    
        # averages
        pt.plot(Ns, avg, 'ko-')
    
        # format axes
        pt.ylabel(f"SVM {lab} per sample")
        pt.xlabel("Input dimension $N$")
        pt.xticks(Ns, Ns)
        # pt.yscale('log')
        # pt.yticks(10.**np.arange(3), ["$10^{%d}$" % k for k in range(3)])
        # pt.gca().yaxis.get_minor_locator().set_params(numticks=10)

    # save and show
    pt.tight_layout()
    pt.savefig('svmbase.pdf')
    pt.show()

