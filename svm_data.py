import itertools as it
import numpy as np
from sklearn.svm import LinearSVC

def random_transitions(X, Y, N, B, rng=None):

    if rng == None: rng = np.random.default_rng()

    w_new, w_old, x = tuple(np.empty((B, N)) for _ in range(3))
    y = np.empty(B)
    for b in range(B):

        # randomly sample training data
        i = rng.choice(Y[N].shape[0]) # dichotomy
        t = rng.choice(Y[N].shape[1])+1 # time-step
        K = rng.choice(Y[N].shape[1], size=t, replace=False) # example indices

        # most recent example
        k_new, k_old = K[0], K[1:]
        x[b] = X[N][:,k_new]
        y[b] = Y[N][i,k_new]

        # get max-margin classifiers via svm
        svc = LinearSVC(dual='auto', fit_intercept=False)
        svc.fit(
            np.concatenate((X[N][:,K], -X[N][:,K]), axis=1).T,
            np.concatenate((Y[N][i,K], -Y[N][i,K]), axis=0))
        w_new[b] = svc.coef_.flatten()

        # old weights exclude most recent example
        if len(k_old) == 0:
            w_old[b] = np.zeros(N)
        else:
            svc = LinearSVC(dual='auto', fit_intercept=False)
            svc.fit(
                np.concatenate((X[N][:,k_old], -X[N][:,k_old]), axis=1).T,
                np.concatenate((Y[N][i,k_old], -Y[N][i,k_old]), axis=0))
            w_old[b] = svc.coef_.flatten()

    return w_new, w_old, x, y
    

def all_transitions(X, Y, N, T=None):

    if T == None: T = 2**(N-1)

    w, m = {}, {}
    w_new, w_old, x, y, margins = [], [], [], [], []
    for i in range(Y[N].shape[0]):
        w[i] = {(): np.zeros((1,N))}
        m[i] = {}
        for t in range(1, T+1):
            for K in it.combinations(range(Y[N].shape[1]), t):

                # get max-margin classifier via svm
                svc = LinearSVC(dual='auto', fit_intercept=False)
                svc.fit(
                    np.concatenate((X[N][:,K], -X[N][:,K]), axis=1).T,
                    np.concatenate((Y[N][i,K], -Y[N][i,K]), axis=0))

                w[i][K] = svc.coef_
                w[i][K] /= np.linalg.norm(w[i][K])
                m[i][K] = np.fabs(w[i][K] @ X[N][:,K]).min() / N**.5 # cos theta(w,x)

                # need special case handling here for t == 1, m[i][K] = 1.

                # make sure it fits data
                assert (np.sign(w[i][K] @ X[N][:,K]) == Y[N][i,K]).all()

        # for t in range(T):
        for t in range(1, T): # first step is prescribed (w = yx)
            for K in it.combinations(range(Y[N].shape[1]), t):
                for k in range(Y[N].shape[1]):
                    if k in K: continue
                    Kk = tuple(sorted(K + (k,)))

                    w_old.append(w[i][K])
                    x.append(X[N][:,k])
                    y.append(Y[N][i,k])
                    w_new.append(w[i][Kk])
                    margins.append(m[i][Kk])

    return w_new, w_old, x, y, margins

if __name__ == "__main__":

    X, Y = {}, {}
    for N in range(3, 6):
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X[N], Y[N] = ltms["X"], ltms["Y"]

    for N in range(3,5):
        for T in range(1, 2**(N-1)+1):
            w_new, w_old, x, y = all_transitions(X, Y, N, T)
            print(f"{N},{T} of {2**(N-1)}: {len(w_new)} transitions total")

    # N = 4
    # T = 2**(N-1)
    # B = 10

    # w_new, w_old, x, y = all_transitions(X, Y, N, T)
    # print(f"{len(w_new)} transitions total")

    # w_new, w_old, x, y = random_transitions(X, Y, N, B)
    # print(len(w_new))

    # print(w_new)
    # print(x)
    # print(y)

