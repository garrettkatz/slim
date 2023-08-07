import itertools as it
import numpy as np
from sklearn.svm import LinearSVC

def random_transitions(X, Y, N, B):

    w_new, w_old, x = tuple(np.empty((B, N)) for _ in range(3))
    y = np.empty(B)
    for b in range(B):

        # randomly sample training data
        i = np.random.choice(Y[N].shape[0]) # dichotomy
        t = np.random.choice(Y[N].shape[1])+1 # time-step
        K = np.random.choice(Y[N].shape[1], size=t, replace=False) # example indices

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

    w = {}
    w_new, w_old, x, y = [], [], [], []
    for i in range(Y[N].shape[0]):
        w[i] = {(): np.zeros(N)}
        for t in range(1, T+1):
            for K in it.combinations(range(Y[N].shape[1]), t):

                # get max-margin classifier via svm
                svc = LinearSVC(dual='auto', fit_intercept=False)
                svc.fit(
                    np.concatenate((X[N][:,K], -X[N][:,K]), axis=1).T,
                    np.concatenate((Y[N][i,K], -Y[N][i,K]), axis=0))

                w[i][K] = svc.coef_

        for t in range(T):
            for K in it.combinations(range(Y[N].shape[1]), t):
                for k in range(Y[N].shape[1]):
                    if k in K: continue

                    w_old.append(w[i][K])
                    x.append(X[N][:,k])
                    y.append(Y[N][i,k])
                    w_new.append(w[i][tuple(sorted(K + (k,)))])

    return w_new, w_old, x, y

if __name__ == "__main__":

    X, Y = {}, {}
    for N in range(3, 6):
        fname = f"ltms_{N}_c.npz"
        ltms = np.load(fname)
        X[N], Y[N] = ltms["X"], ltms["Y"]

    # N = 5
    # T = 4 #2**(N-1)

    N = 4
    T = 2**(N-1)
    B = 10

    w_new, w_old, x, y = all_transitions(X, Y, N, T)
    print(len(w_new))

    w_new, w_old, x, y = random_transitions(X, Y, N, B)
    print(len(w_new))

    print(w_new)
    print(x)
    print(y)

