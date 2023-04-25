import itertools as it
import numpy as np

if __name__ == "__main__":

    N = 6
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    eta = 1

    i = np.random.randint(Y.shape[0])
    y = Y[i]

    num_presentations = np.zeros(X.shape[1])

    w = np.zeros(N)

    for k in it.count():

        if (np.sign(w @ X) == y).all(): break

        print(f"{k}: {(np.sign(w @ X) == y).sum()} of {X.shape[1]} correct")

        j = np.random.randint(X.shape[1])
        num_presentations[j] += 1

        if np.sign(w @ X[:,j]) == y[j]: continue

        w = w + eta * y[j] * X[:,j]


    print(f"{k} iterations total.  num presentations (sums to {num_presentations.sum()}):")
    print(num_presentations)

