import numpy as np

def adjacency(Y, sym=True):
    A = {i: [] for i in range(Y.shape[0])}

    for i in range(Y.shape[0]-1):

        a_i = (i+1) + np.flatnonzero((Y[i] != Y[i+1:]).sum(axis=1) == 1)
        A[i].extend(a_i)

        if sym: # other half of symmetric A
            for j in a_i: A[j].append(i)

    return A

if __name__ == "__main__":

    N = 5
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    A = adjacency(Y)

    import matplotlib.pyplot as pt
    pt.bar(range(len(A)), [len(A[i]) for i in range(len(A))])
    pt.show()

