import numpy as np

def adjacency(Y, sym=True):
    A = {i: [] for i in range(Y.shape[0])} # rows (i,j) of Y that differ  in one column
    K = {i: [] for i in range(Y.shape[0])} # columns k where they differ

    for i in range(Y.shape[0]-1):

        a_i = (i+1) + np.flatnonzero((Y[i] != Y[i+1:]).sum(axis=1) == 1)
        k_i = (Y[i] != Y[a_i]).argmax(axis=1)
        A[i].extend(a_i)
        K[i].extend(k_i)

        if sym: # other half of symmetric A
            for j,k in zip(a_i, k_i):
                A[j].append(i)
                K[j].append(k)

    return A, K

if __name__ == "__main__":

    N = 3
    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    A, K = adjacency(Y)

    import matplotlib.pyplot as pt
    pt.bar(range(len(A)), [len(A[i]) for i in range(len(A))])
    pt.show()

    MA = np.zeros((len(A), len(A)), dtype=int)
    MK = -np.ones(MA.shape, dtype=int)
    for i in A:
        MA[i][A[i]] = 1
        MK[i][A[i]] = K[i]

    print(MA)
    print(MK)

    pt.subplot(1,2,1)
    pt.imshow(MA)
    pt.subplot(1,2,2)
    pt.imshow(MK)
    pt.show()
    
