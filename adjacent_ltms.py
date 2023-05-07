import numpy as np

def adjacency(Y, sym=True):
    A = {i: [] for i in range(Y.shape[0])} # rows (i,j) of Y that differ in one column
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

    N = 5
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

    for k in range(2**(N-1)):
        print(f"{k}: {(MK==k).sum()}")

    pt.subplot(1,2,1)
    pt.imshow(MA)
    pt.subplot(1,2,2)
    pt.imshow(MK)
    for i in A:
        for j,k in zip(A[i], K[i]):
            pt.text(i, j, k)
    pt.show()
    
    # check if every adjacency also has canonical adjacency

    print(W.shape)
    canon = np.empty(len(W), dtype=int)
    for i in range(len(W)):
        wc = np.sort(np.fabs(W[i]))
        yc = np.sign(wc @ X)
        canon[i] = (yc == Y).all(axis=1).argmax()
    print(canon)

    all_in_canon = True
    for i in A:
        for j, k in zip(A[i], K[i]):
            if canon[j] not in A[canon[i]]:
                all_in_canon = False
                break
        if not all_in_canon: break

    print(f"All in canon: {all_in_canon}")

    # for i in canon:
    #     any_in_canon = False
    #     for j, k in zip(A[i], K[i]):
    #         if j not in canon: continue
    #         any_in_canon = True

    #         ab = np.linalg.lstsq(np.vstack((W[i], X[:,k])).T, W[j], rcond=None)[0]
    #         print(X[:,k], W[i], W[j], ab, i,j,k)

    #     assert any_in_canon
