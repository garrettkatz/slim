import itertools as it
import numpy as np

N = 2
V = np.array(tuple(it.product((-1, 1), repeat=N))).T

# this is wrong because W rows/cols are not one-hot

P = set()
for entries in it.product((-1, 0, 1), repeat=N**2):
    W = np.array(entries).reshape((N, N))
    WV = W @ V
    p = tuple(
        (WV[:,j:j+1] == V).all(axis=0).argmax()
        for j in range(2**N))
    # p = np.empty(2**N, dtype=int)
    # for j in range(2**N):
    #     p[j] = (WV[:,j:j+1] == V).all(axis=0).argmax()
    P.add(p)

print(len(P))

