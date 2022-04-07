import itertools as it
import numpy as np

N = 3
V = np.array(tuple(it.product((-1, 1), repeat=N))).T

P = []

for perm in it.permutations(range(N)):
    for signs in it.product((-1, 1), repeat=N):
        WV = (np.array(signs).reshape(-1,1) * V)[perm, :]
        p = tuple(
            (WV[:,j:j+1] == V).all(axis=0).argmax()
            for j in range(2**N))
        P.append(p)

print(2**N * np.arange(1,N+1).prod())
print(len(P))

print(np.array(P))
