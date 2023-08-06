import itertools as it
import numpy as np

N = 3
V = np.array(tuple(it.product((-1, 1), repeat=N))).T

W = np.random.randn(N,N)
net = 2**np.arange(N-1,-1,-1).reshape(1,N) @ (W @ V > 0).astype(int)
net = net.flatten()

print((V > 0).astype(int))
print((W @ V > 0).astype(int))
print(net)

for P in it.permutations(range(2**N), 2**N):
    P = np.array(P)
    if not (net[P] == P[net]).all():
        print("P, netP, Pnet")
        print(P)
        print(net[P])
        print(P[net])
    assert (net[P] == P[net]).all()
