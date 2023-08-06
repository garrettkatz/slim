import itertools as it
import numpy as np

N = 5
mx = int(2**(N-1) / N)
u = np.random.randint(-mx, mx+1, size=(N,))

sums = {}
for x in it.product((-1,1), repeat=N):
    s = u @ np.array(x)
    sums[s] = sums.get(s, 0) + 1

print(mx)
print(u)
print(len(sums))
print(max(sums.keys()))
print(max(sums.values()))

