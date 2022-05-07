import matplotlib.pyplot as pt
import numpy as np
import itertools as it
from sign_solve import solve
import scipy as sp

np.set_printoptions(linewidth=1000)

k = 8
R = 2**k # num reps = small cube size
M = R**2 # = 2**(2*k) = large cube size

B = 2*np.eye(R, dtype=int) - np.ones((R, R), dtype=int)
H = np.concatenate((
    np.tile(B, (1,R)),
    np.repeat(B, R, axis=1),
    np.ones((1, M)), # bias
), axis=0)

print((B > 0).astype(int))
print((H > 0).astype(int))
print(H.shape)
input('.')

H = sp.sparse.csr_array((H > 0).astype(int))

num_samples = 100
success = np.empty(num_samples, dtype=bool)
for s in range(num_samples):
    Y = np.random.choice((-1,+1), size=(1,M))
# for s,out in enumerate(it.product((-1,+1), repeat=M)):
#     if s == num_samples: break
#     Y = np.array(out).reshape(1,M)
    print(Y)
    success[s], W = solve(H, Y)
    print(f"{s} of {num_samples}: {success[s]}")

print(f"{success.mean()} successful")

