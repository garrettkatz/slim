import os, sys
import numpy as np
import itertools as it

# np.set_printoptions(sign="+")
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

# N = 7
N = int(sys.argv[1])
fname = f"perms_{N}.npy"

X = np.array(tuple(it.product((-1, 1), repeat=N))).T
print(X.shape) # (num neurons N, num verticies 2**N)

if os.path.exists(fname):

    perms = np.load(fname)

else:

    # calculate all column perms of X for signed row permutations of neurons
    pows = 2**np.arange(N-1,-1,-1)
    perms = []
    for p in map(list, it.permutations(range(N))):
        for s in it.product((+1,-1), repeat=N): # +1 first so perms[0] is identity
            perms.append( pows @ ((np.array(s).reshape(N,1) * X[p,:]) > 0) )
    
    perms = np.stack(perms)
    np.save(fname, perms)

print(perms)
print(f"{len(perms)} = {2**N * np.arange(1,N+1).prod()} perms")



