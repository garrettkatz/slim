import os, sys
from collections import deque
import itertools as it
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

N = int(sys.argv[1])
fname = f"hemitree_{N}.npz"

if False: #os.path.exists(fname):

    npz = np.load(fname)
    weights, hemis = npz["weights"], npz["hemis"]

else:

    X = np.array(tuple(it.product((-1, +1), repeat=N))).T
    Xh = X[:,:2**(N-1)] # more numerically stable linprog without antiparallel data
    
    weights = []
    hemis = []
    
    def representative(w): return tuple(np.sort(np.fabs(w)).astype(int))
    
    w0 = np.eye(1, N, dtype=int).flatten()
    frontier = deque([w0])
    queued = set([representative(w0)])
    while len(frontier) > 0:
        print(f"{len(weights)}: |frontier|={len(frontier)}")
    
        w = frontier.popleft()
        h = np.sign(w @ X)

        weights.append(w)
        hemis.append(h)

        for k in range(2**(N-1)):
        # for k in np.flatnonzero(np.fabs(w @ Xh) == 1): # only assuming linprog preserves boundary condition
            hf = h[:2**(N-1)].copy()
            hf[k] *= -1

            result = linprog(
                c = Xh @ hf,
                A_ub = -(Xh * hf).T,
                b_ub = -np.ones(2**(N-1)),
                bounds = (None, None),

                # method = "revised simplex",
                # x0 = N*(N-2)*w - (N-1)*h[k]*X[:,k], # guaranteed feasible? only if linprog preserves boundary condition
            )
            wf = result.x
            feasible = (np.sign(wf @ Xh) == hf).all()
            if not feasible: continue
    
            wf = wf.round().astype(int) # empirically observed to be integer-valued
            wf_rep = representative(wf)
            if wf_rep in queued: continue
    
            queued.add(wf_rep)
            frontier.append(wf)
    
    weights = np.stack(weights)
    hemis = np.stack(hemis)
    
    np.savez(fname, weights=weights, hemis=hemis)

print(hemis)
print(weights)

print(f"{len(hemis)} regions (<~ {len(hemis) * 2**N * np.arange(1,N+1).prod()} with perms)")

# print("Expanding with perms...")
# weights_with_perms = set()
# for w in weights:
#     for p in it.permutations(range(N)):
#         for s in it.product((+1,-1), repeat=N): # +1 first so perms[0] is identity
#             wps = tuple(w[list(p)] * np.array(s))
#             weights_with_perms.add(wps)
# print(f"{len(hemis)} regions ({len(weights_with_perms)} with perms)")

