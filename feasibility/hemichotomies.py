import os, sys
from operator import neg
import itertools as it
import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
from nondet_sized import NonDeterminator
import warnings

warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
np.set_printoptions(formatter={"int": lambda x: "%+d" % x}, linewidth=1000)

# N = 4
N = int(sys.argv[1])
fname = f"hemichotomies_{N}.npz"

if os.path.exists(fname):

    npz = np.load(fname)
    weights, hemis = npz["weights"], npz["hemis"]

else:

    X = np.array(tuple(it.product((-1, 1), repeat=N))).T
    
    perms = np.load(f"perms_{N}.npy")
    
    weights = []
    hemis = []
    leads = tuple({} for k in range(2**(N-1)))
    carry = [2**(N-1)] # singleton list makes it global
    
    nd = NonDeterminator()
    def check_hemi():
        new_hemi = False
        y = ()
        for k in range(2**(N-1)):
    
            yk = nd.choice((-1, +1))
            if yk == +1 and k < carry[0]:
                print(f"First inc {k}")
                carry[0] = k

            y += (yk,)
            if y in leads[k]:
                if leads[k][y]: continue
                else: return False
    
            # y not in leads[k]
            new_hemi = True
            Y = np.array(y)
            result = linprog(
                c = X[:,:k+1] @ Y,
                A_ub = -(X[:,:k+1] * Y).T,
                b_ub = -np.ones(k+1),
                bounds = (None, None),
            )
            w = result.x
            if k+1 == 2**(N-1): w = w.round().astype(int) # empirically observed to be integer-valued
            feasible = (np.sign(w @ X[:,:k+1]) == Y).all()
    
            leads[k][y] = feasible
            if not feasible: return False
    
        if not new_hemi: return True
    
        # save new hemi and its permutations
        Y = np.concatenate((Y, -Y[::-1]))
        for perm in perms:
            yp = tuple(Y[perm])
            for k in range(2**(N-1)):
                leads[k][yp[:k+1]] = True
    
        weights.append(w)
        hemis.append(Y)
    
        print(f"{len(hemis)} hemis found, {sum(leads[k].values())} with perms")
    
        return True
    
    for _ in nd.runs(check_hemi): pass
    
    weights = np.stack(weights)
    hemis = np.stack(hemis)
    np.savez(fname, weights=weights, hemis=hemis)

# print(N, sum(leads[-1].values()))
print(hemis)
print(weights)
print(N, hemis.shape)
