import itertools as it
import numpy as np
from sign_solve import solve

N = 4
V = np.array(tuple(it.product((-1, 1), repeat=N))).T

M = 4
kidx = list(range(M))

for perm in it.permutations(kidx):
    success, _ = solve(V[:,kidx], V[:, perm])
    if not success:
        print(f"Failed on {perm}")
        break

if success:
    print("All succeeded")

