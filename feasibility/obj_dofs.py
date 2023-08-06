"""
Quartic span violation objective: get a feel for solution set
obj = 0 for each transition: how many eqs, how many vars?
"""

import numpy as np
from adjacent_ltms import adjacency

if __name__ == "__main__":

    for N in [3, 4, 5]:

        ltms = np.load(f"ltms_{N}.npz")
        Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
        A, K = adjacency(Y, sym=True)

        numcon = sum(map(len, A.values()))
        print(f"N={N}: {W.size} vars, {numcon} constraints ({numcon/2} wo sym)")

