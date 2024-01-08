import numpy as np
import pickle as pk

# If Ns is a list, returns dicts Yc[N], W[N], X[N], Ac[N] by N
# If Ns is an integer N, returns Yc, W, X, Ac for that N
def load_ltm_data(Ns):

    singleton = (type(Ns) == int)
    if singleton: Ns = [Ns]

    Yc, W, X, Ac = {}, {}, {}, {}
    for N in Ns:

        # load canonical hemis
        ltms = np.load(f"ltms_{N}_c.npz")
        Yc[N], W[N], X[N] = ltms["Y"], ltms["W"], ltms["X"]

        with open(f"adjs_{N}_jc.npz", "rb") as f: Ac[N] = pk.load(f)

    if singleton:
        return Yc[Ns[0]], W[Ns[0]], X[Ns[0]], Ac[Ns[0]] 
    else:
        return Yc, W, X, Ac

# organize adjacencies by source region
# Ac: set { ... (i,j,k) ... } of canonical adjacencies, as saved in adjs_{N}_jc.npz
# returns dict A[i] = [..., (j,k), ...] adjacent region j with boundary k to region i
def organize_by_source(Ac):
    A = {}
    for (i,j,k) in Ac:
        if i not in A: A[i] = []
        A[i].append((j, k))
    return A


