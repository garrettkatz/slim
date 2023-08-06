import numpy as np
import itertools as it
import pickle as pk
import matplotlib.pyplot as pt

for N in reversed(range(2, 7)):

    fname = f"hemis_{N}.npy"
    with open(fname,"rb") as f: _, hemis = pk.load(f)
    
    print(f"N={N}:")
    unicounts = []
    for M in range(1,2**(N-1)+1):
        uni = np.unique(hemis[:,:M], axis=0)
        unicounts.append(uni.shape[0])
        print(f" N,M={N},{M}: {uni.shape[0]} unique hemis[:,:M]")
    print(f" D*_N = {hemis.shape[0]} unique hemis")
    print()

    pt.plot(range(1,2**(N-1)+1), unicounts, label=str(N))

pt.xlabel("M")
pt.ylabel("unicount")
pt.yscale("log")
pt.legend()
pt.show()
    
    
