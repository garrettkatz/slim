import matplotlib.pyplot as pt
from matplotlib.scale import LogScale
import numpy as np
import pickle as pk

Ns = np.arange(2,6)
num_regions = []
num_classes = []

for N in Ns:
    print(f"N={N}...")

    with open(f"hemis_{N}.npy","rb") as f: weights, hemis = pk.load(f)
    perms = np.load(f"perms_{N}.npy")

    reprs = set()
    seive = set(map(tuple, hemis))
    while len(seive) > 0:
        print(f" {len(seive)} left in seive")
        hemi = seive.pop()
        reprs.add(hemi)
        seive -= set(map(tuple, np.array(hemi)[perms]))

    num_regions.append(hemis.shape[0])
    num_classes.append(len(reprs))

print("Done.  Plotting...")

# pt.subplot(1,2,1)
pt.figure(figsize=(4,3))
pt.plot(Ns, num_regions, 'ko-', label="Regions")
pt.plot(Ns, num_classes, 'ks-', label="Classes")
pt.plot(Ns, 2**((Ns-1)**2), 'k--', label="$2^{(N-1)^2}$")
pt.plot(Ns, 2**((Ns-1)**2/2), 'k:', label="$2^{\\frac{1}{2}(N-1)^2}$")
pt.yscale(LogScale(axis=pt.gca(), base=2))
pt.xlabel("N")
# pt.ylabel("Number of regions")
pt.xticks(Ns, Ns)
pt.legend()
pt.tight_layout()
pt.savefig("region_counts.pdf")

# pt.subplot(1,2,2)
# pt.plot(Ns, num_classes, 'ko-', label="Classes")
# # pt.yscale(LogScale(axis=pt.gca(), base=2))
# pt.xlabel("N")
# pt.ylabel("Number of equivalence classes")
# pt.xticks(Ns, Ns)
# pt.legend()

pt.show()

