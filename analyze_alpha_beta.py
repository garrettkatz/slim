import os
import random
import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mpl
import pickle as pk

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True

Ns = [4,5,6,7]
markers = '^sod'

pt.figure(figsize=(3,2.85))

for n,(N, marker) in enumerate(zip(Ns, markers)):

    fname = f"sq_ccg_ltm_mp_{N}.pkl"
    if not os.path.exists(fname): continue

    # load optimized weights
    with open(fname, "rb") as f:
        opt_metrics = pk.load(f)
        W = opt_metrics[0]

    # load canonical regions and adjacencies
    ltms = np.load(f"ltms_{N}_c.npz")
    Yc, Wc, X = ltms["Y"], ltms["W"], ltms["X"]
    with open(f"adjs_{N}_jc.npz", "rb") as f:
        Ac = pk.load(f)

    # solve alpha and beta
    ab = {}
    resid = {}
    for (i, j, k) in Ac:
        # least-squares solution and residual
        wi = W[i]
        wjx = np.stack((W[j], X[:,k])).T        
        ab[i,j,k] = np.linalg.lstsq(wjx, wi, rcond=None)[0]
        resid[i,j,k] = np.fabs(wi - wjx @ ab[i,j,k]).max()
    alpha, beta = zip(*ab.values())

    # reflection coefficients
    betas = np.empty(len(Ac))
    refls = np.empty(len(Ac))
    for a, (i,j,k) in enumerate(Ac):
        _, betas[a] = ab[i,j,k]
        refls[a] =  - 2 * (Wc[j] @ X[:,k]) / N

    # print maximum residual over all adjacencies
    print(f"{N}: span residual <= {max(resid.values())}")

    # alpha-beta scatter plot
    pt.subplot(2,1,1)
    pt.plot(alpha, beta, marker, mfc='none', mec='k', label=f"$N={N}$", alpha=0.2)
    pt.xlabel("$\\alpha$", fontsize=12)
    pt.ylabel("$\\beta$", rotation=0, fontsize=12)

    # beta reflection plot
    pt.subplot(2,1,2)
    pt.plot(refls, betas, marker, mfc='none', mec='k', label=f"$N={N}$", alpha=0.2)
    pt.xlabel("$-2w_j^{\\top} x_k/N$", fontsize=12)
    pt.ylabel("$\\beta$", rotation=0, fontsize=12)
    pt.legend(ncol=2, columnspacing=0.1, borderpad=0.2)

# save and show
pt.tight_layout()
pt.savefig("ab_analysis.pdf")
pt.show()

