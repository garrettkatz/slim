import random
import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mpl
import pickle as pk

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True

Ns = [4,5,6,7]
markers = '^sod'
postfix = "_ana"

pt.figure(figsize=(3,2.85))

for n,(N, marker) in enumerate(zip(Ns, markers)):

    # load canonical regions and adjacencies
    ltms = np.load(f"ltms_{N}_c.npz")
    Yc, Wc, X = ltms["Y"], ltms["W"], ltms["X"]
    with open(f"adjs_{N}_c.npz", "rb") as f:
        (Yn, Wn) = pk.load(f)

    # build canonical adjacency matrix
    Ac = set()
    for i in range(len(Yn)):
        for j in range(len(Yn[i])):
            # canonicalize neighbor
            w = np.sort(np.fabs(Wn[i][j]))
            y = np.sign(w @ X)
            j = (y == Yc).all(axis=1).argmax()
            assert (y == Yc[j]).all() # hypothesis that every adjacency has joint canonicalization
            k = (Yc[j] == Yc[i]).argmin()
            Ac.add((i,j,k))
    Ac = tuple(Ac)

    # random subset to declutter image
    # Ac = random.sample(Ac, min(30, len(Ac)))

    # load optimized weights
    with open(f"sq_ccg_ltm_mp_{N}{postfix}.pkl", "rb") as f:
        (W, sq_loss_curve, loss_curve, extr_curve, gn_curve, pgn_curve, cos_curve, scale_curve) = pk.load(f)

    # solve alpha and beta
    ab = {}
    resid = {}
    for (i, j, k) in Ac:
        # if i > j: continue # avoid redundancies

        wi = W[i]
        k = (Yc[i] == Yc[j]).argmin()
        wjx = np.stack((W[j], X[:,k])).T
        
        ab[i,j,k] = np.linalg.lstsq(wjx, wi, rcond=None)[0]
        resid[i,j,k] = np.fabs(wi - wjx @ ab[i,j,k]).max()

    print(f"{N}: span residual <= {max(resid.values())}")
    alpha, beta = zip(*ab.values())

    # pt.subplot(1, len(Ns), n+1)
    pt.subplot(2,1,1)
    pt.plot(alpha, beta, marker, mfc='none', mec='k', label=f"$N={N}$", alpha=0.2)
    pt.xlabel("$\\alpha$", fontsize=12)
    pt.ylabel("$\\beta$", rotation=0, fontsize=12)
    # pt.legend()

    alphas = np.empty(len(Ac))
    betas = np.empty(len(Ac))
    i_sums = np.empty(len(Ac))
    j_sums = np.empty(len(Ac))
    wjx_dots = np.empty(len(Ac))
    sgn_ham = np.empty(len(Ac))
    for a, (i,j,k) in enumerate(Ac):
        alphas[a], betas[a] = ab[i,j,k]
        i_sums[a] = Wc[i].sum()
        j_sums[a] = Wc[j].sum()
        wjx_dots[a] =  - (Wc[j] @ X[:,k]) / N
        sgn_ham[a] = (Wc[j] * X[:, k] > 0).sum()

    # pt.subplot(1,3,2)
    # pt.plot(np.fabs(wjx_dots), alphas, marker, mfc='none', mec='k', label=f"$N = {N}$")
    # pt.xlabel("jsum", fontsize=12)
    # pt.ylabel("alpha", rotation=0)
    # pt.legend()

    pt.subplot(2,1,2)
    pt.plot(2*wjx_dots, betas, marker, mfc='none', mec='k', label=f"$N={N}$", alpha=0.2)
    pt.xlabel("$-2w_j^{\\top} x_k/N$", fontsize=12)
    pt.ylabel("$\\beta$", rotation=0, fontsize=12)
    pt.legend(ncol=2, columnspacing=0.1, borderpad=0.2)
    
    # alphas = np.empty((len(Wc), len(Wc)))
    # betas = np.empty((len(Wc), len(Wc)))
    # for (i, j, k) in Ac:
    #     alphas[i,j], betas[i,j] = ab[i,j,k]

    # pt.subplot(1,3,2)
    # pt.imshow(alphas)
    # pt.xlabel("$i$")
    # pt.ylabel("$j$")
    # pt.title("$\\alpha$")

    # pt.subplot(1,3,3)
    # pt.imshow(betas)
    # pt.xlabel("$i$")
    # pt.ylabel("$j$")
    # pt.title("$\\beta$")

pt.tight_layout()
pt.savefig("ab_analysis.pdf")
pt.show()

