import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mp

np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'
mp.rcParams['text.usetex'] = True

if __name__ == "__main__":

    Ns = [3, 4, 5, 6, 7]
    residuals = {N: [] for N in Ns}

    for N in Ns:

        ltms = np.load(f"ltms_{N}_c.npz")
        Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    
        # get adjacencies to canonicals
        with open(f"adjs_{N}_c.npz", "rb") as f:
            (Yn, Wn) = pk.load(f)

        # measure span viols
        for i, (yc, wi) in enumerate(zip(Y, W)):
            for yn, wj in zip(Yn[i], Wn[i]):

                k = (yc == yn).argmin()
                x = X[:,k]
                wjx = np.stack((wj, x)).T
                
                ab = np.linalg.lstsq(wjx, wi, rcond=None)[0]
                resid = np.fabs(wi - wjx @ ab).max()
                residuals[N].append(resid)

    pt.figure(figsize=(2.5,2))
    # vioplots
    parts = pt.violinplot([residuals[N] for N in Ns], Ns, widths=.8)
    for pc in parts['bodies']:
        pc.set_edgecolor('black')
        pc.set_facecolor((.8,.8,.8))
    for key in ['cmins', 'cmaxes', 'cbars']:
        # parts[key].set_edgecolor((.8,.8,.8))
        parts[key].set_edgecolor('none')

    # raw datapoints
    for N in Ns:
        M = len(residuals[N])
        # pt.plot(
        #     # np.random.uniform(-.1, .1, M) + N,
        #     np.full((M,), N),
        #     # np.random.uniform(-.01, .01, M) + np.array(residuals[N]),
        #     residuals[N],
        #     '.', color=(.5,)*3)
        pt.scatter(
            np.random.randn(M)*0.03 + N,
            # np.full((M,), N),
            # np.random.uniform(-.01, .01, M) + np.array(residuals[N]),
            # np.random.randn(M)*0.03 + np.array(residuals[N]),
            residuals[N],
            .5, color=(.5,)*3)

    pt.xticks(Ns, Ns)
    pt.xlabel('Input dimension $N$')
    pt.ylabel('Span residual')
    pt.tight_layout()
    pt.savefig('spanviols.pdf')
    pt.show()

