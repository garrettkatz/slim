import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import matplotlib as mp

np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'
mp.rcParams['text.usetex'] = True

if __name__ == "__main__":

    # initialize range of Ns and their residuals for plotting
    Ns = [3, 4, 5, 6, 7]
    residuals = {N: [] for N in Ns}

    # process each N
    for N in Ns:

        # load canonical regions and their adjacencies
        ltms = np.load(f"ltms_{N}_c.npz")
        Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
        with open(f"adjs_{N}_c.npz", "rb") as f:
            (Yn, Wn) = pk.load(f)

        # compute coefficients and measure span residuals
        for i, (yc, wi) in enumerate(zip(Y, W)):
            for yn, wj in zip(Yn[i], Wn[i]):

                # get boundary vertex for current adjacency
                k = (yc == yn).argmin()
                x = X[:,k]

                # least-squares fit and residual for alpha and beta                
                wjx = np.stack((wj, x)).T
                ab = np.linalg.lstsq(wjx, wi, rcond=None)[0]
                resid = np.fabs(wi - wjx @ ab).max()
                residuals[N].append(resid)

    # plot results
    pt.figure(figsize=(2.5,1.5))

    # violin plots of distributions
    parts = pt.violinplot([residuals[N] for N in Ns], Ns, widths=.8)
    for pc in parts['bodies']:
        pc.set_edgecolor('black')
        pc.set_facecolor((.8,.8,.8))
    for key in ['cmins', 'cmaxes', 'cbars']:
        parts[key].set_edgecolor('none')

    # individual datapoints for each adjacency
    for N in Ns:
        M = len(residuals[N])
        pt.scatter(
            np.random.randn(M)*0.03 + N, # small horizontal noise
            residuals[N],
            .5, color=(.5,)*3)

    # format axes
    pt.xticks(Ns, Ns)
    pt.xlabel('Input dimension $N$')
    pt.ylabel('Span residual')
    pt.tight_layout()

    # save and show
    pt.savefig('spanviols.pdf')
    pt.show()

