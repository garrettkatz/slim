import numpy as np
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt
import matplotlib as mp

np.set_printoptions(threshold=10e6)
mp.rcParams['font.family'] = 'serif'

if __name__ == "__main__":

    Ns = [3, 4, 5]
    residuals = {N: [] for N in Ns}

    for N in Ns:

        ltms = np.load(f"ltms_{N}.npz")
        Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    
        # get adjacencies
        A, K = adjacency(Y, sym=True)

        # measure span viols
        for i in range(len(Y)):
            for j, k in zip(A[i], K[i]):

                w = np.stack((W[i], W[j])).T
                x = X[:,k]
                
                ab = np.linalg.lstsq(w, x, rcond=None)[0]
                r = np.linalg.norm(w @ ab - x)
                residuals[N].append(r)

    pt.figure(figsize=(3,2))
    parts = pt.violinplot([residuals[N] for N in Ns], Ns, widths=.8)
    for pc in parts['bodies']:
        pc.set_edgecolor('black')
        pc.set_facecolor((.8,.8,.8))
    for key in ['cmins', 'cmaxes', 'cbars']:
        parts[key].set_edgecolor((.8,.8,.8))
    for N in Ns:
        M = len(residuals[N])
        pt.plot(
            np.random.uniform(-.01, .01, M) + N,
            np.random.uniform(-.01, .01, M) + np.array(residuals[N]),
            'k.')
    pt.xlabel('Number of synapses N')
    pt.ylabel('Span residual')
    pt.tight_layout()
    pt.show()

