import pickle as pk
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from sklearn.svm import LinearSVC

if __name__ == "__main__":

    do_gen = True

    N, M = 3, 4
    N, M = 4, 8
    # ltms = np.load(f"ltms_{N}.npz")
    ltms = np.load(f"ltms_{N}_c.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    if do_gen:

        residuals = {m: [] for m in range(1, M+1)}
        for r, y in enumerate(Y):
            print(f"{r} of {len(Y)}")
            w = {}
            for m in range(1,M+1):
                for K in map(np.array, it.combinations(range(2**(N-1)), m)):
    
                    inp = np.concatenate((X[:,K], -X[:,K]), axis=1).T
                    out = np.concatenate((y[K], -y[K]))
    
                    svc = LinearSVC(fit_intercept=False, max_iter=1000)
                    svc.fit(inp, out)
                    acc = svc.score(inp, out)
                    assert acc == 1.0
    
                    key = tuple(K)
                    w[key] = svc.coef_.flatten()
    
                    if m == 1: continue
    
                    w_new = w[key]
                    for k in range(m):
                        w_old = w[key[:k] + key[k+1:]]
                        x_old = X[:,key[k]]
                        y_old = y[key[k]]
    
                        wx = np.vstack((w_old, x_old))
                        a, b = np.linalg.lstsq(wx.T, w_new, rcond=None)[0]
                        resid = np.fabs(w_new - (a * w_old + b * x_old)).max()
    
                    residuals[m].append(resid)
                    # print(resid)

        with open("gentest.pkl","wb") as f:
            pk.dump(residuals, f)

    with open("gentest.pkl","rb") as f:
        residuals = pk.load(f)

    for m, resids in residuals.items():
        # pt.plot([m]*len(resids), resids, 'ko')
        pt.plot(np.random.rand(len(resids))*0.5 + m, resids, 'ko')
    pt.show()
