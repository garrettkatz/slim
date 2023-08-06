import pickle as pk
import numpy as np
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt
import matplotlib as mp
from numpy.linalg import norm

from scipy.optimize import linprog, OptimizeWarning
from scipy.linalg import LinAlgWarning
import warnings
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

np.set_printoptions(threshold=10e6)

# from cvxopt.solvers import qp, options
# from cvxopt import matrix
# options['show_progress'] = False

mp.rcParams['font.family'] = 'serif'

if __name__ == "__main__":
    
    # N = 4 # dim
    # eps = 0.01 # constraint slack threshold
    # lr = .1 # learning rate
    # num_updates = 400

    # N = 5 # dim
    # eps = 0.01 # constraint slack threshold
    # lr = 0.1 # learning rate
    # num_updates = 2000

    N = 6 # dim
    eps = 0.01 # constraint slack threshold
    lr = 0.01 # learning rate
    num_updates = 2000

    # load canonical regions and adjacencies
    ltms = np.load(f"ltms_{N}_c.npz")
    Yc, Wc, X = ltms["Y"], ltms["W"], ltms["X"]
    with open(f"adjs_{N}_c.npz", "rb") as f:
        (Yn, Wn) = pk.load(f)

    # save original
    W_lp = Wc.copy()

    # canonical adjacency matrix
    A = set()
    for i in range(len(Yn)):
        for j in range(len(Yn[i])):
            # canonicalize neighbor
            w = np.sort(np.fabs(Wn[i][j]))
            y = np.sign(w @ X)
            j = (y == Yc).all(axis=1).argmax()
            assert (y == Yc[j]).all() # hypothesis that every adjacency has joint canonicalization
            k = (Yc[j] == Yc[i]).argmin()
            A.add((i,j,k))
    A = tuple(A)

    if True: # do training
    # if False: # just load results

        # gradient update loop
        loss_curve = []
        extr_curve = []
        for update in range(num_updates):

            # update step scale for next iter
            # step_scale = lr # N=4
            # step_scale = lr / (np.log(update+1) + 1) # N=5
            # step_scale = lr / (update + 1)**.5
            # step_scale = lr * 2 / (update + 2) # frank-wolfe default, maybe only for convex opt?
            step_scale = lr * 0.99**update

            # stochastic: sample one adjacency between canonicals
            (i, j, k) = A[np.random.randint(len(A))]

            # get projected weights
            wi, wj, xk = Wc[i], Wc[j], X[:,k]
            Pk = np.eye(N) - xk.reshape((N, 1)) * xk / N
            wiPk = wi @ Pk
            wjPk = wj @ Pk
            wiPk_n, wjPk_n = norm(wiPk), norm(wjPk)

            # sampled span loss
            loss = wiPk_n*wjPk_n - wiPk @ wjPk

            # sampled gradient
            grad = {
                i: wiPk * wjPk_n / wiPk_n - wjPk,
                j: wjPk * wiPk_n  / wjPk_n - wiPk
            }

            # Frank-Wolfe projections
            gnorm2 = 0
            for r, g in grad.items():
                result = linprog(
                    c = g,
                    A_ub = -(X * Yc[r]).T,
                    # b_ub = -np.ones(2**(N-1))*eps,
                    b_ub = np.zeros(2**(N-1)), # allow some boundaries while on lp "sphere"
                    # "sphere"
                    # A_eq = Wc[r],
                    A_eq = W_lp[r:r+1],
                    b_eq = np.array([Wc[r] @ W_lp[r]]),
                    bounds = (None, None),
                    method='simplex',
                    # method='highs-ipm',
                    # method='revised simplex', # this and high-ds miss some solutions
                )
                delta = result.x
                gnorm2 += (grad[r]**2).sum()
                Wc[r] = Wc[r] + step_scale * (delta - Wc[r]) # stays in interior as long as delta feasible and 0 <= step_scale <= 1

            gnorm2 /= len(grad.items())

            # # stop if infeasible (numerical issues when boundaries can be zero)
            # feasible = (np.sign(W @ X) == Y).all()
            # if not feasible:
            #     print("infeasible")
            #     break
    
            # check distance to feasible boundary
            extreme = np.fabs(Wc @ X).min()
    
            message = f"{update}/{num_updates}: loss={loss}, |grad|={gnorm2}, extremality={extreme}, lr={step_scale}"
            print(message)
            loss_curve.append(loss.item())
            extr_curve.append(extreme.item())
    
            np.set_printoptions(formatter = {'float': lambda x: "%+.3f" % x})
    
        with open(f"scg_ltm_{N}.pkl", "wb") as f:
            pk.dump((Wc, loss_curve, extr_curve), f)

    with open(f"scg_ltm_{N}.pkl", "rb") as f:
        (Wc, loss_curve, extr_curve) = pk.load(f)

    np.set_printoptions(formatter={'float': lambda x: "%+0.2f" % x})

    print("\n" + "*"*8 + " change " + "*"*8 + "\n")

    for i in range(len(Wc)):
        print(W_lp[i], Wc[i], np.fabs(Wc[i] @ X).min())

    print("\n   ** adj\n")

    print("wi, xk, wj, ab, resid, ijk")
    for (i, j, k) in A:
        ab = np.linalg.lstsq(np.vstack((Wc[i], X[:,k])).T, Wc[j], rcond=None)[0]
        resid = np.fabs(ab[0]*Wc[i] + ab[1]*X[:,k] - Wc[j]).max()
        print(Wc[i], X[:,k], Wc[j], ab, resid, i,j,k)

    pt.figure(figsize=(6,3))
    for do_log in (False, True):
        # pt.figure(figsize=(3,3))
        pt.subplot(2,2,1+do_log)
        pt.plot(loss_curve, 'k-')
        pt.ylabel("Span Loss")
        if do_log: pt.yscale('log')
        pt.subplot(2,2,3+do_log)
        pt.plot(extr_curve, 'k-')
        pt.ylabel("Constraint Slack")
        if do_log: pt.yscale('log')
        pt.xlabel("Optimization Step")
    pt.tight_layout()
    pt.savefig(f"scg_ltm_{N}.pdf")
    pt.show()




