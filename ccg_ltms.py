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
    # eps = 0.1 # constraint slack threshold
    # lr = .1 # learning rate
    # decay = .995 # lr decay
    # num_updates = 500

    # N = 5 # dim
    # eps = 0.01 # constraint slack threshold
    # lr = 0.05 # learning rate
    # decay = .99 # lr decay
    # num_updates = 500

    # N = 6 # dim
    # eps = 0.01 # constraint slack threshold
    # lr = 0.01 # learning rate
    # decay = .995 # lr decay
    # num_updates = 1000

    N = 7 # dim
    eps = 0.01 # constraint slack threshold
    lr = 0.02 # learning rate
    decay = .9995 # lr decay
    num_updates = 10000

    # load canonical regions and adjacencies
    ltms = np.load(f"ltms_{N}_c.npz")
    Yc, Wc, X = ltms["Y"], ltms["W"], ltms["X"]
    with open(f"adjs_{N}_c.npz", "rb") as f:
        (Yn, Wn) = pk.load(f)

    # set up boundary indices to remove redundant region constraints
    Kn = {}
    for i in Yn:
        Kn[i] = (Yc[i] != Yn[i]).argmax(axis=1)

    # save original
    W_lp = Wc.copy()

    # set up equality constraints for region<->weight invariance
    A_sym = []
    for i, w in enumerate(Wc):
        w = w.round().astype(int) # will not work past n=8
        A_sym_i = np.eye(N-1,N) - np.eye(N-1,N,k=1) # retain w[n] - w[n+1] == 0
        A_sym_i = A_sym_i[w[:-1] == w[1:]] # i.e. w[n] == w[n+1] as constraint
        if w[0] == 0: # zeros must stay so
            A_sym_i = np.concatenate((np.eye(1,N), A_sym_i), axis=0)
        A_sym.append(A_sym_i)

    # canonical adjacency matrix
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

    if True: # do training
    # if False: # just load results

        # gradient update loop
        loss_curve = []
        extr_curve = []
        gn_curve = []
        pgn_curve = [] # projected gradient
        for update in range(num_updates):

            # update step scale for next iter
            # step_scale = lr # N=4
            # step_scale = lr / (np.log(update+1) + 1) # N=5
            # step_scale = lr / (update + 1)**.5
            # step_scale = lr * 2 / (update + 2) # frank-wolfe default, maybe only for convex opt?
            step_scale = lr * decay**update

            # loss on joint-canonical adjacencies
            loss = 0
            grad = np.zeros(Wc.shape)
            for (i,j,k) in Ac:

                # get projected weights
                wi, wj, xk = Wc[i], Wc[j], X[:,k]
                Pk = np.eye(N) - xk.reshape((N, 1)) * xk / N
                wiPk = wi @ Pk
                wjPk = wj @ Pk
                wiPk_n, wjPk_n = norm(wiPk), norm(wjPk)
    
                # accumulate span loss
                loss += wiPk_n*wjPk_n - wiPk @ wjPk
    
                # accumulate gradient
                grad[i] += wiPk * wjPk_n / wiPk_n - wjPk
                grad[j] += wjPk * wiPk_n  / wjPk_n - wiPk

            gn_curve.append(norm(grad.flatten()))

            # Frank-Wolfe projections
            pgnorm = 0
            for r in range(len(grad)):

                # norm constraints
                A_eq = W_lp[r:r+1]
                b_eq = np.array([Wc[r] @ W_lp[r]])

                # # region<->weight invariance constraints
                # A_eq = np.concatenate((A_eq, A_sym[r]), axis=0)
                # b_eq = np.concatenate((b_eq, np.zeros(A_sym[r].shape[0])))

                # solve for delta
                result = linprog(
                    c = grad[r],

                    # faster with only boundary region constraints
                    A_ub = -(X[:,Kn[r]] * Yc[r, Kn[r]]).T,
                    b_ub = -np.ones(len(Kn[r]))*eps,

                    A_eq = A_eq,
                    b_eq = b_eq,

                    bounds = (None, None),
                    # method='simplex',
                    # method='highs-ipm',
                    # method='revised simplex', # this and high-ds miss some solutions
                )

                # take step
                delta = result.x
                step = delta - Wc[r]
                Wc[r] = Wc[r] + step_scale * step # stays in interior as long as delta feasible and 0 <= step_scale <= 1

                # calculate norm of projected gradient
                step /= norm(step)
                pgnorm += (grad[r] * step).sum()**2

            pgn_curve.append(pgnorm)

            # stop if infeasible (numerical issues when boundaries can be zero)
            if eps > 0:
                feasible = (np.sign(Wc @ X) == Yc).all()
                if not feasible:
                    print("infeasible")
                    break
    
            # check distance to feasible boundary
            extreme = np.fabs(Wc @ X).min()
    
            message = f"{update}/{num_updates}: loss={loss}, |grad|={norm(grad.flatten())}, |pgrad|={pgnorm**0.5}, extremality={extreme}, lr={step_scale}"
            print(message)
            loss_curve.append(loss.item())
            extr_curve.append(extreme.item())
    
            np.set_printoptions(formatter = {'float': lambda x: "%+.3f" % x})
    
        with open(f"ccg_ltm_{N}.pkl", "wb") as f:
            pk.dump((Wc, loss_curve, extr_curve, gn_curve, pgn_curve), f)

    with open(f"ccg_ltm_{N}.pkl", "rb") as f:
        (Wc, loss_curve, extr_curve, gn_curve, pgn_curve) = pk.load(f)

    np.set_printoptions(formatter={'float': lambda x: "%+0.2f" % x})

    if N < 6:
        print("\n" + "*"*8 + " change " + "*"*8 + "\n")
    
        for i in range(len(Wc)):
            print(W_lp[i], Wc[i], np.fabs(Wc[i] @ X).min())
    
        print("\n   ** adj\n")
    
        print("wi, xk, wj, ab, resid, ijk")
        for (i, j, k) in Ac:
            ab = np.linalg.lstsq(np.vstack((Wc[i], X[:,k])).T, Wc[j], rcond=None)[0]
            resid = np.fabs(ab[0]*Wc[i] + ab[1]*X[:,k] - Wc[j]).max()
            print(Wc[i], X[:,k], Wc[j], ab, resid, i,j,k)

    # somehow check if solution on joint canonical extends to full adjacency set (complicated)
    # if N < 5:
    #     ltms = np.load(f"ltms_{N}.npz")
    #     Y, W, X = ltms["Y"], ltms["W"], ltms["X"]
    #     A, K = adjacency(Y, sym=False)
    #     for i in range(len(A)):
    
    #         # find canonical equivalent for i
    #         sorter = np.argsort(np.fabs(W[i]))    
    #         for j, k in zip(A[i], K[i]):

    fig, axs = pt.subplots(3,2, figsize=(6,8))
    for do_log in (False, True):
        pt.sca(axs[0,int(do_log)])
        pt.plot(loss_curve, 'k-')
        pt.ylabel("Span Loss")
        if do_log: pt.yscale('log')
        pt.sca(axs[1,int(do_log)])
        pt.plot(extr_curve, 'k-')
        pt.plot([0, num_updates], [eps, eps], 'k:')
        pt.ylabel("Constraint Slack")
        if do_log: pt.yscale('log')
        pt.sca(axs[2,int(do_log)])
        pt.plot(gn_curve, 'k:')
        pt.plot(pgn_curve, 'k-')
        pt.ylabel("Grad Norm")
        if do_log: pt.yscale('log')
        pt.xlabel("Optimization Step")
    pt.tight_layout()
    pt.savefig(f"ccg_ltm_{N}.pdf")
    pt.show()




