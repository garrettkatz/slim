import pickle as pk
import numpy as np
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt
import matplotlib as mp
from numpy.linalg import norm
from numpy.polynomial import Polynomial
import scipy.sparse as sp

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

# @profile
def main():

    do_opt = True
    # do_opt = False

    # postfix = '' # exp
    # postfix = '_jaggi'
    # postfix = '_sqrt'
    postfix='_ana'
    
    N = 4 # dim
    eps = 0.1 # constraint slack threshold
    lr = .1 # learning rate
    decay = .99 # lr decay
    num_updates = 100

    N = 5 # dim
    eps = 0.1 # constraint slack threshold
    lr = 0.05 # learning rate
    decay = .99 # lr decay
    num_updates = 500

    N = 6 # dim
    eps = 0.1 # constraint slack threshold
    lr = 0.02 # learning rate
    decay = .995 # lr decay
    num_updates = 2000

    N = 7 # dim
    eps = 0.1 # constraint slack threshold
    lr = 0.005 # learning rate
    decay = .999 # lr decay
    num_updates = 5000

    # load canonical regions and adjacencies
    ltms = np.load(f"ltms_{N}_c.npz")
    Yc, Wc, X = ltms["Y"], ltms["W"], ltms["X"]
    with open(f"adjs_{N}_c.npz", "rb") as f:
        (Yn, Wn) = pk.load(f)

    # set up boundary indices to remove redundant region constraints
    Kn = {}
    for i in Yn:
        Kn[i] = (Yc[i] != Yn[i]).argmax(axis=1)

    # set up projection matrices
    PX = np.eye(N) - X.T.reshape(-1, N, 1) * X.T.reshape(-1, 1, N) / N

    # save original
    W_lp = Wc.copy()
    B_eq = (W_lp * W_lp).sum(axis=1) # 'norm' constraint

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

    if do_opt:

        # # make linprog coefs once for batch mode
        # A_eq = sp.block_diag([W_lp[r:r+1] for r in range(W_lp.shape[0])])
        # b_eq = B_eq
        # # faster with only boundary region constraints
        # A_ub = sp.block_diag([-(X[:,Kn[r]] * Yc[r, Kn[r]]).T for r in range(W_lp.shape[0])])
        # b_ub = np.full(A_ub.shape[0], -eps)

        # gradient update loop
        loss_curve = []
        sq_loss_curve = []
        extr_curve = []
        gn_curve = []
        pgn_curve = [] # projected gradient
        cos_curve = []
        for update in range(num_updates):

            # loss and gradient on joint-canonical adjacencies
            loss, sq_loss = 0., 0.
            max_1mc = 0
            grad = np.zeros(Wc.shape)
            sq_grad = np.zeros(Wc.shape)
            for (i,j,k) in Ac:

                # get projected weights
                wi, wj, xk = Wc[i], Wc[j], X[:,k]
                Pk = PX[k]
                wiPk = wi @ Pk
                wjPk = wj @ Pk
                wiPkwj = wiPk @ wjPk
                wjPkwj = wjPk @ wjPk
                wiPk_n, wjPk_n = norm(wiPk), norm(wjPk)

                # check minimum cosine
                max_1mc = max(max_1mc, 1. - wiPkwj / (wiPk_n * wjPk_n))
    
                # accumulate span loss
                loss += wiPk_n*wjPk_n - wiPk @ wjPk
    
                # accumulate gradient
                grad[i] += 2 * (wiPk * wjPk_n / wiPk_n - wjPk)

                # accumulate span loss
                sq_loss += (wiPk @ wiPk)*(wjPk @ wjPk) - wiPkwj**2 # sq version
    
                # accumulate gradient
                sq_grad[i] += 4 * (wiPk * wjPkwj - wiPkwj * wjPk) # sq version

            gn_curve.append(norm(sq_grad.flatten()))
            cos_curve.append(max_1mc)
            loss_curve.append(loss)
            sq_loss_curve.append(sq_loss)

            # Frank-Wolfe projections

            # # batch solve for delta
            # result = linprog(
            #     c = sq_grad.flatten(),
            #     A_ub = A_ub,
            #     b_ub = b_ub,
            #     A_eq = A_eq,
            #     b_eq = b_eq,
            #     bounds = (None, None),
            #     # method='simplex',
            #     # method='highs-ipm',
            #     # method='revised simplex', # this and high-ds miss some solutions, but needed for sparse
            # )
            # delta = result.x.reshape(W_lp.shape)

            # sequential
            delta = np.zeros(Wc.shape)
            for r in range(len(grad)):

                # norm constraints
                A_eq = W_lp[r:r+1]
                b_eq = B_eq[r:r+1]

                # # region<->weight invariance constraints
                # A_eq = np.concatenate((A_eq, A_sym[r]), axis=0)
                # b_eq = np.concatenate((b_eq, np.zeros(A_sym[r].shape[0])))

                # solve for delta
                result = linprog(
                    c = sq_grad[r],

                    # faster with only boundary region constraints
                    A_ub = -(X[:,Kn[r]] * Yc[r, Kn[r]]).T,
                    b_ub = -np.ones(len(Kn[r]))*eps,

                    A_eq = A_eq,
                    b_eq = b_eq,

                    bounds = (None, None),
                    method='simplex',
                    # method='highs-ipm',
                    # method='revised simplex', # this and high-ds miss some solutions
                )

                # save feasible descent direction
                delta[r] = result.x

            # calculate step scaling
            if postfix == '':
                step_scale = lr * decay**update

            elif postfix == '_jaggi':
                step_scale = 2 / (update + 2) # frank-wolfe default, maybe only for convex opt?

            elif postfix == '_sqrt':
                step_scale = .5 / (update + 1)**.5

            elif postfix == '_ana':

                cubic = [0, 0, 0, 0]
                step_grad = np.zeros(grad.shape)
                for (i,j,k) in Ac:

                    # get projections
                    di, dj, wi, wj, xk = delta[i], delta[j], Wc[i], Wc[j], X[:,k]
                    Pk = PX[k]
                    wiPk = wi @ Pk
                    wjPk = wj @ Pk
                    diPk = di @ Pk
                    djPk = dj @ Pk

                    # accumulate span grad at step scale == 1
                    vi, vj = wiPk + diPk, wjPk + djPk
                    step_grad[i] += 4 * (vi * (vj @ vj) - (vi @ vj) * vj) # have to use sq version

                    # accumulate cubic coefficients
                    cubic[0] += (wjPk @ wjPk)*(wiPk @ diPk) - (wiPk @ wjPk)*(wjPk @ diPk)
                    cubic[1] += 2*(wjPk @ djPk)*(wiPk @ diPk) + (wjPk @ wjPk)*(diPk @ diPk) \
                                - (wiPk @ wjPk)*(djPk @ diPk) - (wjPk @ diPk)*(diPk @ wjPk + wiPk @ djPk)
                    cubic[2] += 2*(wjPk @ djPk)*(diPk @ diPk) + (wiPk @ diPk)*(djPk @ djPk) \
                                - (wjPk @ diPk)*(diPk @ djPk) - (djPk @ diPk)*(diPk @ wjPk + wiPk @ djPk)
                    cubic[3] += (djPk @ djPk)*(diPk @ diPk) - (diPk @ djPk)*(djPk @ diPk)

                cubic = [4*cub for cub in cubic]

                # sanity check
                assert np.allclose(cubic[0], (sq_grad * delta).sum())
                assert np.allclose(sum(cubic), (step_grad * delta).sum())

                # find min value of any real roots in [0, 1]
                cubic = Polynomial(cubic)
                quartic = cubic.integ()
                roots = cubic.roots()
                gammas = np.append(roots.real, (0., 1.))
                gammas = gammas[(0 <= gammas) & (gammas <= 1)]
                vals = gammas[:,np.newaxis]**np.arange(5) @ quartic.coef
                step_scale = gammas[vals.argmin()]
                if step_scale == 0.:
                    print(roots)
                    print(roots[:,np.newaxis]**np.arange(4) @ cubic.coef)
                    print(gammas)
                    print(vals)

            # take step
            step = delta - Wc
            Wc = Wc + step_scale * step # stays in interior as long as delta feasible and 0 <= step_scale <= 1

            # calculate norm of projected gradient
            pgnorm = np.fabs((sq_grad * step).sum()) / norm(step)
            pgn_curve.append(pgnorm)

            # stop if infeasible (numerical issues when boundaries can be zero)
            if eps > 0:
                feasible = (np.sign(Wc @ X) == Yc).all()
                if not feasible:
                    print("infeasible")
                    break
    
            # check distance to feasible boundary
            extreme = np.fabs(Wc @ X).min()

            message = f"{update}/{num_updates}: loss={loss}, 1-cos<={max_1mc}), ,|sqgrad|={norm(sq_grad.flatten())}, |psqgrad|={pgnorm**0.5}, extremality={extreme}, lr={step_scale}"
            print(message)
            extr_curve.append(extreme)

            # if extreme < eps:
            #     print("hit boundary")
            #     break

            if step_scale == 0:
                print("hit gamma=0 (L=0?)")
                break
    
            np.set_printoptions(formatter = {'float': lambda x: "%+.3f" % x})
    
        with open(f"sq_ccg_ltm_{N}{postfix}.pkl", "wb") as f:
            pk.dump((Wc, sq_loss_curve, loss_curve, extr_curve, gn_curve, pgn_curve, cos_curve), f)

    with open(f"sq_ccg_ltm_{N}{postfix}.pkl", "rb") as f:
        (Wc, sq_loss_curve, loss_curve, extr_curve, gn_curve, pgn_curve, cos_curve) = pk.load(f)

    np.set_printoptions(formatter={'float': lambda x: "%+0.2f" % x})

    if N < 6:
        print("\n" + "*"*8 + " change " + "*"*8 + "\n")
    
        for i in range(len(Wc)):
            print(W_lp[i], Wc[i], np.fabs(Wc[i] @ X).min())
    
        print("\n   ** adj\n")
    
        print("ab, wi, xk, wj, resid, ijk")
        for (i, j, k) in Ac:
            ab = np.linalg.lstsq(np.vstack((Wc[i], X[:,k])).T, Wc[j], rcond=None)[0]
            resid = np.fabs(ab[0]*Wc[i] + ab[1]*X[:,k] - Wc[j]).max()
            print(ab, Wc[i], X[:,k], Wc[j], resid, i,j,k)

    fig, axs = pt.subplots(4,2, figsize=(6,8))
    for do_log in (False, True):
        pt.sca(axs[0,int(do_log)])
        # pt.plot(loss_curve, 'k-')
        pt.plot(np.array(sq_loss_curve) / len(Ac), 'k-', label="squared")
        pt.plot(np.array(loss_curve) / len(Ac), 'k:', label="orig")
        pt.ylabel("Span Loss")
        if do_log: pt.yscale('log')
        pt.legend()

        pt.sca(axs[1,int(do_log)])
        pt.plot(cos_curve, 'k-')
        pt.ylabel("max 1 - cos")
        if do_log: pt.yscale('log')

        pt.sca(axs[2,int(do_log)])
        pt.plot(extr_curve, 'k-')
        pt.plot([0, len(extr_curve)], [eps, eps], 'k:')
        pt.ylabel("Constraint Slack")
        if do_log: pt.yscale('log')

        pt.sca(axs[3,int(do_log)])
        pt.plot(gn_curve, 'k:')
        pt.plot(pgn_curve, 'k-')
        pt.ylabel("Grad Norm")
        if do_log: pt.yscale('log')
        pt.xlabel("Optimization Step")
    pt.tight_layout()
    pt.savefig(f"sq_ccg_ltm_{N}.pdf")
    pt.show()


if __name__ == "__main__": main()


