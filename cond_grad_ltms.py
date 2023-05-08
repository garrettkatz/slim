import pickle as pk
import numpy as np
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt
import matplotlib as mp
from span_loss_derivatives import calc_derivatives

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
    # num_updates = 1000

    N = 5 # dim
    eps = 0.01 # constraint slack threshold
    lr = 0.1 # learning rate
    num_updates = 1000

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # # break symmetry
    # for i in range(len(W)):
    #     A_ub = -(X * Y[i]).T
    #     c = -A_ub.sum(axis=0)
    #     result = linprog(
    #         c = c,
    #         A_ub = A_ub,
    #         b_ub = -(1 + 0.01*np.random.rand(2**(N-1))),
    #         bounds = (None, None),
    #         method='simplex',
    #         # method='highs-ipm',
    #         # method='revised simplex', # this and high-ds miss some solutions
    #     )
    #     W[i] = result.x

    # get adjacencies
    sym = False
    A, K = adjacency(Y, sym)
    numcon = sum(map(len, A.values())) # number of constraints

    # save original
    W_lp = W.copy()

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

            # differentiate loss function
            loss, grad, _ = calc_derivatives(Y, W, X, A, K, sym)

            # Frank-Wolfe projections
            gnorm2 = 0
            for i in range(len(W)):
                result = linprog(
                    c = grad[i],
                    A_ub = -(X * Y[i]).T,
                    # b_ub = -np.ones(2**(N-1))*eps,
                    b_ub = np.zeros(2**(N-1)), # allow some boundaries while on lp "sphere"
                    # "sphere"
                    # A_eq = W[i],
                    A_eq = W_lp[i:i+1],
                    b_eq = np.array([W[i] @ W_lp[i]]),
                    bounds = (None, None),
                    method='simplex',
                    # method='highs-ipm',
                    # method='revised simplex', # this and high-ds miss some solutions
                )
                delta = result.x
                gnorm2 += (grad[i]**2).sum()
                W[i] = W[i] + step_scale * (delta - W[i]) # stays in interior as long as 0 <= step_scale <= 1

            gnorm2 /= len(W)

            # # stop if infeasible (numerical issues when boundaries can be zero)
            # feasible = (np.sign(W @ X) == Y).all()
            # if not feasible:
            #     print("infeasible")
            #     break
    
            # check distance to feasible boundary
            extreme = np.fabs(W @ X).min()
    
            message = f"{update}/{num_updates}: loss={loss}, |grad|={gnorm2}, extremality={extreme}, lr={step_scale}"
            print(message)
            loss_curve.append(loss.item())
            extr_curve.append(extreme.item())
    
            np.set_printoptions(formatter = {'float': lambda x: "%+.3f" % x})
    
        with open(f"result_grad_ltm_{N}.pkl", "wb") as f:
            pk.dump((W, loss_curve, extr_curve), f)

    with open(f"result_grad_ltm_{N}.pkl", "rb") as f:
        (W, loss_curve, extr_curve) = pk.load(f)

    print(np.concatenate((W_lp, W), axis=1)[:10])

    np.set_printoptions(formatter={'float': lambda x: "%+0.2f" % x})
    for i in A:
        for j, k in zip(A[i], K[i]):
            ab = np.linalg.lstsq(np.vstack((W[i], X[:,k])).T, W[j], rcond=None)[0]
            print(X[:,k], W[i], W[j], ab, i,j,k)

    print("\n" + "*"*8 + "canon w" + "*"*8 + "\n")

    A, K = adjacency(Y, sym=True) # bring back redundancies for any_in_canon

    canon = (np.sign(W @ X) == np.sign(np.sort(np.fabs(W), axis=1) @ X)).all(axis=1)
    canon = np.flatnonzero(canon)
    for i in canon:
        print(W_lp[i], W[i], len(A[i]), np.fabs(W[i] @ X).min())

    print("\n   ** adj\n")

    for i in canon:
        any_in_canon = False
        for j, k in zip(A[i], K[i]):
            if j not in canon: continue
            any_in_canon = True

            ab = np.linalg.lstsq(np.vstack((W[i], X[:,k])).T, W[j], rcond=None)[0]
            print(X[:,k], W[i], W[j], ab, i,j,k)

        assert any_in_canon

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
    pt.savefig(f"result_cond_grad_ltm_{N}.pdf")
    pt.show()


