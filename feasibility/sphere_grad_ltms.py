import pickle as pk
import numpy as np
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt
import matplotlib as mp
from cvxopt.solvers import qp, options
from cvxopt import matrix
from span_loss_derivatives import calc_derivatives

# from scipy.optimize import linprog, OptimizeWarning
# from scipy.linalg import LinAlgWarning
# import warnings
# warnings.filterwarnings("ignore", category=LinAlgWarning)
# warnings.filterwarnings("ignore", category=OptimizeWarning)

np.set_printoptions(threshold=10e6)
options['show_progress'] = False

mp.rcParams['font.family'] = 'serif'

if __name__ == "__main__":
    
    N = 4 # dim
    eps = 0.01 # constraint slack threshold
    lr = 0.1 # learning rate
    num_updates = 2000

    # N = 5 # dim
    # eps = 0.0001 # constraint slack threshold
    # lr = 0.01 # learning rate
    # num_updates = 2000

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # get adjacencies
    sym = False
    A, K = adjacency(Y, sym)
    numcon = sum(map(len, A.values())) # number of constraints

    # save original
    W_lp = W.copy()

    # monitor active constraints
    active_constraints = {}

    if True: # do training
    # if False: # just load results

        # gradient update loop
        loss_curve = []
        extr_curve = []
        for update in range(num_updates):
    
            # differentiate loss function
            loss, grad, _ = calc_derivatives(Y, W, X, A, K, sym)
            delta = -grad

            # don't change weight norms
            for i in range(len(W)):
                delta[i] -= (delta[i] @ W[i]) * W[i] / (W[i]**2).sum()

            # update and zero gradient for next iter
            # step_scale = lr # N=4
            # step_scale = lr / (np.log(update+1) + 1) # N=5
            step_scale = lr / (update+1)**.5

            W += delta * step_scale
    
            feasible_preproj = ((W @ X) * Y >= eps).all()
            if not feasible_preproj:
                # project back to constraint set
                # min |W - W'| s.t. W' in convex cone
                for i in range(len(W)):
                    result = qp(
                        P = matrix(np.eye(len(W[i]))),
                        q = matrix(-W[i]),
                        G = matrix(-(X * Y[i]).astype(float).T),
                        h = matrix(-np.ones(len(Y[i]))*eps),
                    )
                    W[i] = np.array(result['x']).flatten()
    
            # stop if infeasible
            feasible = (np.sign(W @ X) == Y).all()
            if not feasible:
                print("infeasible")
                break
    
            # check distance to feasible boundary
            extreme = np.fabs(W @ X).min()
    
            message = f"{update}/{num_updates}: loss={loss}, |delta|={(delta**2).sum()}, extremality={extreme}, lr={step_scale}"
            if feasible_preproj: message += " [no projection needed]"
            print(message)
            loss_curve.append(loss.item())
            extr_curve.append(extreme.item())
    
            np.set_printoptions(formatter = {'float': lambda x: "%+.3f" % x})
    
        with open(f"result_grad_ltm_{N}.pkl", "wb") as f:
            pk.dump((W, loss_curve, extr_curve), f)

    with open(f"result_grad_ltm_{N}.pkl", "rb") as f:
        (W, loss_curve, extr_curve) = pk.load(f)

    W = W.detach().numpy()
    X = X.numpy().astype(int)
    Y = Y.numpy().astype(int)

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

    # for do_log in (False, True):
    #     pt.figure(figsize=(3,3))
    #     pt.subplot(2,1,1)
    #     pt.plot(loss_curve, 'k-')
    #     pt.ylabel("Span Loss")
    #     if do_log: pt.yscale('log')
    #     pt.subplot(2,1,2)
    #     pt.plot(extr_curve, 'k-')
    #     pt.ylabel("Constraint Slack")
    #     if do_log: pt.yscale('log')
    #     pt.xlabel("Optimization Step")
    #     pt.tight_layout()
    #     pt.savefig(f"result_grad_ltm_{N}.pdf")
    #     pt.show()


