import pickle as pk
import torch as tr
import numpy as np
from adjacent_ltms import adjacency
import matplotlib.pyplot as pt
import matplotlib as mp
from cvxopt.solvers import qp, options
from cvxopt import matrix

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
    eps = 0.001 # constraint slack threshold
    lr = 0.1 # learning rate
    num_updates = 2000

    N = 5 # dim
    eps = 0.0001 # constraint slack threshold
    lr = 5. # learning rate
    num_updates = 2000

    ltms = np.load(f"ltms_{N}.npz")
    Y, W, X = ltms["Y"], ltms["W"], ltms["X"]

    # get adjacencies
    # A, K = adjacency(Y, sym=True)
    A, K = adjacency(Y, sym=False) # avoid redundancies
    numcon = sum(map(len, A.values())) # number of constraints

    # wrap in tensors for grad opt
    W_lp = W
    # W = (tr.tensor(W).float() + 0.01*tr.randn(W.shape)).requires_grad_()
    W = tr.tensor(W).float().requires_grad_()
    Y = tr.tensor(Y).float()
    X = tr.tensor(X).float()

    # # form projection matrices
    # # P[j] = Proj_X[:,j]
    # Xt = X.t()
    # P = tr.eye(N) - Xt.unsqueeze(1) * Xt.unsqueeze(2) / N

    if True: # do training
    # if False: # just load results

        # gradient update loop
        loss_curve = []
        extr_curve = []
        for update in range(num_updates):
    
            # differentiate loss function
            # losses = []
            loss = 0
            for i in A:
                for j, k in zip(A[i], K[i]):
                    Pi = W[i] - tr.dot(W[i], X[:,k])*X[:,k] / N
                    Pj = W[j] - tr.dot(W[j], X[:,k])*X[:,k] / N
                    pij = tr.dot(Pi, Pj)
                    pii = tr.dot(Pi, Pi)
                    pjj = tr.dot(Pj, Pj)
    
                    # pij = (tr.outer(W[i], W[j]) * P[k]).sum()
                    # pii = (tr.outer(W[i], W[i]) * P[k]).sum()
                    # pjj = (tr.outer(W[j], W[j]) * P[k]).sum()
    
                    # loss_ij = (pij**2 - pii*pjj)**2
                    loss_ij = pii*pjj - pij**2 # norm prod is always >= dot product
                    # losses.append(loss_ij)
    
                    loss_ij /= numcon
                    loss_ij.backward()
                    loss += loss_ij.detach()
    
            # loss = tr.stack(losses).sum()
            # loss = tr.stack(losses).mean()
            # loss.backward()
            delta = -W.grad
    
            # update and zero gradient for next iter
            step_scale = lr # N=4
            # step_scale = lr / (np.log(update+1) + 1) # N=5
            # step_scale = lr / (update+1)**.5
            W.data += delta * step_scale
            W.grad *= 0
    
            feasible_preproj = (tr.mm(W, X) * Y >= eps).all().item()
            if not feasible_preproj:
                # project back to constraint set
                # min |W - W'| s.t. W' in convex cone
                for i in range(len(W)):
                    result = qp(
                        P = matrix(np.eye(len(W[i]))),
                        q = matrix(-W[i].detach().numpy().astype(float)),
                        G = matrix(-(X * Y[i]).numpy().T.astype(float)),
                        h = matrix(-np.ones(len(Y[i]))*eps),
                    )
                    W.data[i] = tr.tensor(np.array(result['x']).flatten()).float()
    
            # stop if infeasible
            feasible = (tr.mm(W, X).sign() == Y).all().item()
            if not feasible:
                print("infeasible")
                break
    
            # check distance to feasible boundary
            extreme = tr.mm(W, X).abs().min()
    
            message = f"{update}/{num_updates}: loss={loss.item()}, extremality={extreme}, lr={step_scale}"
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
